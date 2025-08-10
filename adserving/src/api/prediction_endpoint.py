# Python
import math
import time
import uuid
import warnings
from typing import Any, Dict, List, Tuple, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from ray import serve

from adserving.src.datahandler.data_handler import DataHandler
from adserving.src.datahandler.models import APIResponse, PredictionRequest
from adserving.src.api.api_dependencies import (
    get_input_handler,
)
from adserving.src.deployment.request_processor import RequestProcessor
from adserving.src.utils.logger import get_logger

warnings.filterwarnings(
    "ignore", category=UserWarning, module="pydantic.type_adapter"
)

logger = get_logger()
router = APIRouter()

_HANDLE: Optional[Any] = None


async def _persist_details_to_db(details: List[Dict[str, Any]]) -> None:
    # TODO: triển khai lưu nền vào DB / queue
    pass


def _float_or_error(v: Any) -> Tuple[bool, Optional[float], str]:
    try:
        fv = float(v)
        if math.isnan(fv):
            return False, None, "Giá trị là NaN."
        return True, fv, ""
    except Exception:
        return False, None, "Giá trị không thể chuyển sang float."


def _get_handle() -> Any:
    global _HANDLE
    if _HANDLE is None:
        _HANDLE = serve.get_app_handle("preloaded_model_server")
    return _HANDLE


def _parse_model_name(model_name: Optional[str]) -> Tuple[str, str]:
    """
    Tách (ma_tieu_chi, fld_code) từ model_name: <ma_don_vi>_<ma_bao_cao>_<ma_tieu_chi>_<FNxx>
    """
    if not model_name or not isinstance(model_name, str):
        return "", ""
    parts = model_name.split("_")
    if len(parts) < 4:
        return "", ""
    fld_code = parts[-1]
    ma_tieu_chi = "_".join(parts[2:-1]) if len(parts) > 3 else ""
    return ma_tieu_chi, fld_code


@router.post("/predict", response_model=APIResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    handler: DataHandler = Depends(get_input_handler),
):
    """
    Response:
    {
      status,
      anomalies: [{ma_tieu_chi, list_anomaly}],
      failed: [{ma_tieu_chi, column, error_message}],
      (details: [...])  // tuỳ chọn, có thể bỏ để tối ưu latency
    }
    """
    req_id = str(uuid.uuid4())
    start = time.time()

    try:
        # Các trường top-level (ma_don_vi, ma_bao_cao, ky_du_lieu) đã được Pydantic validate:
        # nếu thiếu/sai kiểu -> FastAPI sẽ trả HTTP 400 ngay.
        raw_req: Dict[str, Any] = (
            request.model_dump() if hasattr(request, "model_dump") else dict(request)
        )

        # Xử lý/nắn lỗi validate ở cấp phần tử (data[..]) để trả về failed trong response
        processed = await handler.process_request(request)

        # Chuẩn hoá input thành prediction_tasks
        rp = RequestProcessor()
        try:
            prediction_tasks = rp.prepare_input_data(raw_req)
        except HTTPException:
            # Giữ nguyên HTTP 400 của các validate cứng (ví dụ thiếu/invalid ma_tieu_chi)
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Build tasks dạng float
        tasks: List[Dict[str, Any]] = []
        failed_local: List[Dict[str, Any]] = []
        for idx, t in enumerate(prediction_tasks):
            m = t.get("model_name")
            df = t.get("input_data")
            if m is None or df is None:
                mtc, fld = _parse_model_name(m)
                failed_local.append(
                    {
                        "element_index": idx,
                        "model_name": m,
                        "ma_tieu_chi": mtc,
                        "fld_code": fld,
                        "error": "invalid_task",
                        "error_details": "Thiếu model_name hoặc input_data.",
                    }
                )
                continue
            try:
                val = df["gia_tri"].iloc[0]
            except Exception:
                mtc, fld = _parse_model_name(m)
                failed_local.append(
                    {
                        "element_index": idx,
                        "model_name": m,
                        "ma_tieu_chi": mtc,
                        "fld_code": fld,
                        "error": "missing_value",
                        "error_details": "Không tìm thấy trường gia_tri.",
                    }
                )
                continue

            ok_val, fv, msg = _float_or_error(val)
            if not ok_val or fv is None:
                mtc, fld = _parse_model_name(m)
                failed_local.append(
                    {
                        "element_index": idx,
                        "model_name": m,
                        "ma_tieu_chi": mtc,
                        "fld_code": fld,
                        "error": "invalid_value",
                        "error_details": msg,
                    }
                )
                continue

            tasks.append({"model_name": str(m), "value": float(fv)})

        # Gọi Serve để lọc model + predict
        handle = _get_handle()
        result_obj = await handle.validate_and_predict.remote(tasks)
        details: List[Dict[str, Any]] = (result_obj or {}).get("results", [])
        failed_remote: List[Dict[str, Any]] = (result_obj or {}).get(
            "failed_elements", []
        )

        # Hợp nhất lỗi local và remote để tạo "failed" [{ma_tieu_chi, column, error_message}]
        failed_all_raw = failed_local + failed_remote

        def _infer_column(it: Dict[str, Any]) -> str:
            c = (it.get("fld_code") or "").strip()
            if c:
                return c
            mtc2, fld2 = _parse_model_name(it.get("model_name"))
            return fld2 or "UNKNOWN"

        failed: List[Dict[str, str]] = []
        for it in failed_all_raw:
            mtc = (it.get("ma_tieu_chi") or "").strip() or "UNKNOWN"
            col = _infer_column(it)
            msg = (
                it.get("error_details")
                or it.get("error_message")
                or it.get("error")
                or "unknown_error"
            )
            failed.append(
                {
                    "ma_tieu_chi": mtc,
                    "column": col,
                    "error_message": str(msg),
                }
            )

        # Gom anomalies theo ma_tieu_chi: chỉ giữ FNxx bất thường
        grouped_anomalies: Dict[str, set] = {}
        for item in details:
            if (
                item.get("status") == "success"
                and item.get("is_anomaly") is True
                and item.get("ma_tieu_chi")
                and item.get("fld_code")
            ):
                mtc = item["ma_tieu_chi"]
                fld = item["fld_code"]
                grouped_anomalies.setdefault(mtc, set()).add(fld)
        anomalies = [
            {"ma_tieu_chi": mtc, "list_anomaly": sorted(list(flds))}
            for mtc, flds in grouped_anomalies.items()
        ]

        total_time = time.time() - start
        status = (
            "success"
            if details and not failed
            else ("partial_success" if details else "error")
        )

        # Response gọn: anomalies + failed (details lưu nền)
        response_payload = {
            "status": status,
            "anomalies": anomalies,
            "failed": failed,  # [{ma_tieu_chi, column, error_message}]
            # "details": details,  # có thể bỏ để tối ưu latency
        }

        if details:
            print(details)
            background_tasks.add_task(_persist_details_to_db, details)


        return await handler.format_response(
            request_id=req_id,
            total_time=total_time,
            ma_don_vi=raw_req.get("ma_don_vi"),
            ma_bao_cao=raw_req.get("ma_bao_cao"),
            ky_du_lieu=raw_req.get("ky_du_lieu"),
            detailed_results=response_payload,
            validation_errors=processed.get("_validation_errors") or [],
        )

    except HTTPException:
        # Thiếu/sai kiểu các trường bắt buộc top-level/ma_tieu_chi sẽ vào đây (400)
        raise
    except Exception as e:
        total_time = time.time() - start
        raise HTTPException(status_code=500, detail=str(e))