# Python
import math
import time
import uuid
import warnings
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from ray import serve

# NOTE: Import trực tiếp từ api_dependencies để tránh trùng module dẫn tới _ASYNC_RUNNER bị None
from adserving.src.api.api_dependencies import get_input_handler
from adserving.src.datahandler.data_handler import DataHandler
from adserving.src.datahandler.models import APIResponse, PredictionRequest
from adserving.src.deployment.request_processor import RequestProcessor
from adserving.src.utils.logger import get_logger

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.type_adapter")

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
    Tách (ma_tieu_chi, fld_code) từ model_name:
    <ma_don_vi>_<ma_bao_cao>_<ma_tieu_chi>_<FNxx>
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
    handler: DataHandler = Depends(get_input_handler),
):
    req_id = str(uuid.uuid4())
    start = time.time()

    try:
        raw_req = await _extract_raw_request(request)
        processed = await handler.process_request(request)
        prediction_tasks = await _prepare_prediction_tasks(raw_req)
        tasks, failed_local = await _build_float_tasks(prediction_tasks)
        details, failed_remote = await _get_prediction_results(tasks)
        failed = _process_failures(failed_local, failed_remote)
        anomalies = _process_anomalies(details)

        total_time = time.time() - start
        status = _determine_status(details, failed)

        response_payload = {
            "status": status,
            "anomalies": anomalies,
            "failed": failed,
        }


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
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _extract_raw_request(request: PredictionRequest) -> Dict[str, Any]:
    return request.model_dump() if hasattr(request, "model_dump") else dict(request)


async def _prepare_prediction_tasks(raw_req: Dict[str, Any]) -> List[Dict[str, Any]]:
    rp = RequestProcessor()
    try:
        return rp.prepare_input_data(raw_req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def _build_float_tasks(
    prediction_tasks: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tasks = []
    failed_local = []

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

    return tasks, failed_local


async def _get_prediction_results(
    tasks: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    handle = _get_handle()
    result_obj = await handle.validate_and_predict.remote(tasks)
    details = (result_obj or {}).get("results", [])
    failed_remote = (result_obj or {}).get("failed_elements", [])
    return details, failed_remote


def _process_failures(
    failed_local: List[Dict[str, Any]], failed_remote: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    failed_all_raw = failed_local + failed_remote
    failed = []

    for it in failed_all_raw:
        mtc = (it.get("ma_tieu_chi") or "").strip() or "UNKNOWN"
        col = it.get("fld_code", "UNKNOWN")
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

    return failed


def _process_anomalies(details: List[Dict[str, Any]]):
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
    return [
        {
            "ma_tieu_chi": str(mtc),
            "list_anomaly": (
                [str(flds)] if isinstance(flds, str) else sorted([str(x) for x in flds])
            ),
        }
        for mtc, flds in grouped_anomalies.items()
    ]


def _determine_status(
    details: List[Dict[str, Any]], failed: List[Dict[str, str]]
) -> str:
    if details and not failed:
        return "success"
    elif details:
        return "partial_success"
    return "error"