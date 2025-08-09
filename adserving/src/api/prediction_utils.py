"""
Utility functions for prediction endpoint
"""

from typing import Dict


def extract_detailed_model_info(request_data: Dict) -> Dict:
    """Extract detailed information for model identification"""
    info = {
        "ma_don_vi": request_data.get("ma_don_vi", "UNKNOWN"),
        "ma_bao_cao": request_data.get("ma_bao_cao", "UNKNOWN"),
        "ky_du_lieu": request_data.get("ky_du_lieu", "UNKNOWN"),
        "data_elements": [],
    }

    data_list = request_data.get("data", [])
    for element in data_list:
        element_info = {
            "ma_tieu_chi": element.get("ma_tieu_chi", "UNKNOWN"),
            "fn_fields": [k for k in element.keys() if k.startswith("FN")],
            "fn_count": len([k for k in element.keys() if k.startswith("FN")]),
        }
        info["data_elements"].append(element_info)

    info["total_elements"] = len(data_list)
    info["is_single_element"] = len(data_list) == 1

    return info