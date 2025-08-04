"""
Request Context Extractor.

Trích xuất thông tin context từ request để hỗ trợ error handling.
"""

from typing import Dict, List, Any, Optional


class RequestContextExtractor:
    """Extract context information from requests for error handling."""

    @staticmethod
    def extract_request_context(processed_request: Dict) -> Dict[str, Any]:
        """
        Extract detailed context from processed request.

        Args:
            processed_request: Request đã được xử lý

        Returns:
            Dict chứa context information
        """
        try:
            context = {
                "ma_don_vi": processed_request.get("ma_don_vi", "UNKNOWN"),
                "ma_bao_cao": processed_request.get("ma_bao_cao", "UNKNOWN"),
                "ky_du_lieu": processed_request.get("ky_du_lieu", "UNKNOWN"),
                "data_elements": [],
                "total_elements": 0,
                "is_single_element": False,
            }

            # Extract data elements info
            data_list = processed_request.get("data", [])

            for element in data_list:
                element_info = {
                    "ma_tieu_chi": element.get("ma_tieu_chi", "UNKNOWN"),
                    "fn_fields": [k for k in element.keys() if k.startswith("FN")],
                    "fn_count": len([k for k in element.keys() if k.startswith("FN")]),
                    "all_fields": list(element.keys()),
                }
                context["data_elements"].append(element_info)

            context["total_elements"] = len(data_list)
            context["is_single_element"] = len(data_list) == 1

            return context

        except Exception as e:
            # Return minimal context if extraction fails
            return {
                "extraction_error": str(e),
                "total_elements": 0,
                "is_single_element": False,
                "data_elements": [],
            }

    @staticmethod
    def extract_model_name_context(
        model_name: Optional[str], processed_request: Dict
    ) -> Dict[str, Any]:
        """
        Extract context specific to model name extraction.

        Args:
            model_name: Model name (có thể None nếu extraction failed)
            processed_request: Request đã được xử lý

        Returns:
            Dict chứa model name context
        """
        context = RequestContextExtractor.extract_request_context(processed_request)

        context.update(
            {
                "extracted_model_name": model_name,
                "model_name_available": model_name is not None,
                "extraction_method": "automatic" if model_name else "failed",
            }
        )

        return context

    @staticmethod
    def generate_error_patterns(context: Dict[str, Any]) -> List[str]:
        """
        Generate possible model name patterns for debugging.

        Args:
            context: Request context

        Returns:
            List of possible model name patterns
        """
        patterns = []

        try:
            ma_don_vi = context.get("ma_don_vi", "")
            ma_bao_cao = context.get("ma_bao_cao", "")

            for element in context.get("data_elements", []):
                ma_tieu_chi = element.get("ma_tieu_chi", "")
                fn_count = element.get("fn_count", 0)

                if ma_don_vi and ma_bao_cao and ma_tieu_chi:
                    base_pattern = f"{ma_don_vi}_{ma_bao_cao}_{ma_tieu_chi}"
                    patterns.extend(
                        [
                            f"{base_pattern}_FN{fn_count:02d}",  # FN04
                            f"{base_pattern}_FN{fn_count}",  # FN4
                            f"{base_pattern}_{fn_count}F",  # 4F
                            f"{base_pattern}_fields_{fn_count}",  # fields_4
                            base_pattern,  # Base without field count
                        ]
                    )

        except Exception:
            patterns = ["Unable to generate patterns - check request format"]

        return patterns[:5]  # Limit to 5 patterns
