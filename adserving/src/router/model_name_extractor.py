"""
Model name extraction utilities with caching optimization
"""

import threading
from typing import Any, Dict, Optional


class ModelNameExtractor:
    """Extract model names from requests with caching optimization"""

    def __init__(self):
        # Caching for model name extraction optimization
        self._model_name_cache = {}
        self._fn_field_cache = {}
        self._cache_lock = threading.RLock()

    def extract_model_name(self, request: Dict[str, Any]) -> Optional[str]:
        """Extract model name from request using multiple patterns with caching
        optimization"""
        # Create cache key from request structure
        cache_key = self._create_cache_key(request)

        with self._cache_lock:
            if cache_key in self._model_name_cache:
                return self._model_name_cache[cache_key]

        # Optimized extraction logic
        model_name = self._extract_model_name_optimized(request)

        # Cache the result
        if model_name and cache_key:
            with self._cache_lock:
                # Limit cache size to prevent memory issues
                if len(self._model_name_cache) > 10000:
                    # Remove oldest entries (simple FIFO)
                    oldest_keys = list(self._model_name_cache.keys())[:1000]
                    for old_key in oldest_keys:
                        del self._model_name_cache[old_key]

                self._model_name_cache[cache_key] = model_name

        return model_name

    def _create_cache_key(self, request: Dict[str, Any]) -> Optional[str]:
        """Create cache key from request structure"""
        try:
            # Create a deterministic key from request structure
            key_parts = []

            # Add basic fields
            for field in ["ma_don_vi", "ma_bao_cao", "ma_tieu_chi", "fld_code"]:
                if field in request:
                    key_parts.append(f"{field}:{request[field]}")

            # Handle data structure
            if (
                "data" in request
                and isinstance(request["data"], list)
                and request["data"]
            ):
                first_item = request["data"][0]
                if "ma_tieu_chi" in first_item:
                    key_parts.append(f"data_ma_tieu_chi:{first_item['ma_tieu_chi']}")

                # Add FN field info
                fn_fields = [k for k in first_item.keys() if k.startswith("FN")]
                if fn_fields:
                    key_parts.append(
                        f"fn_fields:{sorted(fn_fields)[0]}"
                    )  # Use first FN field

            # Handle format field
            if "format" in request:
                key_parts.append(f"format:{request['format']}")

            return "|".join(key_parts) if key_parts else None

        except Exception:
            # If cache key creation fails, return None to skip caching
            return None

    def _extract_model_name_optimized(self, request: Dict[str, Any]) -> Optional[str]:
        """Optimized model name extraction logic"""
        # Handle current format with nested FN fields
        model_name = self._handle_current_format(request)
        if model_name:
            return model_name

        # Handle direct model specification
        model_name = self._handle_direct_model(request)
        if model_name:
            return model_name

        # Handle processed format
        model_name = self._handle_processed_format(request)
        if model_name:
            return model_name

        # Fallback to direct model_name specification
        return request.get("model_name")

    def _handle_current_format(self, request: Dict[str, Any]) -> Optional[str]:
        if all(key in request for key in ["ma_don_vi", "ma_bao_cao", "data"]):
            ma_don_vi = request.get("ma_don_vi", "")
            ma_bao_cao = request.get("ma_bao_cao", "")
            data_list = request.get("data", [])

            if data_list and isinstance(data_list, list):
                first_item = data_list[0]
                ma_tieu_chi = first_item.get("ma_tieu_chi", "")

                fn_field = self._get_first_fn_field_cached(first_item)
                if fn_field:
                    normalized_fn = self._normalize_field_code(fn_field)
                    return f"{ma_don_vi}_{ma_bao_cao}_{ma_tieu_chi}_{normalized_fn}"
        return None

    def _handle_direct_model(self, request: Dict[str, Any]) -> Optional[str]:
        required_keys = ["ma_don_vi", "ma_bao_cao", "ma_tieu_chi", "fld_code"]
        if all(key in request for key in required_keys):
            return self._create_lookup_key(
                request["ma_don_vi"],
                request["ma_bao_cao"],
                request["ma_tieu_chi"],
                request["fld_code"],
            )
        return None

    def _handle_processed_format(self, request: Dict[str, Any]) -> Optional[str]:
        if "format" in request:
            if request["format"] == "direct" and "model_name" in request:
                return request["model_name"]
            elif request["format"] == "v2" and "data" in request and request["data"]:
                return request["data"][0].get("model_name")
            elif (
                request["format"] in ["enhanced_anomaly", "new_data"]
                and "criteria" in request
                and request["criteria"]
            ):
                return request["criteria"][0].get("model_name")
        return None

    def _get_first_fn_field_cached(self, item: Dict[str, Any]) -> Optional[str]:
        """Get first FN field with caching optimization"""
        # Create efficient hash-based cache key
        item_keys = hash(frozenset(item.keys()))

        with self._cache_lock:
            if item_keys in self._fn_field_cache:
                return self._fn_field_cache[item_keys]

        # Find first FN field
        fn_field = None
        for key in item:
            if key.startswith("FN"):
                fn_field = key
                break

        # Cache the result with LRU eviction
        with self._cache_lock:
            # Limit cache size with more efficient eviction
            if len(self._fn_field_cache) >= 5000:
                # Remove 10% of entries (500) to avoid frequent evictions
                keys_to_remove = list(self._fn_field_cache.keys())[:500]
                for key in keys_to_remove:
                    self._fn_field_cache.pop(key, None)

            self._fn_field_cache[item_keys] = fn_field

        return fn_field

    def _create_lookup_key(
        self, ma_don_vi: str, ma_bao_cao: str, ma_tieu_chi: str, fld_code: str
    ) -> str:
        """Create lookup key for model retrieval (similar to
        enhanced_model_manager)"""
        normalized_fld = self._normalize_field_code(fld_code)
        return f"{ma_don_vi}_{ma_bao_cao}_{ma_tieu_chi}_{normalized_fld}"

    def _normalize_field_code(self, field_name: str) -> str:
        """Normalize field code"""
        if not field_name.upper().startswith("FN"):
            return field_name.upper()

        # Extract number part
        number_part = field_name[2:]
        if number_part.isdigit():
            # Pad with zero if single digit
            if len(number_part) == 1:
                return f"FN0{number_part}"
            else:
                return f"FN{number_part}"

        return field_name.upper()
