"""
Ray-Serializable Model Name Extractor

This module provides a Ray-serializable version of ModelNameExtractor that eliminates 
threading objects that cause pickle/serialization errors when deploying to Ray Serve.
"""

from typing import Any, Dict, Optional


class SerializableModelNameExtractor:
    """
    Ray-serializable model name extractor that can be safely passed to Ray Serve deployments.
    
    This class removes all threading objects (threading.RLock, etc.) that cause 
    Ray serialization errors and uses simple thread-unsafe alternatives.
    """

    def __init__(self):
        # Simple caches without threading locks (thread-unsafe but Ray-serializable)
        self._model_name_cache = {}
        self._fn_field_cache = {}
        # Note: No threading.RLock objects - these cause serialization errors

    def extract_model_name(self, request: Dict[str, Any]) -> Optional[str]:
        """Extract model name from request using multiple patterns with simple caching"""
        # Create cache key from request structure
        cache_key = self._create_cache_key(request)

        # Simple cache lookup without locks (thread-unsafe but Ray-serializable)
        if cache_key in self._model_name_cache:
            return self._model_name_cache[cache_key]

        # Optimized extraction logic
        model_name = self._extract_model_name_optimized(request)

        # Cache the result (simple cache management without locks)
        if model_name and cache_key:
            # Limit cache size to prevent memory issues
            if len(self._model_name_cache) > 10000:
                # Remove oldest entries (simple FIFO without locks)
                oldest_keys = list(self._model_name_cache.keys())[:1000]
                for old_key in oldest_keys:
                    self._model_name_cache.pop(old_key, None)

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
        """Get first FN field with simple caching (no locks)"""
        # Create efficient hash-based cache key
        item_keys = hash(frozenset(item.keys()))

        # Simple cache lookup without locks
        if item_keys in self._fn_field_cache:
            return self._fn_field_cache[item_keys]

        # Find first FN field
        fn_field = None
        for key in item:
            if key.startswith("FN"):
                fn_field = key
                break

        # Cache the result with simple eviction (no locks)
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
        """Create lookup key for model retrieval"""
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

    def clear_cache(self):
        """Clear all caches (useful for testing)"""
        self._model_name_cache.clear()
        self._fn_field_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            "model_name_cache_size": len(self._model_name_cache),
            "fn_field_cache_size": len(self._fn_field_cache),
            "threading_locks": False,  # Indicates no threading locks used
            "serializable": True,  # Indicates Ray-serializable
        }