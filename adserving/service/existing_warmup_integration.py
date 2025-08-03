"""
Integration with existing warmup mechanisms
Leverages the built-in tiered loading and model warming
"""

import logging
from typing import Dict

from ..config.config import Config

logger = logging.getLogger(__name__)


class ExistingWarmupIntegration:
    """Integrates with existing warmup mechanisms"""

    def __init__(self, config: Config) -> None:
        """Initialize with existing warmup config."""
        self.config = config
        self.tiered_config = config.tiered_loading

    def ensure_warmup_enabled(self) -> None:
        """Ensure all existing warmup mechanisms are enabled."""
        logger.info("Configuring existing warmup mechanisms...")

        # Log current warmup configuration
        logger.info(
            f"Tiered loading enabled: {self.tiered_config.enable_tiered_loading}"
        )
        logger.info(f"Model warming enabled: {self.tiered_config.enable_model_warming}")
        logger.info(f"Hot cache size: {self.tiered_config.hot_cache_size}")
        logger.info(f"Warm cache size: {self.tiered_config.warm_cache_size}")
        logger.info(
            f"Warm popular models: {self.tiered_config.warm_popular_models_count}"
        )
        logger.info(f"Warming interval: {self.tiered_config.warming_interval}s")

        # Verify optimal settings for cold start prevention
        if not self.tiered_config.enable_model_warming:
            logger.warning(
                "Model warming is disabled - consider enabling for better performance"
            )

        if self.tiered_config.hot_cache_size < 10:
            logger.warning(
                "Hot cache size is small - consider increasing for better performance"
            )

    def get_warmup_status(self) -> Dict[str, bool]:
        """Get status of existing warmup mechanisms."""
        return {
            "tiered_loading_enabled": self.tiered_config.enable_tiered_loading,
            "model_warming_enabled": self.tiered_config.enable_model_warming,
            "has_hot_cache": self.tiered_config.hot_cache_size > 0,
            "has_warm_cache": self.tiered_config.warm_cache_size > 0,
        }
