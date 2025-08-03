"""
Main entry point for Anomaly Detection Serve
Simplified launcher following PEP8, Pylint, MyPy standards
"""
import logging
import sys
from pathlib import Path

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the application."""
    try:
        # Ensure we're in the correct directory
        current_dir = Path(__file__).parent
        if not (current_dir / "adserving").exists():
            logger.error("adserving package not found in current directory")
            sys.exit(1)

        # Import and run service
        from adserving.service.main_service import AnomalyDetectionServe

        service = AnomalyDetectionServe()
        service.run()

    except ImportError as e:
        logger.error(f"Failed to import service components: {e}")
        logger.error("Ensure all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start Anomaly Detection Serve: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()