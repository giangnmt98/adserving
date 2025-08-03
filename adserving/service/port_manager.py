"""
Port management utilities
Handles port availability checking and conflict resolution
"""
import logging
import socket

logger = logging.getLogger(__name__)


class PortManager:
    """Manages port availability and conflicts"""

    def is_port_available(self, host: str, port: int) -> bool:
        """Check if a port is available for binding."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
                return True
        except OSError:
            return False

    def find_available_port(self, start_port: int,
                           host: str = "0.0.0.0") -> int:
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + 100):
            if self.is_port_available(host, port):
                return port
        raise RuntimeError(f"No available ports found from {start_port}")

    def handle_port_conflict(self, host: str, port: int) -> None:
        """Handle port conflicts by logging detailed error info."""
        logger.error(f"Port {port} is already in use!")
        logger.error("=" * 70)
        logger.error("CRITICAL: Port Conflict Detected!")
        logger.error("=" * 70)
        logger.error(f"Port {port} on {host} is occupied by another service.")
        logger.error("")
        logger.error("Possible solutions:")
        logger.error("1. Stop the service using this port")
        logger.error("2. Change api_port in config.yaml")
        logger.error("3. Set MLOPS_PORT environment variable")
        logger.error("4. Use netstat to find the conflicting service:")
        logger.error(f"   Linux: netstat -tulpn | grep :{port}")
        logger.error(f"   Windows: netstat -an | findstr :{port}")
        logger.error("=" * 70)