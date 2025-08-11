# Python
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from adserving.src.config.config import get_config
from adserving.src.utils.logger import get_logger

logger = get_logger()


def make_pg_engine() -> Engine:
    cfg = get_config()
    raw = getattr(cfg, "raw", {}) if hasattr(cfg, "raw") else {}
    db = raw.get("database", {}) or {}
    user = db.get("user")
    pwd = db.get("password")
    host = db.get("host", "127.0.0.1")
    port = int(db.get("port", 5432))
    name = db.get("name")

    # Cần driver psycopg2, nếu thiếu sẽ ném lỗi khi sử dụng.
    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
    min_con = int(db.get("min_connections", 5))
    max_con = int(db.get("max_connections", 20))
    engine = create_engine(
        url,
        pool_size=min_con,
        max_overflow=max(0, max_con - min_con),
        pool_pre_ping=True,
    )
    logger.info("PostgreSQL engine created.")
    return engine
