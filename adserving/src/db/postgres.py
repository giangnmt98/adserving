# Python
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from adserving.src.config.config import get_config
from adserving.src.utils.logger import get_logger

logger = get_logger()


def make_pg_engine() -> Engine:
    """Creates and returns a SQLAlchemy PostgreSQL engine with connection pool.

    Returns:
        Engine: SQLAlchemy Engine instance configured for PostgreSQL.
    """

    cfg = get_config()
    database_cfg = cfg.database
    database_name = database_cfg.database_name
    user = database_cfg.username
    pwd = database_cfg.password
    host = database_cfg.host
    port = int(database_cfg.port)

    # Cần driver psycopg2, nếu thiếu sẽ ném lỗi khi sử dụng.
    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{database_name}"
    min_con = int(database_cfg.min_connections)
    max_con = int(database_cfg.max_connections)
    engine = create_engine(
        url,
        pool_size=min_con,
        max_overflow=max(0, max_con - min_con),
        pool_pre_ping=True,
    )
    logger.info("PostgreSQL engine created.")
    return engine
