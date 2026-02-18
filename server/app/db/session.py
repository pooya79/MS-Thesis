from typing import Generator

from sqlmodel import Session, create_engine

from server.app.core.config import get_settings

settings = get_settings()


def _create_engine():
    connect_args: dict[str, bool] = {}
    if settings.database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    return create_engine(settings.database_url, echo=False, connect_args=connect_args)


engine = _create_engine()


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
