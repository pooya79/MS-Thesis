from datetime import datetime

from sqlmodel import Field, SQLModel


class Finding(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    title: str = Field(index=True, max_length=255)
    summary: str | None = Field(default=None, max_length=2000)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
