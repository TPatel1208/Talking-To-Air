from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class User(BaseModel):
    id: str
    username: str
    password_hash: str
    created_at: datetime
    is_active: bool

