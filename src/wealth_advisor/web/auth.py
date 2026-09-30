"""JWT authentication, password hashing, and RBAC dependencies for the adviser console."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from wealth_advisor.engine.database import UserModel, get_db

SECRET_KEY = "SUPER_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 8

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Password helpers ───────────────────────────────────────────────────────────


def verify_password(plain: str, hashed: str) -> bool:
    return bool(_pwd_context.verify(plain, hashed))


def get_password_hash(password: str) -> str:
    return str(_pwd_context.hash(password))


# ── Token helpers ──────────────────────────────────────────────────────────────


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(UTC) + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode["exp"] = expire
    return str(jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM))


# ── FastAPI dependencies ───────────────────────────────────────────────────────


def get_current_user(request: Request, db: Session = Depends(get_db)) -> UserModel:
    """Read the JWT from the `access_token` cookie and return the matching user row.

    Raises a 303 redirect to /login if the token is missing, invalid, or expired.
    """
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(
            status_code=303,
            headers={"Location": "/login"},
            detail="Not authenticated",
        )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = str(payload.get("sub", ""))
        if not username:
            raise ValueError("empty sub")
    except Exception:
        raise HTTPException(
            status_code=303,
            headers={"Location": "/login"},
            detail="Invalid token",
        )
    user = db.query(UserModel).filter(UserModel.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=303,
            headers={"Location": "/login"},
            detail="User not found",
        )
    return user


def RequireRole(allowed_roles: list[str]) -> Any:
    """Return a FastAPI dependency that gates access to the given roles."""

    def dependency(user: UserModel = Depends(get_current_user)) -> UserModel:
        if user.role not in allowed_roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user

    return dependency
