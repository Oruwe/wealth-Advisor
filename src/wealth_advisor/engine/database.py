"""SQLAlchemy database layer: models, engine, session factory, and schema initialisation.

Tables are created on first startup via init_db().  The DATABASE_URL env var is read at
import time so it can be overridden by the Docker env_file before the module is loaded.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/wealth",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


class UserModel(Base):
    """Console user — either a CCO (full access) or an ADVISOR (own clients only)."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False, default="ADVISOR")  # 'CCO' | 'ADVISOR' | 'CLIENT'
    linked_client_id = Column(String, nullable=True)  # set for CLIENT role users


class ClientModel(Base):
    """Thin client record — id, display name, primary tax jurisdiction, and owning adviser."""

    __tablename__ = "clients"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, default="")
    jurisdiction = Column(String, nullable=False, default="US")
    advisor_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    target_risk_aversion = Column(Float, nullable=False, default=1.0)
    max_equity_cap = Column(Float, nullable=False, default=1.0)


class TaxLotModel(Base):
    """One cost lot belonging to a client, as uploaded from a custodian CSV."""

    __tablename__ = "tax_lots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String, ForeignKey("clients.id"), nullable=False, index=True)
    symbol = Column(String, nullable=False)
    shares = Column(Numeric(precision=20, scale=8), nullable=False)
    cost_basis = Column(Numeric(precision=20, scale=2), nullable=False)
    purchase_date = Column(Date, nullable=False)
    asset_jurisdiction = Column(String, nullable=False, default="US")


class AlertModel(Base):
    """System-generated alert for a client — drift warnings, LRS warnings, etc."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String, ForeignKey("clients.id"), nullable=False, index=True)
    advisor_id = Column(Integer, nullable=False, default=0, index=True)
    alert_type = Column(String, nullable=False)  # 'DRIFT_WARNING' | 'LRS_WARNING'
    message = Column(String, nullable=False)
    is_resolved = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=lambda: datetime.now(UTC))


# ── Session dependency ─────────────────────────────────────────────────────────


def get_db() -> Generator[Session, None, None]:
    """FastAPI / plain-Python dependency: yield a DB session and close it on exit."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# ── Schema initialisation & seeding ───────────────────────────────────────────


def init_db() -> None:
    """Create all tables that do not yet exist, then seed default users if absent.

    The import of get_password_hash is deferred inside the function body to break
    the otherwise-circular import between database ↔ auth.
    """
    # Deferred import: auth imports UserModel from here, so we cannot import auth at
    # module level without creating a circular dependency.
    from wealth_advisor.web.auth import get_password_hash  # noqa: PLC0415

    Base.metadata.create_all(bind=engine)

    with SessionLocal() as session:
        if session.query(UserModel).count() == 0:
            session.add(UserModel(
                username="admin",
                hashed_password=get_password_hash("admin"),
                role="CCO",
            ))
            session.add(UserModel(
                username="advisor",
                hashed_password=get_password_hash("advisor"),
                role="ADVISOR",
            ))
            session.add(UserModel(
                username="client",
                hashed_password=get_password_hash("client"),
                role="CLIENT",
                linked_client_id="C-1049",
            ))
            session.commit()
