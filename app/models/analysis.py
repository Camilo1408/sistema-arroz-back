# app/models/analysis.py
# ================================================================
# MODELO DE LA TABLA "analysis_records"
# ================================================================
# Define la estructura de la tabla en Python.
# SQLAlchemy la traduce a SQL automáticamente.
# Compatible con SQLite (desarrollo) y PostgreSQL (producción).
# ================================================================

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import String, Float, Integer, Boolean, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class AnalysisRecord(Base):
    """
    Representa un análisis de frame realizado por cualquier subsistema.

    Campos nullable: Los campos específicos de cada subsistema son
    opcionales (NULL) cuando el registro pertenece a otro subsistema.
    Por ejemplo, un análisis de S2 tendrá s3_non_grain_pct = NULL.
    """
    __tablename__ = "analysis_records"

    # ── Campos generales (SIEMPRE presentes) ─────────────────────
    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    subsystem: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True
        # "corte" | "trilla" | "limpieza"
        # index=True porque filtraremos por subsistema frecuentemente
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    alert_level: Mapped[str] = mapped_column(
        String(10), nullable=False, default="NORMAL"
        # "NORMAL" | "ATENCION" | "CRITICO"
    )
    alert_count: Mapped[int] = mapped_column(Integer, default=0)
    alerts_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
        # JSON string con las alertas generadas
    )

    # ── Campos S2 — Zona de Trilla (nullable) ─────────────────────
    s2_intact_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    s2_broken_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    s2_straw_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    s2_overload: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # ── Campos S3 — Sistema de Limpieza (nullable) ────────────────
    s3_intact_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    s3_broken_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    s3_non_grain_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    s3_total_detections: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # ── Campos S1 — Cabezal de Corte (nullable, pendiente) ────────
    s1_panicle_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    s1_lodging_detected: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    s1_density_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    def to_dict(self) -> dict:
        """Convierte el registro a diccionario para respuesta JSON."""
        import json
        return {
            "id": self.id,
            "subsystem": self.subsystem,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "latency_ms": self.latency_ms,
            "alert_level": self.alert_level,
            "alert_count": self.alert_count,
            "alerts": json.loads(self.alerts_json) if self.alerts_json else [],
            # S2
            "s2_intact_pct": self.s2_intact_pct,
            "s2_broken_pct": self.s2_broken_pct,
            "s2_straw_pct": self.s2_straw_pct,
            "s2_overload": self.s2_overload,
            # S3
            "s3_intact_pct": self.s3_intact_pct,
            "s3_broken_pct": self.s3_broken_pct,
            "s3_non_grain_pct": self.s3_non_grain_pct,
            "s3_total_detections": self.s3_total_detections,
            # S1
            "s1_panicle_count": self.s1_panicle_count,
            "s1_lodging_detected": self.s1_lodging_detected,
            "s1_density_level": self.s1_density_level,
        }