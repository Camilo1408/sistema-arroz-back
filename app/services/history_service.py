# app/services/history_service.py
# ================================================================
# SERVICIO DE HISTORIAL PERSISTENTE
# ================================================================
# Encapsula todas las operaciones de base de datos.
# Los endpoints solo llaman funciones de este servicio.
# Si en el futuro cambias la BD, solo tocas este archivo.
# ================================================================

import json
import logging
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis import AnalysisRecord
from app.schemas.responses import Alert

logger = logging.getLogger(__name__)


async def save_analysis(
    db: AsyncSession,
    subsystem: str,
    latency_ms: float,
    alert_level: str,
    alerts: List[Alert],
    indicators: dict,
) -> AnalysisRecord:
    """
    Guarda un nuevo análisis en la base de datos.
    Se llama automáticamente después de cada inferencia exitosa.
    """
    # Serializar alertas a JSON string para guardar en SQLite/PostgreSQL
    alerts_data = []
    for a in alerts:
        if hasattr(a, 'model_dump'):
            alerts_data.append(a.model_dump())
        elif isinstance(a, dict):
            alerts_data.append(a)

    record = AnalysisRecord(
        subsystem=subsystem,
        timestamp=datetime.now(timezone.utc),
        latency_ms=round(latency_ms, 2),
        alert_level=alert_level,
        alert_count=len(alerts),
        alerts_json=json.dumps(alerts_data),
    )

    # Rellena campos específicos según el subsistema
    if subsystem == "trilla":
        record.s2_intact_pct     = indicators.get("intact_grain_pct")
        record.s2_broken_pct     = indicators.get("broken_grain_pct")
        record.s2_straw_pct      = indicators.get("straw_pct")
        record.s2_overload       = indicators.get("overload_detected")

    elif subsystem == "limpieza":
        record.s3_intact_pct        = indicators.get("intact_grain_pct")
        record.s3_broken_pct        = indicators.get("broken_grain_pct")
        record.s3_non_grain_pct     = indicators.get("non_grain_pct")
        record.s3_total_detections  = indicators.get("total_detections")

    elif subsystem == "corte":
        # S1 pendiente — guarda lo que haya disponible
        record.s1_panicle_count    = indicators.get("panicle_count")
        record.s1_lodging_detected = indicators.get("lodging_detected")
        record.s1_density_level    = indicators.get("panicle_density")

    db.add(record)
    await db.flush()  # Obtiene el ID sin hacer commit todavía
    await db.refresh(record)

    logger.info(f"Historial guardado: id={record.id} subsistema={subsystem}")
    return record


async def get_history(
    db: AsyncSession,
    subsystem: Optional[str] = None,
    limit: int = 30,
    offset: int = 0,
) -> List[AnalysisRecord]:
    """
    Retorna registros del historial, ordenados del más reciente al más antiguo.
    Si se especifica `subsystem`, filtra por ese subsistema.
    """
    stmt = select(AnalysisRecord).order_by(AnalysisRecord.timestamp.desc())

    if subsystem:
        stmt = stmt.where(AnalysisRecord.subsystem == subsystem)

    stmt = stmt.limit(limit).offset(offset)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_by_id(
    db: AsyncSession,
    record_id: int,
) -> Optional[AnalysisRecord]:
    """Retorna un registro específico por su ID."""
    result = await db.execute(
        select(AnalysisRecord).where(AnalysisRecord.id == record_id)
    )
    return result.scalar_one_or_none()


async def delete_record(
    db: AsyncSession,
    record_id: int,
) -> bool:
    """Elimina un registro por ID. Retorna True si existía."""
    record = await get_by_id(db, record_id)
    if not record:
        return False
    await db.delete(record)
    return True


async def get_total_count(
    db: AsyncSession,
    subsystem: Optional[str] = None,
) -> int:
    """Retorna el número total de registros (para paginación)."""
    stmt = select(func.count(AnalysisRecord.id))
    if subsystem:
        stmt = stmt.where(AnalysisRecord.subsystem == subsystem)
    result = await db.execute(stmt)
    return result.scalar() or 0


async def get_summary_stats(
    db: AsyncSession,
    subsystem: str,
) -> dict:
    """
    Calcula estadísticas resumidas del historial de un subsistema.
    Usado por el Dashboard General para mostrar promedios históricos.
    """
    records = await get_history(db, subsystem=subsystem, limit=100)
    if not records:
        return {"count": 0}

    count = len(records)
    critical_count = sum(1 for r in records if r.alert_level == "CRITICO")

    stats = {"count": count, "critical_count": critical_count}

    if subsystem == "trilla":
        broken_values = [r.s2_broken_pct for r in records if r.s2_broken_pct is not None]
        if broken_values:
            stats["avg_broken_pct"] = round(sum(broken_values) / len(broken_values), 2)
            stats["max_broken_pct"] = round(max(broken_values), 2)

    elif subsystem == "limpieza":
        ng_values = [r.s3_non_grain_pct for r in records if r.s3_non_grain_pct is not None]
        if ng_values:
            stats["avg_non_grain_pct"] = round(sum(ng_values) / len(ng_values), 2)
            stats["max_non_grain_pct"] = round(max(ng_values), 2)

    return stats