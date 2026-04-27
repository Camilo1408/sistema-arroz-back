# app/api/routes.py
# ================================================================
# ENDPOINTS DE LA API REST
# ================================================================
# Todos los endpoints del sistema están aquí.
# Cada endpoint:
# 1. Recibe la petición HTTP del frontend
# 2. Llama a la función de inferencia correspondiente
# 3. Evalúa alertas con alert_service
# 4. Retorna el resultado formateado
# ================================================================

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
import io

from app.inference.s2_inference import run_s2_inference
from app.inference.s3_inference import run_s3_inference
from app.inference.s1_inference import run_s1_inference, S1NotAvailableError
from app.inference.model_loader import get_models_status
from app.services.alert_service import (
    evaluate_s2_alerts, evaluate_s3_alerts, get_alert_level
)
from app.services.export_service import generate_csv
from app.schemas.responses import (
    InferenceResponse, HealthResponse, SystemStatusResponse,
    S1Status, S2Status, S3Status, AlertLevel
)
from app.core.config import settings

# Agrega ESTAS líneas en la sección de imports de routes.py:
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from app.core.database import get_db
from app.services import history_service
from app.schemas.responses import AlertLevel

router = APIRouter()
logger = logging.getLogger(__name__)

# Historial en memoria (máximo 100 registros por subsistema)
# En producción reemplazar por base de datos SQLite/PostgreSQL
_history: dict = {"corte": [], "trilla": [], "limpieza": []}


# ================================================================
# HEALTH CHECK — GET /health
# ================================================================
@router.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health_check():
    """
    Verifica que el servidor está activo y qué modelos están cargados.
    Útil para confirmar que todo funciona antes de hacer inferencia.
    """
    models_status = get_models_status()
    return HealthResponse(
        status="ok",
        models_loaded=models_status,
        device=settings.DEVICE,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ================================================================
# ESTADO GENERAL — GET /api/v1/status
# ================================================================
@router.get("/status", response_model=SystemStatusResponse, tags=["Sistema"])
async def get_system_status():
    """
    Retorna el estado resumido de los tres subsistemas.
    El Dashboard General del frontend consume este endpoint.
    Se actualiza cada 30 segundos (el frontend hace polling).
    """
    models = get_models_status()

    # Estado de S2 basado en el último análisis
    last_s2 = _history["trilla"][-1] if _history["trilla"] else None
    s2_status = S2Status(
        status=last_s2.get("status", AlertLevel.NORMAL) if last_s2 else AlertLevel.NORMAL,
        last_latency_ms=last_s2.get("latency_ms", 0.0) if last_s2 else 0.0,
        available=models.get("s2", False),
        broken_grain_pct=last_s2.get("broken_grain_pct", 0.0) if last_s2 else 0.0,
        intact_grain_pct=last_s2.get("intact_grain_pct", 0.0) if last_s2 else 0.0,
    )

    # Estado de S3 basado en el último análisis
    last_s3 = _history["limpieza"][-1] if _history["limpieza"] else None
    s3_status = S3Status(
        status=last_s3.get("status", AlertLevel.NORMAL) if last_s3 else AlertLevel.NORMAL,
        last_latency_ms=last_s3.get("latency_ms", 0.0) if last_s3 else 0.0,
        available=models.get("s3", False),
        non_grain_pct=last_s3.get("non_grain_pct", 0.0) if last_s3 else 0.0,
        intact_grain_pct=last_s3.get("intact_grain_pct", 0.0) if last_s3 else 0.0,
    )

    # S1 aún no está disponible
    s1_status = S1Status(
        status=AlertLevel.NORMAL,
        last_latency_ms=0.0,
        available=False,
        last_panicle_count=0,
        lodging_detected=False,
    )

    # Recolecta alertas recientes de todos los subsistemas
    all_alerts = []
    for sub in ["corte", "trilla", "limpieza"]:
        for record in _history[sub][-5:]:  # Últimas 5 de cada uno
            all_alerts.extend(record.get("alerts", []))

    # Ordena por timestamp descendente y toma las 10 más recientes
    all_alerts.sort(key=lambda a: a.timestamp if hasattr(a, 'timestamp') else "", reverse=True)

    return SystemStatusResponse(
        s1=s1_status,
        s2=s2_status,
        s3=s3_status,
        recent_alerts=all_alerts[:10],
        session_start=_history["trilla"][0].get("timestamp", datetime.now(timezone.utc).isoformat())
        if _history["trilla"] else datetime.now(timezone.utc).isoformat(),
    )


# ================================================================
# INFERENCIA S2 — POST /api/v1/infer/trilla
# ================================================================
@router.post("/infer/trilla", tags=["Inferencia"])
async def infer_trilla(image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)):
    """
    Analiza una imagen de la zona de trilla.

    Recibe: imagen como multipart/form-data (campo "image")
    Retorna: JSON con indicadores de S2, mapa de segmentación y alertas

    El frontend envía la imagen con:
        const formData = new FormData();
        formData.append('image', file);
        fetch('/api/v1/infer/trilla', { method: 'POST', body: formData })
    """
    # ── Validar tipo de archivo ───────────────────────────────────
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Archivo no válido. Se esperaba imagen, se recibió: {image.content_type}"
        )

    # ── Leer bytes de la imagen ───────────────────────────────────
    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="La imagen está vacía")

    # ── Ejecutar inferencia ───────────────────────────────────────
    try:
        indicators, latency_ms = run_s2_inference(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # ── Evaluar alertas ───────────────────────────────────────────
    indicators_dict = indicators.model_dump()
    alerts = evaluate_s2_alerts(indicators_dict)
    overall_status = get_alert_level(alerts)

    # ── Guardar en historial ──────────────────────────────────────
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": latency_ms,
        "status": overall_status,
        "broken_grain_pct": indicators.broken_grain_pct,
        "intact_grain_pct": indicators.intact_grain_pct,
        "alerts": alerts,
        "alert_count": len(alerts),
    }
    _history["trilla"].append(record)
    if len(_history["trilla"]) > 100:
        _history["trilla"].pop(0)

    # ── Construir respuesta ───────────────────────────────────────
    response = {
        "subsystem": "trilla",
        "frame_id": datetime.now(timezone.utc).isoformat(),
        "latency_ms": round(latency_ms, 2),
        "detections": [],  # S2 usa segmentación, no bounding boxes individuales
        "indicators": indicators_dict,
        "alerts": [a.model_dump() for a in alerts],
    }

     # AGREGA ESTAS LÍNEAS justo antes del return:
    try:
        await history_service.save_analysis(
            db=db,
            subsystem="trilla",
            latency_ms=latency_ms,
            alert_level=overall_status.value,
            alerts=alerts,
            indicators=indicators_dict,
        )
    except Exception as e:
        logger.error(f"Error guardando historial S2: {e}")
        # No falla la petición si el historial falla

    return JSONResponse(content=response)


# ================================================================
# INFERENCIA S3 — POST /api/v1/infer/limpieza
# ================================================================
@router.post("/infer/limpieza", tags=["Inferencia"])
async def infer_limpieza(image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)):
    """
    Analiza una imagen del sistema de limpieza.

    Recibe: imagen como multipart/form-data (campo "image")
    Retorna: JSON con bounding boxes, indicadores de S3 y alertas
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Se esperaba una imagen")

    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="La imagen está vacía")

    try:
        indicators, detections, latency_ms = run_s3_inference(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    indicators_dict = indicators.model_dump()
    alerts = evaluate_s3_alerts(indicators_dict)
    overall_status = get_alert_level(alerts)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": latency_ms,
        "status": overall_status,
        "non_grain_pct": indicators.non_grain_pct,
        "intact_grain_pct": indicators.intact_grain_pct,
        "alerts": alerts,
        "alert_count": len(alerts),
    }
    _history["limpieza"].append(record)
    if len(_history["limpieza"]) > 100:
        _history["limpieza"].pop(0)

    response = {
        "subsystem": "limpieza",
        "frame_id": datetime.now(timezone.utc).isoformat(),
        "latency_ms": round(latency_ms, 2),
        "detections": detections,
        "indicators": indicators_dict,
        "alerts": [a.model_dump() for a in alerts],
    }

    # Agrega antes del return:
    try:
        await history_service.save_analysis(
            db=db,
            subsystem="limpieza",
            latency_ms=latency_ms,
            alert_level=overall_status.value,
            alerts=alerts,
            indicators=indicators_dict,
        )
    except Exception as e:
        logger.error(f"Error guardando historial S3: {e}")

    return JSONResponse(content=response)


# ================================================================
# INFERENCIA S1 — POST /api/v1/infer/corte (PLACEHOLDER)
# ================================================================
@router.post("/infer/corte", tags=["Inferencia"])
async def infer_corte(image: UploadFile = File(...)):
    """
    PENDIENTE — Análisis del cabezal de corte.
    Retorna 503 con mensaje explicativo hasta que S1 esté disponible.
    El endpoint ya existe y está documentado en Swagger.
    Cuando S1 esté listo, solo se modifica run_s1_inference().
    """
    image_bytes = await image.read()

    try:
        result = run_s1_inference(image_bytes)
        return JSONResponse(content=result)
    except S1NotAvailableError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "S1_NOT_AVAILABLE",
                "message": str(e),
                "subsystem": "corte",
            }
        )


# ================================================================
# HISTORIAL — GET /api/v1/history/{subsystem}
# ================================================================
@router.get("/history/{subsystem}", tags=["Historial"])
async def get_history(
    subsystem: str,
    limit: int = Query(default=30, ge=1, le=100),
):
    """
    Retorna el historial de análisis de un subsistema.
    """
    if subsystem not in ["corte", "trilla", "limpieza"]:
        raise HTTPException(
            status_code=400,
            detail=f"Subsistema no válido: {subsystem}. Use: corte, trilla, limpieza"
        )

    records = _history[subsystem][-limit:]
    
    # Serializar alertas a dict para JSON
    serialized = []
    for r in records:
        r_copy = dict(r)
        if "alerts" in r_copy:
            r_copy["alerts"] = [
                a.model_dump() if hasattr(a, 'model_dump') else a
                for a in r_copy["alerts"]
            ]
        serialized.append(r_copy)

    # ← IMPORTANTE: Retorna solo el array, no un objeto
    return serialized  # CAMBIO: era return JSONResponse(content={...})


# ================================================================
# EXPORTACIÓN CSV — GET /api/v1/history/export
# ================================================================
@router.get("/history/export", tags=["Historial"])
async def export_history(
    subsystem: str = Query(..., description="corte | trilla | limpieza")
):
    """
    Exporta el historial de un subsistema como archivo CSV.
    El frontend descarga el archivo automáticamente.
    """
    if subsystem not in _history:
        raise HTTPException(status_code=400, detail="Subsistema no válido")

    records = _history[subsystem]
    # Simplifica los registros para CSV (elimina objetos anidados)
    flat_records = []
    for r in records:
        flat = {k: v for k, v in r.items() if k != "alerts"}
        flat["alert_count"] = r.get("alert_count", 0)
        flat_records.append(flat)

    csv_content = generate_csv(flat_records)
    filename = f"diagnostico_{subsystem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

# ================================================================
# HISTORIAL PERSISTENTE — Nuevos endpoints
# ================================================================

@router.get("/history", tags=["Historial"])
async def get_all_history(
    subsystem: Optional[str] = None,
    limit: int = Query(default=30, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    Retorna el historial de análisis con paginación.
    - Sin parámetros: todos los subsistemas
    - ?subsystem=trilla: solo zona de trilla
    - ?limit=10&offset=20: paginación
    """
    records = await history_service.get_history(db, subsystem, limit, offset)
    total = await history_service.get_total_count(db, subsystem)

    return JSONResponse(content={
        "total": total,
        "limit": limit,
        "offset": offset,
        "records": [r.to_dict() for r in records],
    })


@router.get("/history/record/{record_id}", tags=["Historial"])
async def get_analysis_by_id(
    record_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Retorna un análisis específico por su ID numérico.
    Útil para ver detalles de un análisis en particular.
    """
    record = await history_service.get_by_id(db, record_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Análisis #{record_id} no encontrado")
    return JSONResponse(content=record.to_dict())


@router.delete("/history/record/{record_id}", tags=["Historial"])
async def delete_analysis(
    record_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Elimina un registro específico por su ID.
    Útil para limpiar análisis de prueba antes de demostración.
    """
    deleted = await history_service.delete_record(db, record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Análisis #{record_id} no encontrado")
    return JSONResponse(content={"deleted": True, "id": record_id})


@router.get("/history/stats/{subsystem}", tags=["Historial"])
async def get_subsystem_stats(
    subsystem: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Retorna estadísticas resumidas del historial de un subsistema.
    Usado por el Dashboard General para mostrar métricas históricas.
    """
    if subsystem not in ["corte", "trilla", "limpieza"]:
        raise HTTPException(status_code=400, detail="Subsistema no válido")

    stats = await history_service.get_summary_stats(db, subsystem)
    return JSONResponse(content={"subsystem": subsystem, **stats})


@router.get("/history/export", tags=["Historial"])
async def export_history_csv(
    subsystem: str = Query(..., description="corte | trilla | limpieza"),
    db: AsyncSession = Depends(get_db),
):
    """
    Exporta el historial como CSV (ahora desde BD real, no memoria).
    """
    from app.services.export_service import generate_csv
    import io

    records = await history_service.get_history(db, subsystem=subsystem, limit=1000)
    flat_records = []
    for r in records:
        d = r.to_dict()
        d.pop("alerts", None)  # No incluir JSON de alertas en CSV
        flat_records.append(d)

    csv_content = generate_csv(flat_records)
    filename = f"diagnostico_{subsystem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )