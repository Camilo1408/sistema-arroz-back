# app/services/alert_service.py
# ================================================================
# SERVICIO DE ALERTAS OPERATIVAS
# ================================================================
# Evalúa los indicadores calculados por los modelos de IA
# contra los umbrales operativos definidos en .env y genera
# alertas estructuradas que el frontend puede mostrar.
# ================================================================

import uuid
from datetime import datetime, timezone
from typing import List

from app.core.config import settings
from app.schemas.responses import Alert, AlertLevel


def _make_alert(level: AlertLevel, message: str,
                action: str, subsystem: str) -> Alert:
    """Crea un objeto Alert con timestamp automático."""
    return Alert(
        id=str(uuid.uuid4())[:8],
        level=level,
        message=message,
        action=action,
        timestamp=datetime.now(timezone.utc).isoformat(),
        subsystem=subsystem,
    )


def evaluate_s2_alerts(indicators: dict) -> List[Alert]:
    """
    Evalúa los indicadores de S2 y genera alertas si corresponde.

    Umbrales (configurables desde .env):
    - broken_grain_pct < WARNING  → NORMAL  (sin alerta)
    - WARNING ≤ broken_grain_pct < CRITICAL → ATENCION
    - broken_grain_pct ≥ CRITICAL            → CRITICO
    """
    alerts = []
    broken = indicators.get("broken_grain_pct", 0.0)
    overload = indicators.get("overload_detected", False)

    if broken >= settings.S2_BROKEN_GRAIN_CRITICAL:
        alerts.append(_make_alert(
            level=AlertLevel.CRITICO,
            message=f"Grano roto supera umbral crítico: {broken:.2f}% (umbral: {settings.S2_BROKEN_GRAIN_CRITICAL}%)",
            action="Reducir velocidad del cilindro de trilla inmediatamente",
            subsystem="trilla",
        ))
    elif broken >= settings.S2_BROKEN_GRAIN_WARNING:
        alerts.append(_make_alert(
            level=AlertLevel.ATENCION,
            message=f"Grano roto cerca del umbral: {broken:.2f}% (advertencia: {settings.S2_BROKEN_GRAIN_WARNING}%)",
            action="Monitorear activamente el flujo de trilla",
            subsystem="trilla",
        ))

    if overload:
        alerts.append(_make_alert(
            level=AlertLevel.CRITICO,
            message="Flujo de trilla saturado detectado",
            action="Reducir velocidad de avance para disminuir carga de trilla",
            subsystem="trilla",
        ))

    return alerts


def evaluate_s3_alerts(indicators: dict) -> List[Alert]:
    """
    Evalúa los indicadores de S3 y genera alertas si corresponde.
    """
    alerts = []
    non_grain = indicators.get("non_grain_pct", 0.0)

    if non_grain >= settings.S3_NON_GRAIN_CRITICAL:
        alerts.append(_make_alert(
            level=AlertLevel.CRITICO,
            message=f"Material no-grano excede umbral: {non_grain:.1f}% (umbral: {settings.S3_NON_GRAIN_CRITICAL}%)",
            action="Verificar velocidad del ventilador y apertura de cribas",
            subsystem="limpieza",
        ))
    elif non_grain >= settings.S3_NON_GRAIN_WARNING:
        alerts.append(_make_alert(
            level=AlertLevel.ATENCION,
            message=f"Material no-grano cerca del umbral: {non_grain:.1f}%",
            action="Monitorear el sistema de limpieza activamente",
            subsystem="limpieza",
        ))

    return alerts


def get_alert_level(alerts: List[Alert]) -> AlertLevel:
    """Determina el nivel de alerta general a partir de una lista de alertas."""
    if any(a.level == AlertLevel.CRITICO for a in alerts):
        return AlertLevel.CRITICO
    if any(a.level == AlertLevel.ATENCION for a in alerts):
        return AlertLevel.ATENCION
    return AlertLevel.NORMAL