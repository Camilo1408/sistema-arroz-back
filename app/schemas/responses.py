# app/schemas/responses.py
# ================================================================
# ESTRUCTURAS DE DATOS DE LA API
# ================================================================
# Define exactamente qué formato tienen las respuestas JSON.
# El frontend TypeScript espera exactamente estos campos con
# exactamente estos nombres.

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime


class AlertLevel(str, Enum):
    """Niveles de alerta del sistema operativo."""
    NORMAL   = "NORMAL"
    ATENCION = "ATENCION"
    CRITICO  = "CRITICO"


class Alert(BaseModel):
    """Una alerta generada por el diagnóstico operativo."""
    id: str
    level: AlertLevel
    message: str
    action: Optional[str] = None
    timestamp: str
    subsystem: str


class Detection(BaseModel):
    """Una detección de objeto con bounding box (S1 y S3)."""
    id: str
    class_name: str = Field(alias="class")  # "class" en el JSON del frontend
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] en píxeles

    class Config:
        populate_by_name = True


class S2Indicators(BaseModel):
    """Indicadores operativos calculados por el modelo S2."""
    intact_grain_pct: float  # % grano íntegro
    broken_grain_pct: float  # % grano roto — INDICADOR CRÍTICO
    straw_pct: float         # % paja
    overload_detected: bool  # ¿Flujo saturado?
    segmentation_map_b64: Optional[str] = None  # PNG en base64


class S3Indicators(BaseModel):
    """Indicadores operativos calculados por el modelo S3."""
    intact_grain_pct: float  # % grano íntegro (por área ponderada)
    broken_grain_pct: float  # % grano roto
    non_grain_pct: float     # % material no-grano — INDICADOR CRÍTICO
    total_detections: int    # Total de partículas detectadas
    recommended_action: Optional[str] = None


class InferenceResponse(BaseModel):
    """
    Respuesta completa de un análisis de frame.
    Este es el objeto principal que el frontend consume.
    """
    subsystem: str                        # "corte" | "trilla" | "limpieza"
    frame_id: str                         # Timestamp del análisis
    latency_ms: float                     # Tiempo de inferencia
    detections: List[dict]                # Bounding boxes (S1 y S3)
    indicators: dict                      # S2Indicators o S3Indicators
    alerts: List[Alert]


class SubsystemStatus(BaseModel):
    """Estado resumido de un subsistema para el Dashboard General."""
    status: AlertLevel
    last_latency_ms: float
    available: bool


class S1Status(SubsystemStatus):
    last_panicle_count: int = 0
    lodging_detected: bool = False


class S2Status(SubsystemStatus):
    broken_grain_pct: float = 0.0
    intact_grain_pct: float = 0.0


class S3Status(SubsystemStatus):
    non_grain_pct: float = 0.0
    intact_grain_pct: float = 0.0


class SystemStatusResponse(BaseModel):
    """Estado completo del sistema para el Dashboard General."""
    s1: S1Status
    s2: S2Status
    s3: S3Status
    recent_alerts: List[Alert]
    session_start: str


class HealthResponse(BaseModel):
    """Respuesta del endpoint de health check."""
    status: str
    models_loaded: dict
    device: str
    timestamp: str