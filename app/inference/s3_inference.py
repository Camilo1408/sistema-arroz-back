# app/inference/s3_inference.py
# ================================================================
# INFERENCIA SUBSISTEMA 3 — SISTEMA DE LIMPIEZA
# ================================================================
# Recibe una imagen en bytes.
# La pasa al modelo YOLOv11s que detecta y clasifica partículas.
# Calcula la composición del flujo POR ÁREA DE BB PONDERADA
# (no por conteo simple — un fragmento de 8mm ≠ un grano de 3mm).
# Retorna S3Indicators + lista de detecciones con bounding boxes.

import time
import io
import logging
from typing import Tuple, List

import numpy as np
from PIL import Image

from app.core.config import settings
from app.inference.model_loader import get_s3_model
from app.schemas.responses import S3Indicators

logger = logging.getLogger(__name__)


def run_s3_inference(image_bytes: bytes) -> Tuple[S3Indicators, List[dict], float]:
    """
    Ejecuta la inferencia completa del Subsistema 3.

    Args:
        image_bytes: Bytes de la imagen recibida del frontend

    Returns:
        (S3Indicators, detections_list, latency_ms)

    Nota sobre el cálculo de proporciones:
        Se usa ÁREA DE BOUNDING BOX PONDERADA, no conteo de objetos.
        La proporción de cada clase = Σ(áreas de BB de esa clase) / Σ(áreas totales)
        Esto es correcto porque un fragmento grande ocupa más área que un grano pequeño.
    """
    model = get_s3_model()

    # ── 1. Abrir imagen desde bytes ──────────────────────────────
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"No se pudo abrir la imagen: {e}")

    # ── 2. Inferencia con YOLO ────────────────────────────────────
    # YOLO maneja el redimensionamiento internamente (imgsz=640)
    t0 = time.perf_counter()
    results = model.predict(
        source=pil_image,
        conf=settings.S3_CONF_THRESHOLD,    # Umbral de confianza mínimo
        iou=settings.S3_IOU_THRESHOLD,      # Umbral IoU para NMS
        imgsz=settings.S3_INPUT_SIZE,       # 640x640
        verbose=False,                       # Sin logs de YOLO en consola
        device=settings.DEVICE,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    # ── 3. Extraer detecciones ────────────────────────────────────
    detections_list = []
    class_areas = {}  # Acumula áreas por clase: {"grano_integro": 1234.5, ...}

    result = results[0]  # Solo procesamos una imagen a la vez

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes

        for i, box in enumerate(boxes):
            # Coordenadas del bounding box en píxeles [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_idx  = int(box.cls[0])
            class_name = model.names[class_idx]

            # Área del bounding box en píxeles²
            area = (x2 - x1) * (y2 - y1)

            # Acumula área por clase
            if class_name not in class_areas:
                class_areas[class_name] = 0.0
            class_areas[class_name] += area

            # Formatea la detección para el frontend
            detections_list.append({
                "id": f"s3_det_{i}",
                "class": class_name,
                "confidence": round(confidence, 4),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
            })

    # ── 4. Calcular proporciones por área ponderada ───────────────
    total_area = sum(class_areas.values())

    if total_area > 0:
        intact_pct = class_areas.get("grano_integro",    0.0) / total_area * 100
        broken_pct = class_areas.get("grano_roto",       0.0) / total_area * 100
        non_grain_pct = class_areas.get("material_no_grano", 0.0) / total_area * 100
    else:
        # Sin detecciones: no hay partículas visibles
        intact_pct = broken_pct = non_grain_pct = 0.0

    # ── 5. Recomendación operativa ────────────────────────────────
    recommended_action = None
    if non_grain_pct >= settings.S3_NON_GRAIN_CRITICAL:
        recommended_action = (
            f"Material no-grano excede {settings.S3_NON_GRAIN_CRITICAL}% "
            "— Verificar velocidad del ventilador y apertura de cribas"
        )
    elif non_grain_pct >= settings.S3_NON_GRAIN_WARNING:
        recommended_action = (
            "Material no-grano cerca del umbral — Monitorear activamente"
        )

    indicators = S3Indicators(
        intact_grain_pct=round(intact_pct, 2),
        broken_grain_pct=round(broken_pct, 2),
        non_grain_pct=round(non_grain_pct, 2),
        total_detections=len(detections_list),
        recommended_action=recommended_action,
    )

    logger.info(
        f"S3 | Detecciones: {len(detections_list)} | "
        f"Íntegro: {intact_pct:.1f}% | "
        f"No-grano: {non_grain_pct:.2f}% | "
        f"Latencia: {latency_ms:.1f}ms"
    )

    return indicators, detections_list, latency_ms