# app/inference/s2_inference.py
# ================================================================
# INFERENCIA SUBSISTEMA 2 — ZONA DE TRILLA
# ================================================================
# Recibe una imagen PIL o bytes.
# La preprocesa al tamaño 512x512 con normalización específica.
# Ejecuta el modelo U-Net + MobileNetV2.
# Post-procesa el mapa de segmentación.
# Calcula proporciones de cada clase.
# Retorna S2Indicators + mapa de segmentación coloreado en base64.

import time
import base64
import io
import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')  # Sin GUI — necesario en servidores sin pantalla
import matplotlib.pyplot as plt

from app.core.config import settings
from app.inference.model_loader import get_s2_model
from app.schemas.responses import S2Indicators

logger = logging.getLogger(__name__)

# Estadísticos de normalización de ImageNet
# (estándar para modelos preentrenados en ImageNet como MobileNetV2)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def preprocess_image_s2(pil_image: Image.Image) -> torch.Tensor:
    """
    Preprocesa una imagen PIL para inferencia con el modelo S2.
    IMPORTANTE: Garantiza que el tensor es float32 para evitar mismatch de tipos.
    """
    # Paso 1: Asegurar formato RGB
    img = pil_image.convert("RGB")

    # Paso 2: Redimensionar a 512x512
    img = img.resize(settings.S2_INPUT_SIZE, Image.BILINEAR)

    # Paso 3: Convertir a numpy y normalizar
    img_np = np.array(img, dtype=np.float32) / 255.0         # float32 explícito
    img_np = (img_np - IMAGENET_MEAN.astype(np.float32)) / IMAGENET_STD.astype(np.float32)

    # Paso 4: Convertir a tensor float32
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()  # .float() = float32
    img_tensor = img_tensor.unsqueeze(0)                       # [1, 3, 512, 512]

    return img_tensor.to(settings.DEVICE)


def generate_segmentation_map_b64(pred_mask: np.ndarray) -> str:
    """
    Genera un PNG coloreado del mapa de segmentación y lo retorna en base64.
    Este PNG es lo que el frontend muestra cuando el usuario activa
    el toggle "Ver segmentación".

    Colores por clase:
    - grano_integro: verde  (34, 197, 94)
    - grano_roto:    rojo   (239, 68, 68)
    - paja:          amarillo (234, 179, 8)
    """
    h, w = pred_mask.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, color in enumerate(settings.S2_CLASS_COLORS):
        mask = pred_mask == class_idx
        color_map[mask] = color

    # Convierte a PIL y luego a PNG base64
    pil_map = Image.fromarray(color_map)
    buffer = io.BytesIO()
    pil_map.save(buffer, format="PNG")
    buffer.seek(0)

    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode("utf-8")


def run_s2_inference(image_bytes: bytes) -> Tuple[S2Indicators, float]:
    """
    Ejecuta la inferencia completa del Subsistema 2.

    Args:
        image_bytes: Bytes de la imagen recibida del frontend

    Returns:
        (S2Indicators, latency_ms)
    """
    model = get_s2_model()

    # ── 1. Abrir imagen desde bytes ──────────────────────────────
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"No se pudo abrir la imagen: {e}")

    # ── 2. Preprocesar ───────────────────────────────────────────
    input_tensor = preprocess_image_s2(pil_image)

    # ── 3. Inferencia ─────────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():          # Sin gradientes = más rápido y menos memoria
        output = model(input_tensor)  # [1, 3, 512, 512] — logits por clase
    latency_ms = (time.perf_counter() - t0) * 1000

    # ── 4. Post-procesamiento ─────────────────────────────────────
    # Softmax convierte logits a probabilidades por clase
    probs = F.softmax(output, dim=1)  # [1, 3, 512, 512]

    # argmax da la clase con mayor probabilidad para cada píxel
    pred_mask = probs.argmax(dim=1).squeeze(0).cpu().numpy()  # [512, 512]

    # ── 5. Calcular proporciones ─────────────────────────────────
    total_pixels = pred_mask.size  # 512 * 512 = 262144
    # np.mean con condición boolean es equivalente a count / total
    intact_pct = float(np.mean(pred_mask == 0) * 100)  # Clase 0: grano_integro
    broken_pct = float(np.mean(pred_mask == 1) * 100)  # Clase 1: grano_roto
    straw_pct  = float(np.mean(pred_mask == 2) * 100)  # Clase 2: paja

    # ── 6. Detectar sobrecarga ───────────────────────────────────
    # Sobrecarga = más del 50% de paja Y menos del 30% de grano íntegro
    overload = straw_pct > 50.0 and intact_pct < 30.0

    # ── 7. Generar mapa coloreado ────────────────────────────────
    seg_map_b64 = generate_segmentation_map_b64(pred_mask)

    indicators = S2Indicators(
        intact_grain_pct=round(intact_pct, 2),
        broken_grain_pct=round(broken_pct, 2),
        straw_pct=round(straw_pct, 2),
        overload_detected=overload,
        segmentation_map_b64=seg_map_b64,
    )

    logger.info(
        f"S2 | Íntegro: {intact_pct:.1f}% | "
        f"Roto: {broken_pct:.2f}% | Paja: {straw_pct:.1f}% | "
        f"Latencia: {latency_ms:.1f}ms"
    )

    return indicators, latency_ms