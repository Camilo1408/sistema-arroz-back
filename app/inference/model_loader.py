# app/inference/model_loader.py
# ================================================================
# CARGA DE MODELOS AL INICIAR EL SERVIDOR
# ================================================================
# Los modelos se cargan UNA SOLA VEZ cuando FastAPI arranca.
# Esto evita cargar 50+ MB de pesos en cada petición HTTP,
# lo cual tardaría varios segundos por request.
# En cambio, quedan en memoria RAM/VRAM y cada inferencia
# tarda solo milisegundos.

import torch
import segmentation_models_pytorch as smp
from ultralytics import YOLO
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Variables globales que guardan los modelos cargados
_s2_model = None
_s3_model = None
# _s1_model = None  # Placeholder para S1

# Estado de carga de cada modelo
_models_status = {
    "s1": False,  # Pendiente de integración
    "s2": False,
    "s3": False,
}


def load_s2_model():
    """
    Carga el modelo U-Net + MobileNetV2 entrenado para S2.
    """
    global _s2_model, _models_status

    if not settings.S2_MODEL_PATH.exists():
        logger.error(f"Modelo S2 no encontrado en: {settings.S2_MODEL_PATH}")
        logger.error("Asegúrate de copiar best_model.pth en la carpeta models/s2/")
        return False

    try:
        logger.info(f"Cargando modelo S2 desde: {settings.S2_MODEL_PATH}")

        # Crear modelo vacío sin descargar pesos
        model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=3,
            classes=settings.S2_NUM_CLASSES,
        )

        # Cargar pesos guardados directamente
        checkpoint = torch.load(
            settings.S2_MODEL_PATH,
            map_location=settings.DEVICE,
        )
        
        # Si el checkpoint es un diccionario con clave 'state_dict', usarla
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)

        model.eval()
        model.to(settings.DEVICE)

        _s2_model = model
        _models_status["s2"] = True
        logger.info("✓ Modelo S2 (U-Net MobileNetV2) cargado correctamente")
        return True

    except Exception as e:
        logger.error(f"✗ Error cargando modelo S2: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_s3_model():
    """
    Carga el modelo YOLOv11s entrenado para S3.

    Ultralytics YOLO maneja toda la complejidad internamente:
    - La arquitectura está embebida en el archivo .pt
    - No es necesario recrear la arquitectura manualmente
    - model.names contiene los nombres de las clases
    """
    global _s3_model, _models_status

    if not settings.S3_MODEL_PATH.exists():
        logger.error(f"Modelo S3 no encontrado en: {settings.S3_MODEL_PATH}")
        logger.error("Asegúrate de copiar best.pt en la carpeta models/s3/")
        return False

    try:
        logger.info(f"Cargando modelo S3 desde: {settings.S3_MODEL_PATH}")

        # YOLO carga todo automáticamente: arquitectura + pesos + nombres de clases
        model = YOLO(str(settings.S3_MODEL_PATH))

        # Verifica que las clases son las esperadas
        class_names = list(model.names.values())
        logger.info(f"  Clases S3: {class_names}")

        _s3_model = model
        _models_status["s3"] = True
        logger.info("✓ Modelo S3 (YOLOv11s) cargado correctamente")
        return True

    except Exception as e:
        logger.error(f"✗ Error cargando modelo S3: {e}")
        return False


def load_all_models():
    """
    Carga todos los modelos disponibles.
    Llamado una sola vez en el arranque del servidor (lifespan de FastAPI).
    """
    logger.info("=" * 50)
    logger.info("INICIANDO CARGA DE MODELOS DE IA")
    logger.info(f"Dispositivo: {settings.DEVICE}")
    logger.info("=" * 50)

    s2_ok = load_s2_model()
    s3_ok = load_s3_model()
    # s1_ok = load_s1_model()  # Descomentar cuando S1 esté listo

    loaded = sum([s2_ok, s3_ok])
    logger.info(f"Modelos cargados: {loaded}/2")
    logger.info("=" * 50)


def get_s2_model():
    """Retorna el modelo S2 cargado. Lanza error si no está disponible."""
    if _s2_model is None:
        raise RuntimeError(
            "Modelo S2 no está disponible. "
            "Verifica que models/s2/best_model.pth existe y se cargó correctamente."
        )
    return _s2_model


def get_s3_model():
    """Retorna el modelo S3 cargado. Lanza error si no está disponible."""
    if _s3_model is None:
        raise RuntimeError(
            "Modelo S3 no está disponible. "
            "Verifica que models/s3/best.pt existe y se cargó correctamente."
        )
    return _s3_model


def get_models_status() -> dict:
    """Retorna el estado de carga de todos los modelos."""
    return dict(_models_status)