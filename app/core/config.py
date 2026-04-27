# app/core/config.py
# ================================================================
# CONFIGURACIÓN GLOBAL DEL SISTEMA
# ================================================================
# Lee el archivo .env y expone las configuraciones como atributos
# de clase. Se importa en todos los módulos que necesiten config.

import os
from pathlib import Path
from dotenv import load_dotenv

# load_dotenv() busca .env en la carpeta actual y carga las variables
load_dotenv()

class Settings:
    """
    Configuración central del sistema.
    Todos los parámetros ajustables están aquí.
    Para cambiar el puerto, los umbrales o las rutas, solo se edita .env
    """
    # ── Servidor ──────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # ── Dispositivo de inferencia ─────────────────────────────────
    DEVICE: str = os.getenv("DEVICE", "cpu")

    # ── Rutas a los modelos ───────────────────────────────────────
    # Path.cwd() es la carpeta desde donde se ejecuta main.py
    BASE_DIR: Path = Path.cwd()

    S2_MODEL_PATH: Path = BASE_DIR / os.getenv("S2_MODEL_PATH", "models/s2/best_model.pth")
    S3_MODEL_PATH: Path = BASE_DIR / os.getenv("S3_MODEL_PATH", "models/s3/best.pt")
    # S1 está pendiente: cuando el modelo esté listo, descomentar esta línea:
    # S1_MODEL_PATH: Path = BASE_DIR / os.getenv("S1_MODEL_PATH", "models/s1/best.pt")

    # ── Parámetros de los modelos ────────────────────────────────
    # S2: U-Net + MobileNetV2
    S2_INPUT_SIZE: tuple = (512, 512)  # Resolución con la que fue entrenado
    S2_NUM_CLASSES: int = 3            # grano_integro, grano_roto, paja
    S2_CLASS_NAMES: list = ["grano_integro", "grano_roto", "paja"]
    # Colores RGB para el mapa de segmentación (verde, rojo, amarillo)
    S2_CLASS_COLORS: list = [(34, 197, 94), (239, 68, 68), (234, 179, 8)]

    # S3: YOLOv11s
    S3_INPUT_SIZE: int = 640           # Resolución con la que fue entrenado
    S3_CONF_THRESHOLD: float = 0.25    # Confianza mínima para aceptar una detección
    S3_IOU_THRESHOLD: float = 0.45     # IoU para NMS
    # Las clases reales se cargan del modelo; este es el orden esperado:
    S3_CLASS_NAMES: list = ["grano_integro", "grano_roto", "material_no_grano"]

    # ── Umbrales operativos ───────────────────────────────────────
    S2_BROKEN_GRAIN_WARNING: float  = float(os.getenv("S2_BROKEN_GRAIN_WARNING",  "0.3"))
    S2_BROKEN_GRAIN_CRITICAL: float = float(os.getenv("S2_BROKEN_GRAIN_CRITICAL", "0.5"))
    S3_NON_GRAIN_WARNING: float     = float(os.getenv("S3_NON_GRAIN_WARNING",     "1.5"))
    S3_NON_GRAIN_CRITICAL: float    = float(os.getenv("S3_NON_GRAIN_CRITICAL",    "2.0"))

    # ── CORS ─────────────────────────────────────────────────────
    CORS_ORIGINS: list = os.getenv(
        "CORS_ORIGINS", "http://localhost:5173"
    ).split(",")

    # En app/core/config.py, dentro de la clase Settings, agrega:

    # ── Base de datos ─────────────────────────────────────────────
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite+aiosqlite:///./diagnostico_arroz.db"
    )

    # Si la URL viene de Heroku/Render como "postgres://...", 
    # SQLAlchemy moderno requiere "postgresql+asyncpg://..."
    @property
    def async_database_url(self) -> str:
        url = self.DATABASE_URL
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        return url

# Instancia global única de Settings
# Se importa como: from app.core.config import settings
settings = Settings()