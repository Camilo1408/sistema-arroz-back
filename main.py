# main.py
# ================================================================
# PUNTO DE ENTRADA PRINCIPAL DEL SERVIDOR
# ================================================================
# Este es el archivo que arranca todo el backend.
# Se ejecuta con: uvicorn main:app --reload

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.config import settings
from app.inference.model_loader import load_all_models
from app.api.routes import router
# En main.py — modifica solo el bloque @asynccontextmanager
from app.core.database import create_tables

# ── Configuración del sistema de logging ─────────────────────────
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan: código que corre al inicio y fin del servidor ──────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan gestiona eventos de inicio y cierre.
    Al iniciar: carga todos los modelos de IA en memoria.
    Al cerrar: libera recursos (automático).

    ¿Por qué cargar aquí y no en cada petición?
    Cargar un modelo PyTorch/YOLO tarda 2-5 segundos.
    Si se cargara en cada petición, cada análisis tardaría 5+ segundos.
    Cargado una vez al inicio, cada inferencia tarda 20-200ms.
    """
    logger.info("🚀 Iniciando servidor de diagnóstico de arroz...")

    # NUEVO: Crear tablas de base de datos
    await create_tables()

    load_all_models()
    logger.info("✅ Servidor listo — esperando peticiones")

    yield  # El servidor está activo entre yield y el bloque de cierre

    logger.info("🔴 Cerrando servidor...")


# ── Creación de la aplicación FastAPI ────────────────────────────
app = FastAPI(
    title="API de Diagnóstico de Cosecha Mecanizada de Arroz",
    description="""
    Backend del sistema de visión artificial para diagnóstico
    del proceso de cosecha mecanizada de arroz.

    ## Subsistemas disponibles
    - **S2 — Trilla**: U-Net + MobileNetV2 (segmentación semántica)
    - **S3 — Limpieza**: YOLOv11s multiclase (detección de objetos)
    - **S1 — Corte**: Pendiente de integración

    ## Uso
    1. POST /api/v1/infer/trilla — Analizar imagen de zona de trilla
    2. POST /api/v1/infer/limpieza — Analizar imagen de sistema de limpieza
    3. GET /api/v1/status — Estado general del sistema
    4. GET /api/v1/history/{subsystem} — Historial de análisis
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",      # Swagger UI en http://localhost:8000/docs
    redoc_url="/redoc",    # ReDoc en http://localhost:8000/redoc
)


# ── Configuración de CORS ─────────────────────────────────────────
# CORS permite que el frontend (localhost:5173) llame al backend (localhost:8000)
# Sin esto el navegador bloquea las peticiones por seguridad (Same-Origin Policy)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,   # ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],       # GET, POST, OPTIONS, etc.
    allow_headers=["*"],       # Content-Type, Authorization, etc.
)


# ── Registro de rutas ─────────────────────────────────────────────
# El prefix /api/v1 se agrega automáticamente a todas las rutas
# Ejemplo: @router.get("/status") → GET http://localhost:8000/api/v1/status
app.include_router(router, prefix="/api/v1")

# ── Health check raíz ─────────────────────────────────────────────
@app.get("/", tags=["Sistema"])
async def root():
    return {
        "mensaje": "API de Diagnóstico de Cosecha de Arroz",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "/api/v1/health",
    }


# ── Arranque directo (python main.py) ─────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,  # reload=True reinicia el servidor al guardar cambios
        log_level="info",
    )