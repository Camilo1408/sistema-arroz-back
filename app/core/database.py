# app/core/database.py
# ================================================================
# CONEXIÓN A LA BASE DE DATOS
# ================================================================
# Configura SQLAlchemy con soporte asíncrono.
# En desarrollo usa SQLite (archivo local).
# En producción usa PostgreSQL (Supabase).
# El código de la app NO necesita cambiar entre entornos.
# ================================================================

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Motor de base de datos asíncrono
# echo=False en producción (no imprime SQL en consola)
engine = create_async_engine(
    settings.async_database_url,
    echo=settings.DEBUG,
    # Para SQLite: permite usar la misma conexión en distintos threads
    connect_args={"check_same_thread": False}
    if "sqlite" in settings.async_database_url else {},
)

# Fábrica de sesiones: cada petición HTTP crea su propia sesión
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """
    Clase base para todos los modelos SQLAlchemy.
    Todos los modelos heredan de aquí.
    """
    pass


async def create_tables():
    """
    Crea todas las tablas si no existen.
    Se llama al arrancar el servidor.
    NO borra datos existentes — solo crea lo que falta.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✓ Tablas de base de datos verificadas/creadas")


async def get_db():
    """
    Dependencia de FastAPI para inyectar sesión de BD en endpoints.
    Garantiza que la sesión se cierra correctamente al terminar.

    Uso en un endpoint:
        async def mi_endpoint(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()