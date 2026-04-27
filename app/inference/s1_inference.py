# app/inference/s1_inference.py
# ================================================================
# INFERENCIA SUBSISTEMA 1 — CABEZAL DE CORTE (PENDIENTE)
# ================================================================
# Este archivo existe como PLACEHOLDER para cuando el modelo S1
# esté entrenado y listo.
#
# Para integrar S1 cuando el modelo esté disponible:
# 1. Copiar best.pt (o best_model.pth) en models/s1/
# 2. Descomentar S1_MODEL_PATH en .env y config.py
# 3. Descomentar load_s1_model() en model_loader.py
# 4. Implementar run_s1_inference() en este archivo
# 5. Descomentar el endpoint /infer/corte en routes.py
#
# NO SE NECESITA CAMBIAR NADA MÁS EN EL PROYECTO.
# ================================================================

from typing import Tuple, List


class S1NotAvailableError(Exception):
    """Error para cuando S1 aún no está disponible."""
    pass


def run_s1_inference(image_bytes: bytes) -> Tuple[dict, List[dict], float]:
    """
    PLACEHOLDER — Inferencia Subsistema 1 (pendiente).
    Lanza error explicativo hasta que el modelo esté disponible.
    """
    raise S1NotAvailableError(
        "El modelo S1 (Cabezal de Corte) aún no está disponible. "
        "El sistema está preparado para integrarlo. "
        "Coloca el modelo en models/s1/ y sigue las instrucciones en este archivo."
    )