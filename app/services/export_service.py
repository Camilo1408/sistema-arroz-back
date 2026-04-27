# app/services/export_service.py
# ================================================================
# SERVICIO DE EXPORTACIÓN CSV
# ================================================================

import csv
import io
from datetime import datetime, timezone
from typing import List


def generate_csv(records: List[dict]) -> str:
    """
    Genera un CSV en memoria con los registros del historial.
    Retorna el contenido CSV como string UTF-8.
    """
    if not records:
        return "timestamp,subsystem,latency_ms,alert_count\n"

    output = io.StringIO()
    fieldnames = list(records[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

    return output.getvalue()