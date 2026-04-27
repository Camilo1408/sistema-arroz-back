# download_models.py
# ================================================================
# Descarga los modelos desde Hugging Face Hub al arrancar en producción.
# Se ejecuta automáticamente si los archivos no existen localmente.
# ================================================================

import os
import urllib.request
from pathlib import Path

# URLs de Hugging Face (reemplaza con tu usuario real)
HF_USER = os.getenv("HF_USER", "tuusuario")
MODELS = {
    "models/s2/best_model.pth": f"https://huggingface.co/{HF_USER}/sistema-arroz-modelos/resolve/main/best_model.pth",
    "models/s3/best.pt": f"https://huggingface.co/{HF_USER}/sistema-arroz-modelos/resolve/main/best.pt",
}

def download_models():
    """Descarga los modelos que no existan localmente."""
    for local_path, url in MODELS.items():
        path = Path(local_path)
        if path.exists():
            print(f"✓ Ya existe: {local_path}")
            continue

        print(f"⬇ Descargando {local_path}...")
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            urllib.request.urlretrieve(url, path)
            size_mb = path.stat().st_size / 1_000_000
            print(f"  ✓ Descargado: {size_mb:.1f} MB")
        except Exception as e:
            print(f"  ✗ Error descargando {local_path}: {e}")

if __name__ == "__main__":
    download_models()