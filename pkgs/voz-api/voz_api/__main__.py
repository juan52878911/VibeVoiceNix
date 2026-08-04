"""Arranque del servidor. El modulo NixOS invoca `voz-api` sin argumentos y
pasa la configuracion por entorno."""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "voz_api.api:app",
        host=os.environ.get("VOZ_HOST", "0.0.0.0"),
        port=int(os.environ.get("VOZ_PORT", "8080")),
        workers=1,  # las voces de Piper viven en memoria del proceso; un worker
        log_level=os.environ.get("VOZ_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
