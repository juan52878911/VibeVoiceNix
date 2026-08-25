"""API de voz del homelab."""

from voz_api.api import app
from voz_api import openai_api

# La fachada de OpenAI se monta AQUI y no dentro de api.py para que no haya
# ciclo: openai_api importa de api, y api no importa de openai_api. Al hacerlo
# en el paquete vale igual `uvicorn voz_api.api:app` que `python -m voz_api`:
# importar el submodulo ejecuta antes este __init__.
openai_api.instalar(app)

__all__ = ["app"]
