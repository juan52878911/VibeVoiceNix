# API HTTP

Referencia de `voz-api`, el servicio que escucha en el puerto 8080. Está pensada para que un agente
—OpenClaw en este caso— mande y entienda notas de voz.

El código es [`pkgs/voz-api/voz_api/api.py`](../pkgs/voz-api/voz_api/api.py). El servicio genera además su
propia documentación interactiva en `/docs` y el esquema en `/openapi.json`.

- [Autenticación](#autenticación)
- [`POST /tts` — texto a voz](#post-tts--texto-a-voz)
- [`POST /stt` — voz a texto](#post-stt--voz-a-texto)
- [`GET /voces`](#get-voces)
- [`GET /health`](#get-health)
- [Errores](#errores)
- [Ejemplos de integración](#ejemplos-de-integración)

---

## Autenticación

Bearer token en la cabecera `Authorization`:

```
Authorization: Bearer 7f3a9c1e…
```

El token se genera solo en el primer arranque y vive en `/var/lib/voz/token.env`:

```bash
ssh juan@voz sudo cat /var/lib/voz/token.env
```

Si `services.voz-api.ficheroToken` es `null`, la API queda **abierta** y no pide nada. Eso solo tiene
sentido en una red de confianza, y el módulo no deja combinarlo con `abrirCortafuegos = true`.

| Ruta | ¿Pide token? |
|---|---|
| `POST /tts` | sí |
| `POST /stt` | sí |
| `GET /voces` | no |
| `GET /health` | no — para poder monitorizarla sin repartir el secreto |
| `GET /docs`, `GET /openapi.json` | no |

---

## `POST /tts` — texto a voz

Convierte texto en audio con Piper. **Content-Type:** `application/json`.

### Parámetros

| Campo | Tipo | Por defecto | Notas |
|---|---|---|---|
| `texto` | string | *(obligatorio)* | entre 1 y 8000 caracteres |
| `voz` | string \| null | la voz por defecto del sistema | tiene que estar instalada; ver `GET /voces` |
| `formato` | `ogg` \| `mp3` \| `wav` | `ogg` | |
| `velocidad` | float | `1.0` | mayor que `0.3` y menor que `3.0` |

> **`velocidad` va al revés de lo que parece.** Es el `length_scale` de Piper: **por encima de 1 habla más
> despacio** y por debajo, más deprisa. `1.0` es la velocidad nativa de la voz.

### Formatos de salida

| Formato | Códec | Para qué |
|---|---|---|
| `ogg` | Opus 32 kbps, perfil `voip` | **el que tragan WhatsApp y Telegram como nota de voz**; es el defecto |
| `mp3` | LAME `-q:a 4` (VBR ~165 kbps) | compatibilidad universal |
| `wav` | PCM tal cual sale de Piper | sin recodificar; útil para encadenar procesado |

### Petición

```bash
curl -X POST http://voz:8080/tts -H "Authorization: Bearer $VOZ_TOKEN" -H "Content-Type: application/json" -d '{"texto":"El backup de anoche terminó sin errores.","voz":"es_MX-claude-high","formato":"ogg","velocidad":1.0}' -D- --output nota.ogg
```

### Respuesta

El cuerpo es el audio. Las cabeceras traen la medición de esa síntesis concreta:

```
HTTP/1.1 200 OK
content-type: audio/ogg
X-Duracion-S: 3.84
X-Proceso-S: 0.16
X-RTF: 0.042
X-Voz: es_MX-claude-high
Content-Disposition: inline; filename="voz.ogg"
```

| Cabecera | Significado |
|---|---|
| `X-Duracion-S` | duración del audio generado, en segundos |
| `X-Proceso-S` | lo que costó sintetizarlo |
| `X-RTF` | `X-Proceso-S ÷ X-Duracion-S`; por debajo de 1 es más rápido que el tiempo real |
| `X-Voz` | la voz que se usó de verdad |

**La primera petición de cada voz es más lenta**: hay que leer el `.onnx` del store. A partir de ahí la voz
se queda en memoria del proceso y el RTF baja a su valor real.

---

## `POST /stt` — voz a texto

Transcribe audio con `whisper.cpp`. **Content-Type:** `multipart/form-data`.

### Parámetros

| Campo | Tipo | Por defecto | Notas |
|---|---|---|---|
| `archivo` | fichero | *(obligatorio)* | **cualquier formato que entienda ffmpeg**: ogg, mp3, m4a, wav, incluso vídeo |
| `idioma` | string | `es` | `auto` para detectarlo |
| `prompt` | string | el prompt de vocabulario del sistema | lo que mandes sustituye al de por defecto |

No hace falta convertir nada antes de subirlo: la API normaliza siempre a WAV PCM 16 kHz mono con `ffmpeg`,
que es lo único que acepta `whisper.cpp`.

### El prompt: el ajuste que más cambia el resultado

`whisper` acepta un texto inicial que sesga su vocabulario. Sin él, en un homelab, *«WireGuard»* se
transcribe *«We The War»* y *«homelab»* se convierte en *«omelab»*. Con él, acierta.

El valor por defecto es una lista de jerga del homelab (Proxmox, LXC, Caddy, systemd, NixOS, Terraform,
WireGuard…) y se configura en `services.voz-api.promptSTT`. Si mandas `prompt` en la petición, sustituyes
el del sistema para esa llamada — útil cuando sabes de qué va el audio:

```bash
-F "prompt=Nombres propios: Beatriz, Kubernetes, Grafana, Loki, Tailscale."
```

### Petición

```bash
curl -X POST http://voz:8080/stt -H "Authorization: Bearer $VOZ_TOKEN" -F "archivo=@nota-de-voz.ogg" -F "idioma=es" | jq
```

### Respuesta

```json
{
  "texto": "Revisa el backup de Proxmox de anoche, creo que falló el contenedor de Postgres.",
  "idioma": "es",
  "duracion_s": 7.72,
  "proceso_s": 5.18,
  "rtf": 0.671
}
```

| Campo | Significado |
|---|---|
| `texto` | la transcripción, ya sin espacios sobrantes |
| `duracion_s` | duración del audio que subiste |
| `proceso_s` | lo que costó transcribirlo |
| `rtf` | el cociente; `null` si el audio tenía duración cero |

El tiempo de espera hacia `whisper-server` es de **300 segundos**, holgado a propósito: con el modelo
`small` y un RTF de 0,671, cubre audios de varios minutos.

---

## `GET /voces`

Las voces instaladas y cuál se usa por defecto.

```bash
curl -s http://voz:8080/voces | jq
```

```json
{
  "voces": ["es_ES-davefx-medium", "es_MX-ald-medium", "es_MX-claude-high"],
  "por_defecto": "es_MX-claude-high"
}
```

El catálogo completo disponible para instalar —cinco voces en español— está en
[opciones.md](opciones.md#voces-de-piper-disponibles), con sus tamaños y velocidades medidas.

---

## `GET /health`

Estado del servicio. No pide token, para poder monitorizarla sin repartir el secreto.

```bash
curl -s http://voz:8080/health | jq
```

```json
{
  "estado": "ok",
  "tts": {
    "motor": "piper",
    "voces": ["es_ES-davefx-medium", "es_MX-ald-medium", "es_MX-claude-high"],
    "por_defecto": "es_MX-claude-high",
    "cargadas": ["es_MX-claude-high"]
  },
  "stt": {
    "motor": "whisper.cpp",
    "url": "http://127.0.0.1:8081",
    "disponible": true
  },
  "auth": "bearer"
}
```

Los tres campos que hay que mirar:

- **`stt.disponible`** — comprueba de verdad que `whisper-server` responde, con un tiempo de espera de 2 s.
  Si es `false`, el TTS sigue funcionando pero `/stt` devolverá `503`.
- **`auth`** — `bearer` o `abierta`. Si dice `abierta` cuando esperabas token, el `EnvironmentFile` no llegó.
- **`tts.cargadas`** — qué voces están ya en memoria. Empieza vacío y crece con el uso.

---

## Errores

Todos los errores devuelven `{"detail": "…"}`.

| Código | Cuándo | Mensaje |
|---|---|---|
| `401` | falta el token o no coincide | `token invalido o ausente` |
| `404` | la voz pedida no está instalada | `voz 'X' no encontrada; disponibles: [...]` |
| `400` | formato de salida desconocido | `formato debe ser uno de ['mp3', 'ogg', 'wav']` |
| `400` | subiste un fichero vacío | `archivo vacio` |
| `400` | ffmpeg no supo decodificar el audio | `no pude decodificar el audio: …` |
| `422` | el JSON no cumple el esquema (texto vacío, velocidad fuera de rango…) | detalle de Pydantic |
| `500` | ffmpeg falló al convertir la salida | `ffmpeg fallo al convertir a X: …` |
| `502` | whisper respondió, pero con error | `whisper devolvio N: …` |
| `503` | whisper no responde | `whisper-server no responde en …` |

Los mensajes de `400`, `500` y `502` incluyen los primeros 300 caracteres de la salida de error de la
herramienta que falló, que suele bastar para saber qué pasó.

---

## Ejemplos de integración

### Ida y vuelta completa desde Python

```python
import httpx

BASE = "http://voz:8080"
CAB = {"Authorization": f"Bearer {TOKEN}"}

# texto -> nota de voz lista para WhatsApp
r = httpx.post(f"{BASE}/tts", headers=CAB,
               json={"texto": "El despliegue terminó sin errores."}, timeout=60)
r.raise_for_status()
open("nota.ogg", "wb").write(r.content)
print("RTF:", r.headers["X-RTF"])

# nota de voz -> texto
with open("entrante.ogg", "rb") as f:
    r = httpx.post(f"{BASE}/stt", headers=CAB,
                   files={"archivo": f}, data={"idioma": "es"}, timeout=300)
print(r.json()["texto"])
```

### Como notas de voz de WhatsApp o Telegram

El formato por defecto (`ogg` con Opus a 32 kbps y perfil `voip`) es exactamente el que ambos aceptan como
nota de voz nativa, no como fichero adjunto. No hay que recodificar nada.

### Elegir voz por contexto

Con varias voces instaladas se pueden reservar unas para unas cosas y otras para otras —por ejemplo una
voz para las alertas y otra para las respuestas normales— pasando `voz` en cada petición. La primera
llamada a cada voz paga su carga en memoria; el resto, no.

### Medir sin instrumentar

Las cabeceras `X-*` de `/tts` y los campos `proceso_s` / `rtf` de `/stt` salen en cada respuesta, así que
cualquier cliente puede registrar el rendimiento real sin tocar el servidor. Es de donde salen las cifras
de [rendimiento.md](rendimiento.md).
