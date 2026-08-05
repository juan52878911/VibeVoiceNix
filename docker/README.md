# Usar el stack de voz con Docker

**No hace falta Nix.** Tres comandos:

```bash
./bajar-modelo.sh
sed "s/^VOZ_TOKEN=$/VOZ_TOKEN=$(openssl rand -hex 24)/" env.example > .env
docker compose up -d --build
```

Ese `sed` lee de `env.example` y escribe en `.env` en vez de editar en sitio,
que es lo que lo hace portable: `sed -i` necesita un argumento vacío en macOS
(`-i ''`) y ninguno en Linux, y equivocarse deja un `.env` con el token
**vacío** — es decir, la API abierta a cualquiera. Comprueba siempre que quedó
puesto:

```bash
grep VOZ_TOKEN .env
```

La primera vez tarda unos minutos porque compila whisper.cpp y descarga las
voces. Después arranca en segundos. Se construye para **la arquitectura de tu
máquina**: en un Mac ARM salen imágenes ARM, sin emulación ni constructor
remoto.

Abre **`http://localhost:8080`**. Ahí está la consola: pega el token del
`.env` en el campo de arriba y ya puedes generar voz, transcribir audio y
grabar del micrófono.

## Los tres perfiles

Elige según la máquina. Solo el primero arranca por defecto.

| Perfil | Comando | Qué levanta | Necesita |
|---|---|---|---|
| *(ninguno)* | `docker compose up -d` | Piper + whisper + consola | 1 GB |
| `pesado` | `docker compose --profile pesado up -d` | + VibeVoice en CPU | **8 GB para Docker** |
| `gpu` | `docker compose --profile gpu up -d --build` | + VibeVoice en NVIDIA | driver + toolkit |

### Por la API, sin consola

```bash
TOKEN=$(grep VOZ_TOKEN .env | cut -d= -f2)

# Generar una nota de voz
curl -X POST http://localhost:8080/tts \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"texto":"El backup terminó sin errores."}' -o nota.ogg

# Entender una nota de voz
curl -X POST http://localhost:8080/stt \
  -H "Authorization: Bearer $TOKEN" -F archivo=@nota.ogg
```

### El perfil `pesado` y la memoria

Este es el que da problemas, así que léelo antes de lanzarlo.

**El límite del compose no crea memoria.** Docker Desktop reparte lo que tenga
asignado en *Ajustes → Resources*, y por defecto suele quedarse corto. Medido
en un Mac con 5,8 GB asignados: el contenedor sube a **4,9 GiB y muere ahí**,
en silencio, con código de salida 0 y sin marcar siquiera `OOMKilled`. Parece
un cierre limpio y no lo es.

Sube Docker Desktop a **8 GB** antes de usar este perfil. Para comprobar
cuánto tiene ahora:

```bash
docker info --format '{{.MemTotal}}' | awk '{printf "%.1f GB\n", $1/1073741824}'
```

Si aun así falla, el síntoma es un contenedor que reinicia una y otra vez sin
llegar nunca a `Application startup complete`:

```bash
docker inspect voz-voz-stream-1 --format 'salida={{.State.ExitCode}} reinicios={{.RestartCount}}'
docker stats --no-stream voz-voz-stream-1     # ¿roza el límite?
```

Se rinde tras 3 intentos a propósito: reintentar sin memoria es un bucle
infinito que quema CPU y disco sin dar un solo mensaje útil.

Tarda **~2 minutos** en aceptar la primera petición: carga el modelo y hace
una síntesis de calentamiento para no cobrártela a ti.

### El perfil `gpu`

Comprueba primero que Docker ve la tarjeta:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

Si eso responde, `docker compose --profile gpu up -d --build`. Usa el mismo
Dockerfile y solo sustituye torch por la rueda de CUDA, porque la que fija el
lock **no trae CUDA compilado** y `torch.cuda.is_available()` daría `False`
por muy buena que fuera la GPU.

En el anfitrión hacen falta el driver de NVIDIA y `nvidia-container-toolkit`.

## En un Mac, VibeVoice va mejor NATIVO

Los contenedores de macOS corren dentro de una VM Linux que **no tiene acceso
a Metal**. Desde Docker, `torch.backends.mps.is_available()` da `False` siempre.
Así que en el Mac el perfil `pesado` funciona pero desperdicia la GPU:

```bash
cd docker && docker compose up -d     # Piper + whisper + consola, en el 8080
../scripts/voz-stream-mac.sh          # VibeVoice sobre Metal, en el 8082
```

La consola del 8080 detecta el 8082 sola. Medido en el mismo Mac:

| | primer sonido | RTF |
|---|---|---|
| MPS (fp16) | 0,79 s | **1,39** |
| CPU (int8) | 0,70 s | 2,96 |

## Qué esperar de rendimiento

Medido en la VM del homelab (i7-8700T, 6 núcleos):

| | RTF | Nota |
|---|---|---|
| Piper (TTS) | 0,12 | 8x más rápido que el tiempo real |
| whisper (STT) | 0,53 | con audio largo; ver abajo |
| VibeVoice + OpenVINO | 0,97 | más rápido que el tiempo real |

**El RTF de whisper engaña con audio corto.** Procesa en ventanas de 30
segundos, así que un clip de 2 s cuesta casi lo mismo que uno de 30. Medido:
~4,5 s de coste fijo por transcripción más ~0,41 s por segundo de audio. Con
un clip de 2,5 s el RTF sale 2,24; con uno de 37 s, 0,53.

## El micrófono necesita contexto seguro

En `http://localhost:8080` funciona: `localhost` cuenta como seguro. Pero si
abres la consola **por IP** (`http://192.168.2.54:8080`), el navegador no dará
acceso al micrófono. Ahí harían falta HTTPS y un certificado.

Todo lo demás —generar voz, subir un fichero para transcribir, el streaming—
funciona igual por IP.

## Dos formas de tener las imágenes

Este directorio ofrece **Dockerfiles normales** y el flake ofrece **imágenes
generadas por Nix**. No es duplicación por accidente:

| | Dockerfile | Imagen de Nix |
|---|---|---|
| Requisitos | Solo Docker | Nix + máquina x86_64 |
| Arquitectura | La de tu máquina | Solo `x86_64-linux` |
| Reproducible | Las versiones sí (mismo `uv.lock`) | Bit a bit |
| Para quién | **Cualquiera** | Producción |

Ambas fijan las dependencias con **el mismo `uv.lock`**, así que las versiones
de Python no pueden divergir. Lo que cambia es la base del sistema.

Las de Nix: `nix build .#imagenes.voz-api -o voz-api.tar.gz` y luego
`docker load < voz-api.tar.gz`.

## Tamaños reales

| Imagen | Tamaño |
|---|---|
| `voz-whisper` | 158 MB |
| `voz-api` | 1,8 GB |
| `voz-stream` | **9,8 GB** |

La de VibeVoice es enorme y no hay forma de evitarlo: son 1,9 GB de pesos más
torch con todas sus dependencias. Si solo quieres notas de voz, no la
construyas — `voz-api` hace TTS a RTF 0,12 con 150 MB de RAM.

## El modelo de whisper va aparte

`./bajar-modelo.sh` lo baja a `modelos/` y el compose lo monta. No va dentro
de la imagen: son 466 MB y ataría su versión a la del contenedor. Ese
directorio está en `.gitignore`.

## Límites conocidos

- **El `.env` nunca se sube.** Una vez se coló en un commit, acabó en un repo
  público y hubo que rotar el token. Por eso el ejemplo se llama `env.example`
  y el real está ignorado.
- **Un servicio, un contenedor.** whisper corre aparte de `voz-api` en vez de
  meter dos procesos en uno.
- **Una síntesis a la vez** en VibeVoice: el modelo ya satura los núcleos, y
  dos en paralelo solo harían ambas más lentas.
