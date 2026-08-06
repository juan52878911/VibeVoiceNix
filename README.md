<p align="center">
  <img src="docs/banner.svg" alt="VibeVoiceNix — voz local para el homelab" width="100%">
</p>

# VibeVoiceNix

Stack de voz en español para un homelab, declarado por completo en NixOS y
desplegable sobre cualquier Proxmox con un comando.

Nace de montar el stack a mano en un contenedor Debian, medirlo, y pasarlo a
algo inmutable y replicable. **Todos los números de este README están medidos**
en un Intel i7-8700T (6 núcleos, AVX2, sin GPU).

La documentación extendida —arquitectura, despliegue, referencia de la API y de
las opciones, y las mediciones completas— está en [`docs/`](#documentación).

## Qué hace

| Componente | Papel | RTF medido |
|---|---|---|
| **Piper** | TTS de producción | **0,042** (24x tiempo real) |
| **whisper.cpp** | STT | **0,671** con el modelo `small` |
| **VibeVoice-Realtime-0.5B** | Laboratorio de TTS | **3,92x** con 8 núcleos (4,80x con 4) |

La API expone `/tts` y `/stt` por HTTP para que un agente —OpenClaw, por
ejemplo— pueda mandar y entender notas de voz. El TTS devuelve **ogg/opus** por
defecto, que es el formato que aceptan WhatsApp y Telegram como nota de voz.

```bash
# Generar una nota de voz
curl -X POST http://voz:8080/tts \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"texto":"El backup terminó sin errores."}' \
  -o nota.ogg

# Entender una nota de voz
curl -X POST http://voz:8080/stt \
  -H "Authorization: Bearer $TOKEN" \
  -F archivo=@nota.ogg
```

### Narrar un LLM según escribe: sesiones

VibeVoice (puerto 8082) tiene además un modo de **sesión**: una sola locución
continua a la que se le va metiendo texto conforme llega, en vez de una petición
por frase.

```bash
curl -X POST http://voz:8082/tts/sesion/charla \
     -d '{"texto":"El backup de anoche terminó sin errores."}'   # encola
curl http://voz:8082/tts/sesion/charla/audio -o charla.wav &     # un solo WAV
curl -X POST http://voz:8082/tts/sesion/charla \
     -d '{"texto":"Los tres servicios responden con normalidad."}'
curl -X POST http://voz:8082/tts/sesion/charla/fin                # cierra
```

Frase a frase con `/tts/stream` cada una empieza desde cero y suena a lista de
frases sueltas. En una sesión el modelo nunca deja de hablar: el texto le llega
por delante del habla y la prosodia sigue de una frase a la siguiente. Y el
audio resultante es **idéntico bit a bit** al de haber pasado todo el texto de
golpe en una sola llamada — comprobado con `scripts/sesiones_fidelidad.py`.

Si el texto deja de llegar, el modelo espera parado (por defecto 20 s,
`VIBEVOICE_ESPERA_TEXTO`) y cierra la locución bien en vez de dejarla colgada.
Mientras espera no consume CPU, así que otras peticiones normales pasan.

La misma sesión está también en `WS /tts/sesion/ws`, que es lo cómodo desde un
navegador: el texto sube como JSON (`{"accion":"abrir"|"texto"|"fin"}`) y bajan
marcos binarios `[tipo:1][longitud:4 BE][carga]` — tipo 0 PCM de 16 bits a
24 kHz, tipo 1 evento JSON —, el mismo formato que usa
`scripts/asistente_web.py`, así que su lector vale tal cual. La ventaja sobre
HTTP no es el audio, que es **idéntico bit a bit** por los dos caminos, sino que
el aviso de «el modelo se quedó sin texto» llega empujado en vez de por sondeo,
y que al caerse el socket la sesión se aborta y libera el modelo sola. El HTTP
se queda como está: es lo que se prueba con `curl`. Se verifica con
`scripts/ws_fidelidad.py`.

## Tres cosas que conviene saber antes de empezar

**VibeVoice apenas habla español.** El 1.5B y el Large-7B están entrenados solo
con inglés y chino. El único con voces en español es el Realtime-0.5B, y
Microsoft las añadió en diciembre de 2025 marcadas como experimentales. Aquí
está como laboratorio, no como motor de producción: para eso está Piper, que es
~100x más rápido.

**Una GPU integrada no ayuda.** Se probó pasar una Intel UHD 630 al contenedor.
Funciona (OpenCL 3.0, Vulkan 1.3), pero medido con whisper.cpp resultó **2,5x
más lenta que la CPU**:

| | `base` | `small` |
|---|---|---|
| CPU, 6 hilos | 1,53 s | 5,18 s |
| iGPU Vulkan | 7,65 s | 12,88 s |

El propio informe de ggml lo explica: `matrix cores: none`, `bf16: 0`. Y PyTorch
nunca soportará esa generación — XPU e IPEX cubren Arc/Xe en adelante, y en la
máquina `torch.xpu.is_available()` devuelve `False`. Por eso el diseño es
CPU-only y no hay passthrough de GPU en el Terraform.

**El prompt de whisper es el ajuste que más se nota.** whisper.cpp acepta un
prompt inicial que sesga el vocabulario. Sin él:

> Soy tu **omelab**. El túnel de **We The War** se cayó...

Con una lista de la jerga propia en `services.voz-api.promptSTT`:

> Soy tu **homelab**. El túnel de **WireGuard** se cayó...

## Cómo está montado

```
flake.nix                    entradas y nixosConfigurations.voz
nix/
  options.nix                opciones propias (claves SSH, disco)
  host.nix                   ← lo único que tienes que editar
  configuration.nix          sistema base
  disko.nix                  particionado declarativo
  modules/
    voz-api.nix              Piper + fachada HTTP
    whisper.nix              whisper.cpp residente en loopback
    vibevoice.nix            orden `vibevoice` (no es un servicio)
  pkgs/
    piper-voices.nix         voces con hash fijo
    vibevoice-weights.nix    pesos y voces con hash fijo
  overlay.nix                uv2nix -> voz-api y vibevoice-env
pkgs/
  voz-api/                   workspace uv ligero (FastAPI + Piper)
  vibevoice/                 workspace uv pesado (torch CPU + VibeVoice)
terraform/                   la VM en Proxmox
ansible/playbooks/           provision -> install -> update
docs/                        documentacion extendida
  banner.py                  genera banner.svg; el SVG es su salida
```

### Por qué Ansible no configura nada

NixOS ya es configuración declarativa. Si Ansible tocara el sistema habría dos
fuentes de verdad y se perdería justo la propiedad que se busca. Aquí Ansible
hace lo que Nix no hace: **encadenar el ciclo** (Terraform → nixos-anywhere →
nixos-rebuild) y verificar cada paso. La configuración vive entera en el flake.

### Por qué uv2nix

VibeVoice no está en nixpkgs y fija `transformers==4.51.3`. `pkgs/vibevoice/uv.lock`
clava el commit del repo (`94da20d`) y la rueda `torch 2.13.0+cpu` desde el
índice CPU de PyTorch — el lock no contiene **ningún** paquete de NVIDIA, que en
una máquina sin GPU serían 2,5 GB tirados. uv2nix traduce ese lock a
derivaciones Nix, así que el entorno se reconstruye idéntico.

Los pesos (1,9 GB) y las voces entran como descargas con hash fijo: si Hugging
Face sirviera otra cosa, el build falla en vez de instalar algo distinto en
silencio.

## Desplegar

Requisitos: Nix con flakes. Todo lo demás lo trae el `devShell`.

```bash
nix develop            # trae tofu, ansible, uv y nixos-anywhere
```

1. **Configura tu entorno** — dos ficheros, y las claves SSH deben coincidir:

   ```bash
   cp terraform/terraform.tfvars.example terraform/terraform.tfvars
   $EDITOR terraform/terraform.tfvars   # token de Proxmox, IP, recursos
   $EDITOR nix/host.nix                 # tu clave pública y el disco
   ```

   El token de Proxmox se crea con:
   ```bash
   pveum user token add root@pam terraform --privsep 0
   ```

2. **Crea la VM**:
   ```bash
   ansible-playbook ansible/playbooks/provision.yml
   ```

3. **Instala NixOS** (formatea el disco de la VM):
   ```bash
   ansible-playbook ansible/playbooks/install.yml
   ```

4. **Día a día**, tras cambiar cualquier cosa en `nix/`:
   ```bash
   ansible-playbook ansible/playbooks/update.yml
   ```

El token de la API se genera solo en el primer arranque y vive en
`/var/lib/voz/token.env`, **fuera del store de Nix** — el store es legible por
cualquier usuario del sistema.

## Dónde puede correr

El mismo código va en tres sitios, pero **no por el mismo camino**. El
dispositivo se elige solo (`cuda` → `mps` → `cpu`), y se puede forzar con
`VIBEVOICE_DISPOSITIVO`.

| Dónde | Cómo | Dispositivo |
|---|---|---|
| VM Linux / servidor | NixOS, o `docker compose --profile pesado up -d` | `cpu` |
| Mac con Apple Silicon | `./scripts/voz-stream-mac.sh` (**nativo**) | `mps` |
| PC con NVIDIA | `docker compose --profile gpu up -d --build` | `cuda` |

En CPU el modelo se cuantiza a int8; en GPU va en fp16 sin cuantizar, porque
el int8 dinámico de PyTorch **solo está implementado para CPU**. Son dos
caminos distintos, no un parámetro.

### En el Mac tiene que ser nativo

Los contenedores en macOS corren dentro de una VM Linux que **no tiene acceso
a Metal**. Desde un contenedor, `torch.backends.mps.is_available()` siempre da
`False`, se configure lo que se configure. Por eso VibeVoice va nativo en el
Mac; el resto del stack sí funciona por Docker y se compila para ARM:

```bash
cd docker && docker compose up -d   # Piper + whisper en el 8080
./scripts/voz-stream-mac.sh         # VibeVoice en el 8082
```

La consola del 8080 encuentra el 8082 sola.

### Con una NVIDIA no basta con tener la tarjeta

El lock fija la rueda de torch **sin CUDA compilado**, así que
`torch.cuda.is_available()` daría `False` por muy buena que fuera la GPU. El
perfil `gpu` usa el mismo Dockerfile pero sustituye torch por la rueda de
`cu130` — misma versión 2.13.0, así que cambia el backend y no la versión.

Hacen falta el driver de NVIDIA y `nvidia-container-toolkit` en el anfitrión.
Compruébalo antes:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

## Ajustes que importan

```nix
services.voz-api = {
  voces = [ "es_MX-claude-high" "es_MX-ald-medium" "es_ES-davefx-medium" ];
  vozDefecto = "es_MX-claude-high";
  promptSTT = "Vocabulario tecnico: ... tu jerga aqui ...";
};

services.homelab-whisper.modelo = "small";  # o "base", 3x más rápido y peor

services.vibevoice.cfgScale = 1.5;  # calidad, no velocidad — ver abajo
```

### `cfgScale` no acelera nada (medido)

Parecía el ajuste obvio para ganar velocidad y **resultó que no**:

| `cfg_scale` | RTF | Audio generado |
|---|---|---|
| 1.5 | **3,92** | 10,93 s |
| 1.3 | 4,02 | 11,87 s |
| 1.0 | 4,20 | 17,07 s ← divaga |

El motivo está en `sample_speech_tokens`: concatena condicional e incondicional
en un mismo batch **siempre**, sin rama que se salte el segundo. Se parcheó para
saltárselo con `cfg_scale == 1.0` y tampoco sirvió — **RTF 3,90**, dentro del
ruido. Con dim 896 y batch 2 el cuello es el ancho de banda de memoria, no los
FLOPs, así que la segunda mitad del batch sale casi gratis.

**El único lever real fue el número de núcleos**: de 4 a 8 cores el RTF bajó de
4,80 a 3,92 (~18%). Déjalo en 1.5.

### Voces disponibles

Las cinco están medidas con el mismo texto. `es_MX-claude-high` es la de por
defecto por ser latinoamericana y la más rápida de las de calidad `high`.

| Voz | Origen | RTF | Tamaño |
|---|---|---|---|
| `es_MX-claude-high` | México | 0,152 | 61 MB |
| `es_MX-ald-medium` | México | 0,151 | 61 MB |
| `es_ES-sharvard-medium` | España | 0,177 | 74 MB |
| `es_ES-davefx-medium` | España | 0,191 | 61 MB |
| `es_AR-daniela-high` | Argentina | 0,408 | 109 MB |

## Recursos mínimos

VibeVoice hace pico de **3,9 GB** cargando el modelo en fp32 — en CPU no baja de
ahí. Por eso el Terraform pide 6 GB por defecto y valida que no bajes de 4 GB.
Con menos, el sistema se va a swap y un RTF ya malo se vuelve inservible.

## Estado

- [x] El flake evalúa y produce el sistema completo
- [x] Los dos `uv.lock` resuelven, con torch CPU y sin CUDA
- [x] Voces y pesos con hash verificado contra los ficheros reales
- [x] **`voz-api` construye** y sus rutas cargan (`/tts`, `/stt`, `/health`, `/voces`)
- [x] **`vibevoice-env` construye** y genera audio en español desde el store (RTF 4,24x)
- [ ] Despliegue de punta a punta contra Proxmox

### Lo que costó que el build funcionara

El flake evaluaba limpio desde el principio y aun así tenía **tres fallos de
construcción**. Evaluar no es construir:

| Fallo | Causa | Arreglo |
|---|---|---|
| `No module named 'setuptools'` | uv2nix compila sin aislamiento y VibeVoice usa `setuptools.build_meta` | `resolveBuildSystem { setuptools = []; }` |
| `libtbb.so.12` no satisfecha en `numba` | la rueda enlaza oneTBB sin declararlo | añadir `pkgs.tbb` a `buildInputs` |
| `cannot load library 'libsndfile.so'` | `soundfile` hace `dlopen` **por nombre en ejecución**, que autoPatchelf no ve | sustituir por la ruta absoluta del store |

Un cuarto, ya en ejecución: el script de inferencia busca las voces junto a sí
mismo (`dirname(__file__)`), no en el directorio de trabajo. Al vivir en el
store no encontraba ninguna. Se parchea `voices_dir` en la derivación.

Nota: el entorno resuelve **transformers 4.57.6**, no la 4.51.3 que pinea el
extra `streamingtts` del repo original. Se probó y genera audio correctamente,
pero es la diferencia a mirar primero si algo se rompe tras un `uv lock`.

### Construir sin una máquina x86_64

Si trabajas desde un Mac ARM, necesitas un x86_64-linux para construir. Dos
detalles que muerden al instalar Nix en un LXC de Proxmox:

- El **sandbox funciona** en contenedores no privilegiados; no hace falta
  `sandbox = false`.
- `channels.nixos.org` resuelve **solo a IPv6**. Si el contenedor no tiene ruta
  IPv6, cualquier `nixpkgs#loquesea` falla con "Could not resolve host". Se
  desactiva con `flake-registry = ` (vacío) en `/etc/nix/nix.conf`; este repo no
  lo necesita porque tiene `flake.lock`.

## Documentación

Este README es el resumen. El detalle está en [`docs/`](docs/):

| Documento | Qué contiene |
|---|---|
| [arquitectura.md](docs/arquitectura.md) | las capas del flake, el overlay `uv2nix`, los modelos como paquetes, el ciclo de vida de una petición y el endurecimiento de los servicios |
| [despliegue.md](docs/despliegue.md) | el ciclo `provision → install → update` paso a paso, cómo actualizar, cómo volver atrás y los problemas frecuentes |
| [api.md](docs/api.md) | referencia HTTP de `/tts`, `/stt`, `/voces` y `/health`, con ejemplos, cabeceras de métricas y tabla de errores |
| [opciones.md](docs/opciones.md) | todas las opciones NixOS, sus aserciones, las variables de entorno y configuraciones de ejemplo |
| [rendimiento.md](docs/rendimiento.md) | todas las mediciones: Piper contra VibeVoice, `small` contra `base`, la iGPU contra la CPU y qué sí movió la aguja |

## Licencias

VibeVoice es MIT (Microsoft). Las voces de Piper tienen cada una la suya, en su
`.onnx.json`. Los modelos de whisper.cpp son MIT.
