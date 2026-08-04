# Referencia de opciones

Todas las opciones NixOS que define este repositorio, más las variables de entorno que acaban en los
servicios. Los valores por defecto son los que se usan en la máquina del proyecto.

- [`homelab.*` — lo específico de tu máquina](#homelab--lo-específico-de-tu-máquina)
- [`services.voz-api.*`](#servicesvoz-api)
- [`services.homelab-whisper.*`](#serviceshomelab-whisper)
- [`services.vibevoice.*`](#servicesvibevoice)
- [Voces de Piper disponibles](#voces-de-piper-disponibles)
- [Aserciones y avisos](#aserciones-y-avisos)
- [Variables de entorno](#variables-de-entorno)
- [Ejemplos de configuración](#ejemplos-de-configuración)

---

## `homelab.*` — lo específico de tu máquina

Definidas en [`nix/options.nix`](../nix/options.nix), se rellenan en [`nix/host.nix`](../nix/host.nix). Es
el único fichero que hay que editar al clonar el repositorio.

### `homelab.clavesSSH`

**Tipo** `listOf str` · **Por defecto** `[ ]`

Claves públicas con acceso a la VM, tanto para `root` como para el usuario `juan`.

**Sin al menos una, el sistema no evalúa.** Hay una aserción que lo impide, porque la máquina quedaría
inaccesible tras instalar: no hay contraseñas ni consola configurada.

```nix
homelab.clavesSSH = [ "ssh-ed25519 AAAA... juan@mac" ];
```

### `homelab.disco`

**Tipo** `str` · **Por defecto** `/dev/vda`

Disco donde `disko` instala el sistema. Depende de la controladora que elijas en Proxmox:

| Controladora en Proxmox | Disco |
|---|---|
| VirtIO SCSI | `/dev/sda` |
| VirtIO Block | `/dev/vda` |

> **Compruébalo con `lsblk` antes del primer despliegue: `disko` formatea lo que le digas, sin preguntar.**

El particionado resultante está en [`nix/disko.nix`](../nix/disko.nix) y no es configurable: GPT, una ESP
de 512 MB en `/boot`, el resto en ext4 sobre `/`, y un fichero de intercambio de 4 GB en
`/var/lib/swapfile`. Sin LVM ni cifrado a propósito — la VM se reconstruye desde el flake, no se repara.

---

## `services.voz-api.*`

El servicio de producción: TTS con Piper y fachada HTTP para el STT.
Módulo: [`nix/modules/voz-api.nix`](../nix/modules/voz-api.nix).

| Opción | Tipo | Por defecto | Qué hace |
|---|---|---|---|
| `enable` | `bool` | `false` | Levanta el servicio. |
| `puerto` | `port` | `8080` | Puerto HTTP. |
| `direccion` | `str` | `"0.0.0.0"` | Interfaz de escucha. `127.0.0.1` lo deja solo para la propia máquina. |
| `voces` | `listOf enum` | `[ "es_MX-claude-high" "es_MX-ald-medium" "es_ES-davefx-medium" ]` | Voces a instalar; solo valen las [del catálogo](#voces-de-piper-disponibles). |
| `vozDefecto` | `str` | `"es_MX-claude-high"` | La que se usa cuando la petición no dice ninguna. |
| `promptSTT` | `str` | lista de jerga del homelab | Sesga el vocabulario de whisper. |
| `ficheroToken` | `nullOr path` | `null` | Fichero con la línea `VOZ_TOKEN=…`. |
| `abrirCortafuegos` | `bool` | `false` | Abre el puerto en la LAN. |

### Notas sobre las que tienen truco

**`voces`** es un `enum` sobre el catálogo, así que escribir mal el nombre de una voz es un error de
evaluación con la lista de opciones válidas, no un fallo en tiempo de ejecución. Cada voz `high` ocupa
entre ~60 y 109 MB, así que conviene no instalarlas todas si el disco va justo.

**`promptSTT`** es el ajuste que más cambia la calidad del reconocimiento. Sin él, *«WireGuard»* se
transcribe *«We The War»* y *«homelab»* se convierte en *«omelab»*. Merece la pena poner ahí la jerga
propia: nombres de máquinas, de servicios, de personas.

**`ficheroToken`** **no debe estar en el `/nix/store`**, que es legible por cualquier usuario del sistema.
Se lee en el arranque vía `EnvironmentFile`. La configuración del repositorio apunta a
`/var/lib/voz/token.env`, que genera solo el servicio `voz-token` en el primer arranque con
`openssl rand -hex 24` y permisos `600`. Si es `null`, la API queda abierta.

**`abrirCortafuegos`** exige `ficheroToken`; ver [aserciones](#aserciones-y-avisos).

---

## `services.homelab-whisper.*`

El STT: `whisper.cpp` como servidor residente en loopback.
Módulo: [`nix/modules/whisper.nix`](../nix/modules/whisper.nix).

| Opción | Tipo | Por defecto | Qué hace |
|---|---|---|---|
| `enable` | `bool` | `false` | Levanta `whisper-server`. |
| `modelo` | `enum [ "base" "small" ]` | `"small"` | Modelo `ggml`. |
| `idioma` | `str` | `"es"` | Idioma por defecto; `auto` para detectarlo. |
| `puerto` | `port` | `8081` | **Solo en `127.0.0.1`**; no se abre nunca en el cortafuegos. |
| `hilos` | `int` | `6` | Hilos de inferencia. |

### Elegir modelo

Medido sobre 7,72 s de audio en español, en un i7-8700T:

| Modelo | Tiempo | RTF | Tamaño | Veredicto |
|---|---|---|---|---|
| `base` | 1,53 s | 0,198 | ~148 MB | 3× más rápido, pero falla en términos técnicos |
| `small` | 5,18 s | 0,671 | 466 MB | acierta «Proxmox» y «backup» — **el elegido** |

**No hay opción de GPU y es deliberado.** Se probó la iGPU Intel UHD 630 con el backend Vulkan y resultó
**2,5× más lenta que la CPU** (`matrix cores: none`). El detalle está en [rendimiento.md](rendimiento.md).

**Sobre `hilos`:** conviene dejar alguno libre para que `voz-api` pueda responder mientras whisper
transcribe. El valor 6 en una máquina de 6 núcleos / 12 hilos es esa reserva.

---

## `services.vibevoice.*`

El laboratorio. **No levanta ningún servicio**: instala una orden `vibevoice` que se lanza a mano.
Módulo: [`nix/modules/vibevoice.nix`](../nix/modules/vibevoice.nix).

| Opción | Tipo | Por defecto | Qué hace |
|---|---|---|---|
| `enable` | `bool` | `false` | Instala la orden `vibevoice` en el sistema. |
| `hilos` | `int` | `8` | Hilos de OpenMP para la inferencia en CPU. |
| `cfgScale` | `float` | `1.5` | Escala del *classifier-free guidance*. |

**`cfgScale` es la palanca de velocidad.** Con CFG activo, cada paso de difusión hace dos pasadas
—condicional e incondicional—, así que bajarlo a `1.0` casi duplica la velocidad a cambio de
expresividad. `1.5` es el valor original del modelo.

Las dos se pueden pisar por llamada sin reconstruir el sistema:

```bash
VIBEVOICE_HILOS=12 VIBEVOICE_CFG=1.0 vibevoice --txt_path guion.txt --speaker_names sp-Spk0_woman
```

La orden prepara el entorno antes de invocar el script: crea un directorio temporal con las voces
enlazadas donde el script las busca (`./demo/voices/…`), fija `OMP_NUM_THREADS` y pone `HF_HUB_OFFLINE=1`
para que no intente salir a internet. El modelo se le pasa como ruta del store.

**Voces disponibles:** las españolas son `sp-Spk0_woman` y `sp-Spk1_man`, añadidas por Microsoft en
diciembre de 2025 y marcadas por ellos como experimentales. Los modelos 1.5B y Large-7B solo hablan inglés
y chino; este `Realtime-0.5B` es el único con español.

---

## Voces de Piper disponibles

El catálogo está en [`nix/pkgs/piper-voices.nix`](../nix/pkgs/piper-voices.nix). Cinco voces en español,
cada una fijada por el hash de su `.onnx` y de su `.onnx.json`.

| Voz | Región | Calidad | RTF relativo | Notas |
|---|---|---|---|---|
| `es_MX-claude-high` | México | high | 0,152 | **la voz por defecto**: latinoamericana y la más rápida de las `high` |
| `es_MX-ald-medium` | México | medium | 0,151 | la más rápida del catálogo |
| `es_ES-sharvard-medium` | España | medium | 0,177 | |
| `es_ES-davefx-medium` | España | medium | 0,191 | |
| `es_AR-daniela-high` | Argentina | high | 0,408 | la más pesada: 109 MB |

> **Sobre estas cifras.** Se midieron con el mismo texto (~10 s de audio) para **comparar las voces entre
> sí**; úsalas como orden relativo. El repositorio documenta aparte un RTF de **0,042** para el servicio
> con la voz ya cargada en memoria, que es el coste que se ve en producción a partir de la segunda
> petición de cada voz.

Instalar otras es cuestión de añadir una entrada al catálogo con su ruta en el repositorio de
[rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) y los dos hashes. Un hash mal puesto
hace fallar la compilación, no instala otra cosa en silencio.

---

## Aserciones y avisos

El sistema se niega a construirse en tres casos, todos ellos errores que se notarían tarde y mal.

| Dónde | Condición | Por qué |
|---|---|---|
| `configuration.nix` | `homelab.clavesSSH != [ ]` | la VM se instalaría sin ninguna forma de entrar |
| `voz-api.nix` | `vozDefecto ∈ voces` | la voz por defecto no estaría instalada; fallaría en la primera petición |
| `voz-api.nix` | `abrirCortafuegos → ficheroToken != null` | **abrir el puerto sin token deja el TTS y el STT accesibles a cualquiera de la LAN** |
| `piper-voices.nix` | las voces pedidas están en el catálogo | mensaje con los nombres desconocidos y la lista de válidos |

Y un aviso que no bloquea:

> VibeVoice necesita ~4 GB de RAM libres durante la generación. Si esta VM tiene menos de 6 GB, lanzarlo
> mientras `voz-api` sirve peticiones puede empujar el sistema a swap.

---

## Variables de entorno

### Las que inyecta el módulo en `voz-api`

No hace falta tocarlas: existen para que el fichero Python no sepa nada de Nix y se pueda lanzar a mano
para depurar.

| Variable | Origen |
|---|---|
| `VOZ_VOICES_DIR` | directorio del store con las voces seleccionadas |
| `VOZ_DEFECTO` | `services.voz-api.vozDefecto` |
| `VOZ_WHISPER_URL` | `http://127.0.0.1:<puerto de whisper>` |
| `VOZ_PROMPT_STT` | `services.voz-api.promptSTT` |
| `VOZ_FFMPEG` | ruta del `ffmpeg` del store |
| `VOZ_HOST` / `VOZ_PORT` | `direccion` y `puerto` |
| `VOZ_TOKEN` | **no** viene del módulo: entra por `EnvironmentFile` desde `ficheroToken` |
| `VOZ_LOG_LEVEL` | no la fija el módulo; el programa usa `info` si no está |

### Las de la orden `vibevoice`

| Variable | Efecto |
|---|---|
| `VIBEVOICE_HILOS` | pisa `services.vibevoice.hilos` en esa ejecución |
| `VIBEVOICE_CFG` | pisa `services.vibevoice.cfgScale` en esa ejecución |

---

## Ejemplos de configuración

### La del repositorio

Tal cual está en [`nix/configuration.nix`](../nix/configuration.nix): los tres motores activos, la API
abierta a la LAN con token.

```nix
services.homelab-whisper = {
  enable = true;
  modelo = "small";
  idioma = "es";
  hilos  = 6;
};

services.voz-api = {
  enable = true;
  puerto = 8080;
  voces = [ "es_MX-claude-high" "es_MX-ald-medium" "es_ES-davefx-medium" ];
  vozDefecto = "es_MX-claude-high";
  abrirCortafuegos = true;
  ficheroToken = "/var/lib/voz/token.env";
};

services.vibevoice = {
  enable = true;
  hilos = 8;
};
```

### Mínima: solo TTS, sin salir de la máquina

Para una VM pequeña que solo sintetiza voz para algo que corre en el mismo host.

```nix
services.voz-api = {
  enable = true;
  direccion = "127.0.0.1";      # ni siquiera escucha en la LAN
  voces = [ "es_MX-ald-medium" ];  # la más ligera y la más rápida
  vozDefecto = "es_MX-ald-medium";
};
# whisper y vibevoice, desactivados: no se instalan sus modelos
```

Sin `homelab-whisper`, `/stt` responde `503` pero `/tts` funciona con normalidad.

### Ligera: prioriza la latencia sobre la precisión

```nix
services.homelab-whisper = {
  enable = true;
  modelo = "base";              # 3x más rápido; falla en jerga técnica
  hilos = 4;
};

services.voz-api = {
  enable = true;
  voces = [ "es_MX-ald-medium" ];
  vozDefecto = "es_MX-ald-medium";
  ficheroToken = "/var/lib/voz/token.env";
  abrirCortafuegos = true;
  # Con `base`, un buen prompt compensa parte de lo que pierdes de modelo.
  promptSTT = "Vocabulario: Proxmox, NixOS, WireGuard, systemd, backup, contenedor.";
};

services.vibevoice.enable = false;   # ahorra ~4 GB de disco
```

### Multivoz, para distinguir contextos

```nix
services.voz-api = {
  enable = true;
  voces = [
    "es_MX-claude-high"      # respuestas normales
    "es_ES-sharvard-medium"  # lecturas largas
    "es_AR-daniela-high"     # alertas
  ];
  vozDefecto = "es_MX-claude-high";
  ficheroToken = "/var/lib/voz/token.env";
  abrirCortafuegos = true;
};
```

El cliente elige con el campo `voz` en cada petición a `/tts`; ver [api.md](api.md#post-tts--texto-a-voz).
