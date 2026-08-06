# Referencia de opciones

Todas las opciones NixOS que define este repositorio, más las variables de entorno que acaban en los
servicios. Los valores por defecto son los que se usan en la máquina del proyecto.

- [`homelab.*` — lo específico de tu máquina](#homelab--lo-específico-de-tu-máquina)
- [`services.voz-api.*`](#servicesvoz-api)
- [`services.homelab-whisper.*`](#serviceshomelab-whisper)
- [`services.vibevoice.*`](#servicesvibevoice)
- [`services.voz-stream.*`](#servicesvoz-stream)
- [`services.vibevoice-ov.*`](#servicesvibevoice-ov)
- [`homelab.tunel.*`](#homelabtunel)
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
evaluación con la lista de opciones válidas, no un fallo en tiempo de ejecución. Van de 61 a 109 MB por
voz, así que conviene no instalarlas todas si el disco va justo.

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
| `hilos` | `int` | `6` | Hilos de OpenMP. **Más no es mejor**: 12 hilos van un 24 % peor. |
| `anclarNucleos` | `bool` | `true` | `OMP_PLACES=cores`. ⚠️ **Invertir si el motor pasa a OpenVINO.** |
| `cuantizar` | `bool` | `true` | int8 dinámico: casi 2× más rápido. |
| `pasosDifusion` | `int` | `6` | Pasos del *scheduler*. 4 solo mejora un 3 %. |
| `vozDefecto` | `str` | `"sp-Spk1_man"` | Hablante. Las españolas son `sp-Spk1_man` y `sp-Spk0_woman`. |
| `cfgScale` | `float` | `1.5` | Escala del *classifier-free guidance*. **Calidad, no velocidad.** |

**`hilos` y `anclarNucleos` son la pareja delicada.** Medido: 2 hilos RTF 4,19 · 6 anclados **4,04** ·
8 hilos 4,31 · 12 hilos **5,18 (24 % peor)**. Y el anclaje **acelera PyTorch un 3 % pero ralentiza
OpenVINO un 118 %** (89 → 195 ms/llamada): si algún día `voz-stream` deja de caer a torch, hay que
invertirlo.

**`cuantizar` y `pasosDifusion` son las dos palancas que sí pagaron:** juntas llevaron el RTF de 5,39 a
2,18. El viaje completo está en [optimizacion.md](optimizacion.md).

**`cfgScale` no acelera nada, y está medido.** Parecía la palanca obvia —con CFG cada paso de difusión
hace una pasada condicional y otra incondicional— pero bajarlo sale peor por los dos lados:

| `cfg_scale` | RTF | Audio generado |
|---|---|---|
| 1.5 | **3,92** | 10,93 s |
| 1.3 | 4,02 | 11,87 s |
| 1.0 | 4,20 | 17,07 s ← divaga |

`sample_speech_tokens` concatena condicional e incondicional en un mismo batch **siempre**, sin rama que
se salte la segunda; parchearlo tampoco sirvió (RTF 3,90). Con dimensión 896 y batch 2 el cuello es el
ancho de banda de memoria, no los FLOPs.

> ⚠️ **Pero sí conviene subirlo, por calidad.** Un banco de fidelidad posterior (texto → voz → whisper →
> texto) midió que **3,0 baja el WER medio del 13,6 % al 3,6 %** y el peor caso de 85,7 % a 14,3 % — **sin
> coste en tiempo**, porque la difusión evalúa las dos ramas en un lote de 2 pase lo que pase. El servicio
> `voz-stream` ya usa **3.0**; esta opción de la CLI sigue en 1.5 y su descripción no se actualizó.
> Conviene alinearlas. Ver [optimizacion.md](optimizacion.md#el-viaje-completo).

**`hilos` sí importa, pero al revés de lo que parece:** el óptimo son **6 anclados** (RTF 4,04). Con 8
sube a 4,31 y con 12 a 5,18 — un 24 % peor. Más hilos compiten por el mismo bus de memoria, que es el
cuello real.

Las dos se pueden pisar por llamada sin reconstruir el sistema:

```bash
VIBEVOICE_PASOS=4 vibevoice --texto "Compara la calidad." --salida cuatro.wav
```

La orden es un envoltorio fino: fija `OMP_NUM_THREADS`, pone `HF_HUB_OFFLINE=1` para que no intente salir
a internet, y le pasa el modelo como ruta del store. No prepara ningún directorio de trabajo — el script
de inferencia lleva la ruta de las voces sustituida en la propia derivación.

**Voces disponibles:** las españolas son `sp-Spk0_woman` y `sp-Spk1_man`, añadidas por Microsoft en
diciembre de 2025 y marcadas por ellos como experimentales. Los modelos 1.5B y Large-7B solo hablan inglés
y chino; este `Realtime-0.5B` es el único con español.

---

## `services.voz-stream.*`

TTS expresivo **en streaming**: emite el audio según se genera, así que el primer sonido llega en 0,20 s
en vez de esperar a la síntesis completa.
Módulo: [`nix/modules/voz-stream.nix`](../nix/modules/voz-stream.nix).

| Opción | Tipo | Por defecto | Qué hace |
|---|---|---|---|
| `enable` | `bool` | `false` | Levanta el servicio de streaming. |
| `puerto` | `port` | `8082` | Puerto HTTP. |
| `direccion` | `str` | `"0.0.0.0"` | Interfaz de escucha. |
| `abrirCortafuegos` | `bool` | `false` | Abre el puerto en la LAN. Con el túnel activo no hace falta. |
| `ficheroToken` | `nullOr path` | `null` | Igual que en `voz-api`: fuera del store. |

**Es un servicio aparte de `voz-api` a propósito.** Este carga VibeVoice (~2,3 GB); `voz-api` solo las
voces de Piper (~100 MB). Juntarlos haría que una síntesis pesada bloqueara las notas de voz rápidas.

Hereda de `services.vibevoice` el modelo, las voces y la configuración de pasos de difusión, así que no
hay dos sitios donde ajustar lo mismo. Usa `cfg_scale = 3.0` por petición, que es el valor que midió el
banco de fidelidad.

**El audio es bit a bit idéntico** al de la generación normal (mismo md5): no es una versión degradada,
es el mismo resultado entregado según se produce.

```bash
curl -sN -X POST http://voz:8082/tts -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{"texto":"Se oye según se genera."}' | ffplay -autoexit -nodisp -
```

---

## `services.vibevoice-ov.*`

Genera los grafos **OpenVINO** (IR) que aceleran el motor.
Módulo: [`nix/modules/vibevoice-ov.nix`](../nix/modules/vibevoice-ov.nix).

| Opción | Tipo | Por defecto | Qué hace |
|---|---|---|---|
| `enable` | `bool` | `false` | Activa el `oneshot` que convierte el modelo a IR. |
| `directorioIR` | `path` | `/var/lib/voz/ov` | Dónde se dejan los grafos. |
| `precisionLM` | `str` | `"int4"` | Precisión del *backbone*. |
| `precisionCabeza` | `str` | `"int8"` | Precisión de la cabeza de difusión. |

**Los IR no viven en el `/nix/store`, y es deliberado.** Generarlos pica **4,6 GB** y el contenedor
constructor tiene 2560 MB. Se generan en la VM desde entradas fijadas —modelo con hash, scripts
versionados, `openvino` y `nncf` clavados a versión **exacta**—, así que es reproducible **el resultado**,
no el momento. Es el único artefacto derivado del proyecto que no es una derivación de Nix.

**`precisionCabeza` va en int8 y no int4 a propósito:** con semilla fija se midió que el int4 **sesga el
fin de frase** (95 tokens frente a 84 de la base) y encima empata en RTF, así que no compra nada.

> Si el servicio no llegó a generar los IR, `voz-stream` **cae a torch en silencio** — RTF 2,2 en vez de
> 1,09, que se nota como cortes en el streaming.

---

## `homelab.tunel.*`

Túnel **WireGuard** hacia un edge, para llegar a la VM desde fuera estando tras CGNAT.
Módulo: [`nix/modules/tunel.nix`](../nix/modules/tunel.nix).

| Opción | Tipo | Por defecto | Qué hace |
|---|---|---|---|
| `enable` | `bool` | `false` | Levanta el túnel. |
| `ip` | `str` | — | IP de la VM dentro del túnel (p. ej. `10.10.10.5`). |
| `ficheroClave` | `path` | `/var/lib/wireguard/privada` | Clave privada, **fuera del store**, permisos `600`. |
| `clavePublicaEdge` | `str` | — | Clave pública del edge. |
| `endpoint` | `str` | — | Dónde escucha el edge. |
| `redTunel` | `str` | `"10.10.10.0/24"` | Red del túnel. |

**La VM abre el túnel hacia el edge**, no al revés: es lo que permite atravesar el CGNAT. **No expone nada
a internet** — la API sigue escuchando solo en la LAN y en la red del túnel.

La clave privada vive fuera del `/nix/store` por el mismo motivo que el token: el store es legible por
cualquier usuario del sistema.

---

## Voces de Piper disponibles

El catálogo está en [`nix/pkgs/piper-voices.nix`](../nix/pkgs/piper-voices.nix). Cinco voces en español,
cada una fijada por el hash de su `.onnx` y de su `.onnx.json`.

| Voz | Región | Calidad | RTF relativo | Tamaño |
|---|---|---|---|---|
| `es_MX-claude-high` | México | high | 0,152 | 61 MB — **la voz por defecto**: latinoamericana y la más rápida de las `high` |
| `es_MX-ald-medium` | México | medium | 0,151 | 61 MB — la más rápida del catálogo |
| `es_ES-sharvard-medium` | España | medium | 0,177 | 74 MB |
| `es_ES-davefx-medium` | España | medium | 0,191 | 61 MB |
| `es_AR-daniela-high` | Argentina | high | 0,408 | 109 MB — la más pesada |

> **Sobre estas cifras.** Se midieron con el mismo texto (~10 s de audio) para **comparar las voces entre
> sí**; úsalas como orden relativo. El repositorio documenta aparte un RTF de **0,042** para el servicio
> con la voz ya cargada en memoria, que es el coste que se ve en producción a partir de la segunda
> petición de cada voz.

Instalar otras es cuestión de añadir una entrada al catálogo con su ruta en el repositorio de
[rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) y los dos hashes. Un hash mal puesto
hace fallar la compilación, no instala otra cosa en silencio.

---

## Aserciones y avisos

El sistema se niega a construirse en estos casos, todos ellos errores que se notarían tarde y mal.

| Dónde | Condición | Por qué |
|---|---|---|
| `configuration.nix` | `homelab.clavesSSH != [ ]` | la VM se instalaría sin ninguna forma de entrar |
| `voz-api.nix` | `vozDefecto ∈ voces` | la voz por defecto no estaría instalada; fallaría en la primera petición |
| `voz-api.nix` | `abrirCortafuegos → ficheroToken != null` | **abrir el puerto sin token deja el TTS y el STT accesibles a cualquiera de la LAN** |
| `vibevoice.nix` | `swapDevices != [ ]` | VibeVoice carga ~2,8 GB y pica más al arrancar; sin swap, una VM justa se queda sin memoria |
| `piper-voices.nix` | las voces pedidas están en el catálogo | mensaje con los nombres desconocidos y la lista de válidos |

La última es la más fácil de encontrarse: si desactivas la swap que declara
[`nix/disko.nix`](../nix/disko.nix) y dejas `services.vibevoice.enable = true`, el sistema no construye
hasta que añadas `swapDevices` o apagues el laboratorio.

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

services.vibevoice.enable = false;   # ahorra ~4 GB de disco y la asercion de swap
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
