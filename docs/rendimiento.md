# Rendimiento

Las mediciones que explican por qué el stack está montado así: por qué Piper sirve la producción y
VibeVoice no, por qué whisper usa `small` y no `base`, y por qué todo corre en CPU teniendo una GPU
integrada delante.

- [El banco de pruebas](#el-banco-de-pruebas)
- [Lo que cuesta cada motor](#lo-que-cuesta-cada-motor)
- [TTS: Piper contra VibeVoice](#tts-piper-contra-vibevoice)
- [STT: `base` contra `small`](#stt-base-contra-small)
- [Por qué no hay GPU](#por-qué-no-hay-gpu)
- [Comparativa de voces](#comparativa-de-voces)
- [Memoria y disco](#memoria-y-disco)
- [Qué significa en la práctica](#qué-significa-en-la-práctica)
- [Cómo medirlo tú](#cómo-medirlo-tú)

---

## El banco de pruebas

Todas las cifras de este documento están medidas en la máquina del proyecto:

| | |
|---|---|
| CPU | **Intel Core i7-8700T** — 6 núcleos / 12 hilos, 35 W |
| RAM | 7,7 GB utilizables en la VM, más 4 GB de swap |
| GPU | Intel UHD 630 integrada — **no se usa**, ver [más abajo](#por-qué-no-hay-gpu) |
| Sistema | NixOS (`nixos-unstable`), todo en CPU |

**RTF** (*real-time factor*) = tiempo de cómputo ÷ duración del audio.
**Por debajo de 1 es más rápido que el tiempo real.** Un RTF de 0,5 significa que procesar un minuto de
audio cuesta treinta segundos; un RTF de 2 significa que cuesta dos minutos.

---

## Lo que cuesta cada motor

Segundos de cómputo por **cada 10 segundos de audio**. Cada bloque es un segundo:

```
Piper · TTS            0,4 s   ▐
whisper base · STT     2,0 s   ██
whisper small · STT    6,7 s   ███████
VibeVoice · TTS       39,2 s   ███████████████████████████████████████
```

Esa última fila es toda la explicación de por qué el repositorio se llama VibeVoiceNix pero la producción
la sirve Piper.

---

## TTS: Piper contra VibeVoice

| | **Piper** | **VibeVoice-Realtime-0.5B** |
|---|---|---|
| RTF | **0,042** (voz ya en memoria) | **3,92** (8 núcleos) |
| Pico de RAM | decenas de MB | **3,9 GB** |
| Tamaño del modelo | 61–109 MB por voz | 1,9 GB en fp32 |
| Español | sí, cinco voces | experimental, dos voces |
| Papel en el proyecto | **producción** | laboratorio |

La diferencia es de **casi 100×**. No es un ajuste que se pueda cerrar afinando parámetros: son dos
familias de modelos distintas. Piper es un modelo VITS pequeño que se ejecuta de una pasada sobre ONNX
Runtime; VibeVoice es un modelo de difusión que hace decenas de pasos iterativos, y en CPU cada paso se
paga entero.

**Por eso VibeVoice no es un servicio.** El módulo instala una orden que se lanza a mano, y una aserción
exige que el sistema tenga swap declarada. Un servicio que escuchara peticiones competiría por la RAM con
`voz-api` y empujaría la VM a swap justo cuando hay que responder.

### El único lever real fueron los núcleos

Se probaron varias cosas para acelerarlo. **La única que movió la aguja fue dar más CPU:**

| Núcleos | RTF |
|---|---|
| 4 | 4,80 |
| 8 | **3,92** (−18%) |

Por eso el Terraform pide **8 núcleos** por defecto. Medido dentro del entorno de Nix construido desde el
store, el resultado es del mismo orden: **RTF 4,24**.

### `cfgScale` no acelera: se midió

Parecía el ajuste obvio —con *classifier-free guidance* cada paso hace una pasada condicional y otra
incondicional, así que bajarlo debería saltarse la mitad del trabajo— y **resultó que no**:

| `cfg_scale` | RTF | Audio generado |
|---|---|---|
| 1.5 | **3,92** | 10,93 s |
| 1.3 | 4,02 | 11,87 s |
| 1.0 | 4,20 | 17,07 s ← divaga |

Baja la calidad y encima va **más lento**. El motivo está en `sample_speech_tokens`: concatena condicional
e incondicional en un mismo batch **siempre**, sin ninguna rama que se salte la segunda. Se parcheó para
saltársela con `cfg_scale == 1.0` y tampoco sirvió — **RTF 3,90**, dentro del ruido.

La explicación es que con dimensión 896 y batch 2, **el cuello es el ancho de banda de memoria, no los
FLOPs**: la segunda mitad del batch sale casi gratis, así que eliminarla no ahorra nada. Déjalo en `1.5`,
que es donde da mejor calidad.

Es el tipo de resultado que solo aparece midiendo: la explicación teórica era impecable y la realidad dijo
que no.

### Sobre el español en VibeVoice

Conviene saberlo antes de invertir tiempo en el laboratorio:

- Los modelos **1.5B y Large-7B están entrenados solo con inglés y chino**.
- El `Realtime-0.5B` es el único con voces en español: `sp-Spk0_woman` y `sp-Spk1_man`.
- Microsoft las añadió en **diciembre de 2025** y las marca como **experimentales**.

---

## STT: `base` contra `small`

Medido sobre el mismo fichero: **7,72 segundos de audio en español**.

| Modelo | Tiempo | RTF | Tamaño | Calidad observada |
|---|---|---|---|---|
| `ggml-base` | 1,53 s | 0,198 | ~148 MB | rápido, pero **falla en términos técnicos** |
| `ggml-small` | 5,18 s | 0,671 | 466 MB | acierta «Proxmox» y «backup» — **el elegido** |

`base` es 3,4× más rápido. Se eligió `small` igualmente porque en un homelab la mayoría de las notas de voz
contienen exactamente las palabras que `base` no acierta, y una transcripción con el nombre del servicio
mal escrito no sirve de nada aunque llegue antes.

### El prompt vale casi tanto como el modelo

`whisper` acepta un texto inicial que sesga su vocabulario, y es **el ajuste que más cambia la calidad**:

| Sin prompt | Con prompt |
|---|---|
| «We The War» | «WireGuard» |
| «omelab» | «homelab» |

Es gratis: no cuesta tiempo de cómputo. El valor por defecto es una lista de jerga —Proxmox, LXC, Caddy,
systemd, NixOS, Terraform, Ansible…— y se configura en `services.voz-api.promptSTT`. Meter ahí los nombres
propios de tu instalación es la mejora más barata disponible.

El servidor se mantiene **residente** en vez de arrancar un proceso por petición: los 466 MB del modelo
`small` se leen una vez, no en cada nota de voz.

---

## Por qué no hay GPU

La máquina tiene una **Intel UHD 630** integrada, y la pregunta obvia es si descarga trabajo de la CPU. Se
pasó al contenedor y funciona —OpenCL 3.0, Vulkan 1.3—, pero medido con `whisper.cpp`:

| | `base` | `small` |
|---|---|---|
| **CPU, 6 hilos** | **1,53 s** | **5,18 s** |
| iGPU Vulkan | 7,65 s | 12,88 s |

> **La iGPU es 2,5× MÁS LENTA que la CPU.**

La causa está en las capacidades que reporta el propio informe de `ggml`: `matrix cores: none`, `bf16: 0`.
Una iGPU de esa generación no tiene unidades de multiplicación de matrices, que es exactamente la
operación que domina la inferencia. Sin ellas, el sobrecoste de mover los tensores a la GPU y traerlos de
vuelta se come cualquier ganancia de paralelismo.

Y por PyTorch tampoco: **esa generación nunca tendrá soporte** —XPU e IPEX cubren Arc/Xe en adelante—, y en
la máquina `torch.xpu.is_available()` devuelve `False`.

Las consecuencias recorren todo el repositorio:

- El módulo de whisper **no tiene opción de GPU**. No es un olvido: no habría razón para usarla.
- `torch` se instala desde el índice `pytorch-cpu`. La rueda de CUDA pesa ~2,5 GB y no hay GPU NVIDIA en
  esta máquina.
- La orden `vibevoice` fuerza `--device cpu`.
- El Terraform no hace *passthrough* de GPU: no habría a quién pasársela.

Con una GPU NVIDIA la conclusión sería otra, y VibeVoice podría dejar de ser un laboratorio. Con esta, no.

---

## Comparativa de voces

Cinco voces en español, mismo texto (~10 s de audio):

| Voz | Región | Calidad | RTF | Tamaño |
|---|---|---|---|---|
| `es_MX-ald-medium` | México | medium | **0,151** | 61 MB |
| `es_MX-claude-high` | México | high | **0,152** | 61 MB — **la voz por defecto** |
| `es_ES-sharvard-medium` | España | medium | 0,177 | 74 MB |
| `es_ES-davefx-medium` | España | medium | 0,191 | 61 MB |
| `es_AR-daniela-high` | Argentina | high | 0,408 | 109 MB — 2,7× más lenta |

Lo interesante es que **`claude-high` cuesta prácticamente lo mismo que una `medium`** (0,152 frente a
0,151) mientras que `daniela-high` cuesta casi el triple. La calidad `high` no implica por sí sola un coste
mayor: depende de la voz concreta. Por eso el valor por defecto es `es_MX-claude-high`, que da la mejor
calidad disponible sin pagar por ella.

> **Estas cifras sirven para comparar voces entre sí**, no como el coste que verás en producción. El
> repositorio documenta aparte un RTF de **0,042** para el servicio con la voz ya cargada en memoria, que
> es lo que se paga a partir de la segunda petición de cada voz.

---

## Memoria y disco

### RAM

| Componente | Consumo |
|---|---|
| `voz-api` con las voces cargadas | decenas de MB |
| `whisper-server` con `small` | ~500 MB residentes |
| **VibeVoice durante una generación** | **pico de 3,9 GB** |

El modelo de VibeVoice son 1,9 GB en fp32 y **en CPU no baja de ahí**: no hay cuantización en este camino.
Por debajo de 6 GB de RAM el sistema se va a swap y el RTF, que ya era malo, se vuelve inservible.

De ahí salen dos decisiones del repositorio: el **fichero de intercambio de 4 GB** que declara
[`nix/disko.nix`](../nix/disko.nix) como red de seguridad, y una **aserción** en el módulo de VibeVoice que
se niega a construir el sistema si no hay `swapDevices` declarada. El Terraform, por su parte, pide 6 GB
de RAM por defecto y valida que no bajes de 4 GB.

### Disco

| Artefacto | Tamaño |
|---|---|
| Modelo `ggml-small.bin` | 466 MB |
| Voces de Piper (las tres por defecto) | ~250 MB |
| Pesos de VibeVoice | 1,9 GB |
| Entorno Python de VibeVoice (PyTorch CPU y compañía) | ~2 GB |
| Entorno Python de `voz-api` | ~200 MB |

Desactivar `services.vibevoice` ahorra prácticamente 4 GB. A eso hay que sumarle el sistema base de NixOS
y las generaciones antiguas, que el recolector de basura borra semanalmente pasados 30 días.

---

## Qué significa en la práctica

El caso real: un agente recibe una nota de voz de 10 segundos, la entiende y contesta con otra.

```
                     0s        10s       20s       30s       40s       50s       60s
                     ├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤

  PRODUCCIÓN         ███████▓▓░                                            7,5 s
                     STT      TTS
                     6,7 s    0,4 s   (+ lo que tarde el agente en pensar)

  SI EL TTS FUERA    ███████▓▓████████████████████████████████          46,7 s
  VIBEVOICE          STT      TTS · VibeVoice
                     6,7 s    39,2 s
```

Con el stack tal como está, el usuario espera **menos de diez segundos** y la mayor parte es la
transcripción. Con VibeVoice sirviendo el TTS, el mismo intercambio pasaría del minuto —y el agente dejaría
de ser útil para lo que se construyó.

Otras referencias rápidas:

- **Una nota de voz de 10 s sintetizada con Piper:** ~0,4 s. Menos de lo que tarda en llegar por la red.
- **Transcribir un audio de un minuto:** ~40 s con `small`, ~12 s con `base`.
- **Un párrafo de 30 s leído por VibeVoice:** ~2 minutos de cómputo.

---

## Cómo medirlo tú

No hace falta instrumentar nada: **la API devuelve sus propias métricas en cada respuesta.**

**TTS** — las cabeceras `X-*`:

```bash
curl -s -X POST http://voz:8080/tts -H "Authorization: Bearer $VOZ_TOKEN" -H "Content-Type: application/json" -d '{"texto":"Prueba de rendimiento del sintetizador."}' -o /dev/null -D- | grep '^X-'
```

```
X-Duracion-S: 3.12
X-Proceso-S: 0.13
X-RTF: 0.042
X-Voz: es_MX-claude-high
```

**Lanza la misma petición dos veces.** La primera incluye la carga del `.onnx` desde el store; la segunda
es el coste real en producción.

**STT** — los campos del JSON:

```bash
curl -s -X POST http://voz:8080/stt -H "Authorization: Bearer $VOZ_TOKEN" -F "archivo=@muestra.ogg" | jq '{duracion_s, proceso_s, rtf}'
```

**Comparar voces** entre sí, con el mismo texto:

```bash
for v in es_MX-claude-high es_MX-ald-medium es_ES-davefx-medium; do echo -n "$v  "; curl -s -X POST http://voz:8080/tts -H "Authorization: Bearer $VOZ_TOKEN" -H "Content-Type: application/json" -d "{\"texto\":\"El backup de anoche terminó sin errores.\",\"voz\":\"$v\"}" -o /dev/null -D- | grep -i '^x-rtf'; done
```

**Comparar modelos de whisper** requiere reconstruir el sistema con `services.homelab-whisper.modelo`
cambiado, y volver a lanzar el mismo fichero contra `/stt`.
