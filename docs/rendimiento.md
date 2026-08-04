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
VibeVoice · TTS       48,0 s   ████████████████████████████████████████████████
```

Esa última fila es toda la explicación de por qué el repositorio se llama VibeVoiceNix pero la producción
la sirve Piper.

---

## TTS: Piper contra VibeVoice

| | **Piper** | **VibeVoice-Realtime-0.5B** |
|---|---|---|
| RTF | **0,042** (voz ya en memoria) | **4,80** |
| Cómputo por 9 s de audio | ~0,4 s | **44 s** |
| Pico de RAM | decenas de MB | **3,9 GB** |
| Tamaño del modelo | 60–109 MB por voz | 1,9 GB en fp32 |
| Español | sí, cinco voces | experimental, dos voces |
| Papel en el proyecto | **producción** | laboratorio |

La diferencia es de **~100×**. No es un ajuste que se pueda cerrar afinando parámetros: son dos familias de
modelos distintas. Piper es un modelo VITS pequeño que se ejecuta de una pasada sobre ONNX Runtime;
VibeVoice es un modelo de difusión que hace decenas de pasos iterativos, y en CPU cada paso se paga entero.

**La palanca que existe** es `cfgScale`. Con *classifier-free guidance* activo cada paso de difusión hace
dos pasadas —una condicional y otra incondicional—, así que bajarlo de `1.5` a `1.0` casi duplica la
velocidad a cambio de expresividad. Aun así la cuenta deja el RTF en torno a **2,4**: seguiría siendo el
doble de lento que el tiempo real.

**Por eso VibeVoice no es un servicio.** El módulo instala una orden que se lanza a mano y avisa de que
necesita ~4 GB libres. Un servicio que escuchara peticiones competiría por la RAM con `voz-api` y
empujaría la VM a swap justo cuando hay que responder.

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
probó `whisper.cpp` con el backend **Vulkan** y el resultado fue:

> **2,5× MÁS LENTA que la CPU.**

La causa está en las capacidades que reporta el propio dispositivo: `matrix cores: none`. Una iGPU de esa
generación no tiene unidades de multiplicación de matrices, que es exactamente la operación que domina la
inferencia. Sin ellas, el sobrecoste de mover los tensores a la GPU y traerlos de vuelta se come cualquier
ganancia de paralelismo.

Las consecuencias recorren todo el repositorio:

- El módulo de whisper **no tiene opción de GPU**. No es un olvido: no habría razón para usarla.
- `torch` se instala desde el índice `pytorch-cpu`. La rueda de CUDA pesa ~2,5 GB y no hay GPU NVIDIA en
  esta máquina.
- La orden `vibevoice` fuerza `--device cpu`.

Con una GPU NVIDIA la conclusión sería otra, y VibeVoice podría dejar de ser un laboratorio. Con esta, no.

---

## Comparativa de voces

Cinco voces en español, mismo texto (~10 s de audio):

| Voz | Región | Calidad | RTF | Notas |
|---|---|---|---|---|
| `es_MX-ald-medium` | México | medium | **0,151** | la más rápida |
| `es_MX-claude-high` | México | high | **0,152** | **la voz por defecto**: la más rápida de las `high` |
| `es_ES-sharvard-medium` | España | medium | 0,177 | |
| `es_ES-davefx-medium` | España | medium | 0,191 | |
| `es_AR-daniela-high` | Argentina | high | 0,408 | 2,7× más lenta; 109 MB |

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
[`nix/disko.nix`](../nix/disko.nix) como red de seguridad, y el aviso que emite el módulo de VibeVoice
cuando se activa junto a `voz-api`.

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

  SI EL TTS FUERA    ███████▓▓█████████████████████████████████████████   55,5 s
  VIBEVOICE          STT      TTS · VibeVoice
                     6,7 s    48 s
```

Con el stack tal como está, el usuario espera **menos de diez segundos** y la mayor parte es la
transcripción. Con VibeVoice sirviendo el TTS, el mismo intercambio pasaría del minuto —y el agente dejaría
de ser útil para lo que se construyó.

Otras referencias rápidas:

- **Una nota de voz de 10 s sintetizada con Piper:** ~0,4 s. Menos de lo que tarda en llegar por la red.
- **Transcribir un audio de un minuto:** ~40 s con `small`, ~12 s con `base`.
- **Un párrafo de 30 s leído por VibeVoice:** ~2,5 minutos de cómputo.

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
