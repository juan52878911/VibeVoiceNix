# Decisiones de optimización

VibeVoice pasó de **RTF 5,39 a 0,75** y el primer sonido de **23,21 s a 0,20 s**. Este documento cuenta
cómo, y —más útil— **qué se probó y no funcionó**, para que nadie lo repita.

Todo está medido en el mismo banco: **Intel i7-8700T** (6 núcleos, AVX2, sin AVX512, DDR4 *single
channel*), salvo donde se indique otra cosa.

> **RTF** = tiempo de cómputo ÷ duración del audio. Por debajo de 1 es más rápido que el tiempo real.

---

## El viaje completo

<details open>
<summary><b>La tabla que lo resume todo</b></summary>

<br>

| # | Cambio | RTF | Ganancia | Qué lo hizo posible |
|---|---|---|---|---|
| 0 | Punto de partida — fp32, 20 pasos | **5,39** | — | la configuración original de Microsoft |
| 1 | Cuantización int8 dinámica | **2,75** | 1,96× | los pesos pasan a ¼ del tamaño |
| 2 | 6 pasos de difusión en vez de 20 | **2,18** | 1,26× | `DPMSolverMultistepScheduler` está hecho para pocos pasos |
| 3 | Motor OpenVINO (grafos compilados) | **1,09** | 2,00× | el decodificador acústico deja de despachar desde Python |
| 4 | Reescribir las convoluciones *depthwise* | **0,75** | 1,45× | torch no trae kernel optimizado; se vectoriza a mano |
| 5 | Solapar el decodificador acústico | **0,59**¹ | 1,26× | el decodificador es un sumidero: no realimenta el bucle |

¹ Medido en un **Apple M4**, no en el i7 del banco: es el paso que falta por
confirmar en la VM. Los cuatro anteriores sí son del i7. Ver
[el detalle](#5--solapar-el-decodificador-acústico--las-dos-etapas-a-la-vez).

**Total: 7,2× más rápido** en el banco del i7, y otro **1,26×** encima cuando se
confirme el solapamiento. Y en paralelo, sin tocar el RTF:

| Cambio | Antes | Después |
|---|---|---|
| Streaming (emitir según se genera) | primer sonido **23,21 s** | **0,20 s** |
| Subir la guía CFG de 1,5 a 3,0 | WER medio **13,6 %**, peor caso 85,7 % | **3,6 %**, peor caso 14,3 % |
| Soltar pesos muertos | **3718 MB** residentes | **2832 MB** |

</details>

---

## Lo que funcionó

<details>
<summary><b>1 · Cuantización int8 dinámica</b> — 5,39 → 2,75</summary>

<br>

**La hipótesis.** El cuello está medido: la CPU alcanza **17,2 GB/s de los 21,3 teóricos (80,7 %)**, o sea
que ya exprime el bus de memoria. Si el límite es *leer pesos*, lo que paga es **reducir bytes de peso**, no
reducir operaciones.

**El escepticismo honesto.** El i7-8700T (Coffee Lake) **no tiene AVX512-VNNI**, así que la multiplicación
int8 se emula y podía comerse el ahorro. Por eso se midió en vez de asumirlo.

**El resultado.** Casi el doble de rápido. Y el usuario comparó las muestras: no distingue la salida int8
de la original.

```bash
vibevoice --texto "Prueba de cuantización." --salida int8.wav
vibevoice --texto "Prueba de cuantización." --salida fp32.wav --sin-cuantizar
```

**Una trampa por arquitectura.** `supported_engines` trae `qnnpack` **delante incluso en x86**, y qnnpack
es el backend de ARM: cogerlo en un Intel aborta con `RuntimeError: unknown architecure`. Se elige por
`platform.machine()` — x86 → `fbgemm`, ARM → `qnnpack`— y la cuantización va dentro de un `try`: es una
optimización, no un requisito. Si el backend no traga, arranca en fp32 y sirve, en vez de entrar en bucle
de reinicio.

</details>

<details>
<summary><b>2 · Bajar los pasos de difusión</b> — 2,75 → 2,18</summary>

<br>

El modelo usa `DPMSolverMultistepScheduler`, diseñado precisamente para funcionar con pocos pasos.

| Pasos | RTF | Veredicto |
|---|---|---|
| 20 | 5,39 | el original |
| 8 | 3,90 | 1,38× — menos de lo esperado |
| **6** | **2,18** | **el elegido** |
| 4 | 2,11 | solo un 3 % mejor: no compensa el riesgo de calidad |

**Por qué 20→8 rindió menos del 2× teórico:** el LLM no encoge. Solo la difusión se acorta, y el resto
marca el techo de Amdahl. Para dimensionarlo: por token, la difusión hace **20 pasos × 4 capas × batch 2 =
160 evaluaciones de capa**, frente a **24** del LLM.

Se dejan **6 y no 4** porque la diferencia es del 3 % y 6 da margen de calidad.

```bash
VIBEVOICE_PASOS=4 vibevoice --texto "Compara la calidad." --salida cuatro.wav
```

**En qué gastar lo que libera el solapamiento.** Con el decodificador fuera del
camino crítico sobra tiempo, y los pasos de difusión son donde se puede gastar
(M4, 6 hilos, mismo texto):

| | RTF |
|---|---|
| 6 pasos, síncrono — *el punto de partida* | 0,727 |
| 6 pasos, solapado | **0,609** |
| 8 pasos, solapado | 0,649 |
| **10 pasos, solapado** | **0,684** ← más pasos y **aun así más rápido** que el punto de partida |
| 12 pasos, solapado | 0,800 ← ya se pasa |

O sea: **se pueden pagar 10 pasos de difusión al precio de 6**. Lo que no está
medido es si 10 pasos *suenan* mejor que 6 —el RTF no dice nada de la calidad—,
y eso lo tiene que decir el banco de fidelidad:

```bash
VIBEVOICE_PASOS=10 python scripts/fidelidad.py
```

</details>

<details>
<summary><b>3 · Motor OpenVINO</b> — 2,18 → 1,09</summary>

<br>

Convierte el *backbone* TTS, la cabeza de difusión y el decodificador acústico a **grafos compilados**.

**El hallazgo que lo justifica:** el decodificador acústico pasó de **165 a 66 ms por llamada**. Era el
**58 % del tiempo** y —contra lo esperado— **no estaba limitado por memoria**, sino por el **despacho de
Python sobre decenas de convoluciones pequeñas**. Ese coste desaparece al compilar el grafo.

**Los IR no viven en el store, y es a propósito.** Generarlos pica **4,6 GB** y el contenedor constructor
tiene 2560 MB. Se generan en la VM con un `oneshot`, desde entradas fijadas: modelo con hash, scripts
versionados, y `openvino` y `nncf` clavados a **versión exacta** —la conversión depende de APIs concretas y
la paridad numérica se comprobó contra esas, no contra un rango—. Es reproducible **el resultado**, no el
momento. Es el único artefacto derivado del proyecto que no es una derivación de Nix.

**La cabeza va en int8 y no int4 a propósito.** Con semilla fija se midió que el int4 **sesga el fin de
frase**: 95 tokens frente a 84 de la base. Y empata en RTF, así que no compra nada.

</details>

<details>
<summary><b>4 · Reescribir las convoluciones depthwise</b> — 2,66 → 0,75</summary>

<br>

La optimización más rentable del proyecto, y salió de **mirar el perfilador** en vez de suponer.

**El diagnóstico.** `aten::_slow_conv2d_forward` se llevaba el **75 % del tiempo** con **22.434 llamadas
por `decode`**. El decodificador tiene 26 convoluciones *depthwise* de `groups=2048`, y **torch no trae
kernel optimizado para depthwise cuando falta oneDNN** —el caso en ARM y en cualquier máquina sin MKLDNN—.
Cae a la implementación de referencia y las procesa **grupo por grupo**.

**La solución.** Una depthwise es, por cada desplazamiento del kernel, multiplicar por un escalar por canal
y sumar. Vectorizado:

| | Tiempo |
|---|---|
| `Conv1d` depthwise (torch) | 36,85 ms |
| suma de 7 desplazamientos | **0,35 ms** |

**106× por capa.** RTF total 2,66 → 0,75, con el audio verificado idéntico: correlación **0,99998152**,
diferencia media 1,47e-04, misma longitud exacta.

**Y además gana a OpenVINO** (RTF 0,95 en la VM) en algo que no es velocidad: es **torch puro**. Sin
dependencias, sin paso de conversión, sin grafos que mantener, y sirve igual en Mac, VM y GPU.

</details>

<details>
<summary><b>5 · Solapar el decodificador acústico</b> — las dos etapas a la vez</summary>

<br>

**El hallazgo.** El decodificador acústico es un **sumidero**: su salida no
vuelve a entrar en el modelo. Se lee en el bucle de Microsoft
(`modeling_vibevoice_streaming_inference.py`, líneas 776-804):

```python
speech_latent  = sample_speech_tokens(...)           # cabeza de difusión
audio_chunk    = acoustic_tokenizer.decode(...)      # <- el 42% del tiempo
audio_chunks[idx].append(audio_chunk[i])             # se guarda
audio_streamer.put(audio_chunk, ...)                 # se emite
acoustic_embed = acoustic_connector(speech_latent)   # <- el LATENTE, no el audio
```

La realimentación autorregresiva pasa por `acoustic_connector(speech_latent)`.
El audio decodificado **solo se guarda y se emite**. Así que `decode()` estaba
en el camino crítico únicamente porque se llamaba de forma síncrona, no porque
el bucle necesitara su resultado.

**El reparto que lo justifica**, cronometrado dentro de una `generate()` real
(M4, torch-int8, 6 hilos, semilla 11):

| componente | llamadas | ms cada | % del total |
|---|---|---|---|
| `tts_lm` | 189 | 20,47 | 40,3 % |
| cabeza de difusión | 540 | 2,56 | 14,4 % |
| **decodificador acústico** | 90 | 45,19 | **42,4 %** |
| resto | | | 2,8 % |

Con el decodificador fuera del camino crítico el techo es
max(42,4 ; 57,6) = **1,74×**.

**Lo que se consigue de verdad**, con el mismo texto y la misma semilla:

| hilos | síncrono | solapado | gana |
|---|---|---|---|
| 2 | 1,038 | 0,893 | 14 % |
| 4 | 0,700 | **0,594** | 15 % |
| **6** | 0,739 | **0,587** | **21 %** |
| 8 | 0,785 | 0,644 | 18 % |
| 10 | 0,775 | 0,630 | 19 % |

**1,26× y el audio es idéntico bit a bit**: el mismo md5 en las 20 pasadas de la
tabla. No es una aproximación.

**Por qué sale 1,26× y no 1,74×.** Las dos etapas compiten por la misma máquina.
Aun así gana bastante porque **no compiten por lo mismo**: el bucle se pasa el
rato leyendo pesos (ancho de banda) y el decodificador es convolución, más densa
en cómputo. Solapar una etapa limitada por memoria con otra limitada por cómputo
es justo el caso en el que la tubería paga.

**Esta es la forma de aprovechar los hilos que sobran**, y no subir `hilos`
—que está medido que empeora—. Una sola etapa ya satura el bus; dos etapas
distintas, no.

**Lo que cuesta.** El primer sonido pasa de 0,12 a 0,16 s (+40 ms), que es la
profundidad de la tubería. La memoria no se mueve: se comprobó que la diferencia
de RSS entre los dos modos no sigue al modo (síncrono dio 4.014 / 5.983 / 4.922 MB
y solapado 5.296 / 5.290 / 5.443 en pasadas alternas) — es residencia de las
páginas del `mmap` del modelo, no coste del solapamiento.

**Tres detalles de implementación que no son opcionales:**

- **Un solo hilo trabajador y cola FIFO.** El decodificador es causal y con
  estado: las colas de sus convoluciones las deja la llamada anterior. Dos
  `decode()` a la vez, o en otro orden, darían otro audio.
- **Búfer preasignado, no un "futuro".** `generate()` hace
  `torch.cat(audio_chunks)` al final **pase lo que pase** —solo el `return` mira
  `return_speech`—, así que `decode()` tiene que devolver un tensor de verdad. La
  salida tiene forma fija (un latente → 3200 muestras), así que se devuelve un
  tensor vacío y el worker lo rellena in situ.
- **La emisión se muda al worker.** Si `put()` se quedara en el hilo de
  `generate()` habría que esperar ahí al decode y no se solaparía nada. Efecto
  colateral: la cancelación cooperativa ya no salta en el hilo de `generate()`,
  así que se guarda y se relanza en el `decode()` siguiente — el corte tarda como
  mucho un trozo más, ~133 ms.

> **Medido en un M4, no en el i7.** La VM es donde vive el motor OpenVINO y ahí
> falta confirmarlo. Allí el reparto debería favorecerlo aún más —el
> decodificador pesa más—, pero eso hay que verlo. Se activa y desactiva con
> `services.voz-stream.solaparDecodificador`, así que el A/B es una línea.

```bash
VIBEVOICE_SOLAPAR_DECODER=0 python pkgs/vibevoice-cli/voz_stream.py
```

</details>

<details>
<summary><b>6 · Streaming</b> — primer sonido 23,21 s → 0,20 s</summary>

<br>

**No baja el RTF. Es lo que convierte esto en conversación.** Con RTF 0,8 sin streaming esperas 8 segundos
antes de oír nada; con streaming oyes en cientos de milisegundos aunque el RTF siga por encima de 1.

**Lo importante:** el audio es **bit a bit idéntico** al de la generación normal (mismo md5). No es una
versión degradada, es el mismo resultado entregado según se produce.

**Servicio aparte de `voz-api`, a propósito.** `voz-stream` carga VibeVoice (~2,3 GB); `voz-api` solo las
voces de Piper (~100 MB). Juntarlos haría que una síntesis pesada bloqueara las notas de voz rápidas.

```bash
curl -sN -X POST http://voz:8082/tts -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{"texto":"Se oye según se genera."}' | ffplay -autoexit -nodisp -
```

</details>

<details>
<summary><b>7 · Subir la guía CFG de 1,5 a 3,0</b> — cuatro veces menos error, gratis</summary>

<br>

Este salió de construir un **banco de fidelidad** que cierra el circuito: **texto → voz → whisper →
texto**, y compara. Genera cada frase varias veces porque la difusión parte de ruido: si una sale bien y
otra mal, el problema no es la frase sino la **estabilidad**.

| `cfg_scale` | WER medio | Peor caso | Frases inestables |
|---|---|---|---|
| 1,5 | 13,6 % | 85,7 % | 3 de 6 |
| 2,5 | 5,4 % | 42,9 % | 1 de 6 |
| **3,0** | **3,6 %** | **14,3 %** | **1 de 6** |
| 3,5 | 3,3 % | 28,6 % | 2 de 6 |

**Y es gratis en tiempo:** RTF 0,96 en los dos casos. La difusión evalúa la rama positiva y la negativa en
un lote de 2 **pase lo que pase** —ya se midió que doblar el lote cuesta un 5 % más, no el doble—, así que
subir la guía no añade una sola pasada.

Además **3.0 es el defecto del propio upstream** en `sample_speech_tokens`: se iba por debajo de lo que el
modelo espera.

```bash
python scripts/fidelidad.py          # reproduce el banco
```

> **Una trampa de medición que habría falseado todo.** whisper escribe los números **en cifras** aunque se
> digan con letra, y `%` donde se dijo «por ciento». Contarlo como error daba **WER 19,6 %** cuando el real
> es **5,2 %**. Ahora se unifican los dos lados a palabras antes de comparar. Sin eso, el informe habría
> mandado a optimizar un problema inexistente.
>
> **Lo que este banco NO mide:** si la voz suena natural. Whisper entiende perfectamente una voz horrible.

</details>

<details>
<summary><b>8 · Memoria: soltar peso muerto</b> — 3718 → 2832 MB</summary>

<br>

Dos hallazgos, ninguno por deducción: los dos están escritos en el código de Microsoft o se comprueban
cargando el modelo.

**El codificador acústico: 1311 MB de peso muerto.** No hace falta por dos razones independientes:

- **No está en el checkpoint.** `transformers` avisa de que se inicializa desde cero: son pesos
  **aleatorios**.
- **Nadie lo llama.** La única referencia al tokenizador acústico en el camino de streaming es `.decode()`.

Se recuperan 401 MB de los 1311, porque sus capas `Linear` ya se cuantizaban y lo que sobrevivía eran las
convoluciones, que `quantize_dynamic` **no toca**.

**La tabla de embeddings duplicada: 519 MB.** `tts_language_model` trae su propia `embed_tokens` de
151936×896 que no se usa jamás — lo dice el propio código de Microsoft:
*«Note that embed_tokens in tts_language_model is unused»*. Duele porque `quantize_dynamic` tampoco toca
`nn.Embedding`, así que sobrevivía entera en fp32. Se **apunta** a la del otro modelo en vez de borrarla
—misma forma y vocabulario— para que si algún camino la consultara devuelva lo correcto en vez de explotar.

**Devolver la memoria al sistema.** `gc.collect()` solo no basta: **glibc conserva en sus arenas** lo que
Python libera, así que el RSS no baja aunque los objetos hayan muerto. Hace falta `malloc_trim(0)`, que se
llama tras cuantizar, tras el calentamiento y **tras cada petición** —cada síntesis deja cientos de MB de
activaciones—. Y `MALLOC_ARENA_MAX=2` evita que glibc abra una arena por hilo.

> **Dato útil:** **generar** cuesta ~166 MB, no gigas. El pico que mata contenedores es **la carga**.
> `docker stats` engaña porque incluye la caché de disco (1 GB de los 4,8 que reportaba).

</details>

<details>
<summary><b>9 · Detalles de calidad que costaron poco y se notan</b></summary>

<br>

**Cebar el decoder con silencio — fuera el chasquido inicial.** El decoder es causal y en streaming; sus
convoluciones necesitan contexto por la izquierda y en la primera llamada no lo tienen, así que el audio
arrancaba con un salto audible antes de la primera sílaba.

| | Primera muestra | Coste |
|---|---|---|
| sin cebar | −0,017969 | — |
| **1 fotograma** | **−0,000002** | **34 ms** |
| 3 fotogramas | +0,000000 | 114 ms |

Con uno basta, y se paga una vez por síntesis.

**Segmentar por límites del lenguaje, no por número de caracteres.** El corte anterior partía palabras:

```
'El despliegue se'
'realiza mediante la configuracion de infraestructura y depend'
'encias, luego se ejecuta...'
```

El sintetizador recibía texto sin sentido y no podía entonar. Ahora se corta solo donde el lenguaje lo
permite, por orden: **fin de frase** → **fin de cláusula** → **último espacio**. Con válvula de seguridad a
320 caracteres por si un LLM suelta un párrafo sin puntuación.

**Búfer inicial en el narrador.** Medido con la máquina cargada, que es el caso que importa:

| Búfer | Resultado |
|---|---|
| 0,6 s | 1 de 2 pasadas con cortes |
| **1,5 s** | **limpio ← el defecto** |
| 2,5 s | limpio, pero 1 s de espera de más |

</details>

<details>
<summary><b>10 · La pausa de una sesión se lleva TODO su estado, no solo el RNG</b> — «misma semilla = mismo audio» también con una intrusa distinta</summary>

<br>

**La hipótesis.** Una sesión suelta el candado del modelo mientras espera texto, y la generate() que
entra en ese hueco —otra sesión, o `/tts/stream`— toca estado que es del **proceso** y no de la
llamada. El RNG ya se llevaba y traía (`SesionViva._pausar/_reanudar`), y la prueba de concurrencia
de `ws_fidelidad.py` pasaba. Pero esa prueba mete dos sesiones **iguales**: si la intrusa deja los
mismos pasos y el mismo `neg_cada`, que nadie los reponga no se nota. Leyendo el código quedaban
cuatro estados sin reponer: los pasos de difusión (`set_ddpm_inference_steps`, que
`sample_speech_tokens` lee en **cada** latente), `neg_cada` (que las sesiones además nunca fijaban:
heredaban el del último `/tts/stream`), el contador de la rampa de cfg del arranque (una variable
local que **cualquier** generate() ponía a cero) y `_REMATE`.

**El escepticismo honesto.** Podía ser un fallo de papel: quizá ninguna sesión pausa antes de
consumir su rampa de arranque, y los pasos distintos entre clientes son raros. Así que primero se
escribió la prueba y se lanzó contra el código de producción, **antes** de tocar nada.

**El resultado.** Falla, y de forma determinista. La prueba `pausa` abre la sesión A con una frase
corta («Sí, claro.», menos de una ventana de 5 tokens), así que A se para **antes de su primer
fotograma** esperando la ventana de adelanto; la intrusa B entra con `pasos + 4` y otra semilla y
habla entera; A sigue. Medido en la VM (openvino, `sp-Spk3_man`, cfg 4,5, código `898a33c7`):

| | audio | md5 |
|---|---|---|
| A a solas (6 pasos, semilla 11) | 6,93 s | `e3721afb…` |
| **A con B en su pausa** (antes) | **7,20 s** | **`d37167ad…`** — primer byte distinto en el 560, o sea desde el primer fotograma |
| A con `/tts/stream` de intrusa (10 pasos, `neg_cada` 2) (antes) | 7,20 s | `871bdfae…` |
| A con B en su pausa (después, `ec7239c1`) | 6,93 s | `e3721afb…` = a solas |
| A con `/tts/stream` de intrusa (después) | 6,93 s | `e3721afb…` = a solas |

A reanudaba con los 10 pasos de B y sin su rampa de arranque (B ya la había consumido). El arreglo
es una lista única de lo que una generate() arrastra fuera de sí misma —`foto_generacion()` /
`reponer_generacion()`, junto a `_REMATE`— que la sesión fotografía antes de soltar el candado y
repone al recuperarlo. Pasos y `neg_cada` no se fotografían: la sesión los conoce y los vuelve a
fijar. Cuesta lo mismo que antes (5 KB de estado del RNG y cuatro escalares por pausa) y el resto
de la suite sale bit a bit igual.

En ese orden de entrada (A → B → A → B) la intrusa **no** se ve afectada: nadie le cambia los pasos
entre sus pausas. La prueba lo comprueba igualmente por si el orden cambia.

```bash
# En la VM, con el token del servicio en el entorno:
python scripts/ws_fidelidad.py --url http://127.0.0.1:8082 --pruebas pausa,pausa-stream
```

</details>

---

## Lo que NO funcionó

**Esta es la sección más útil del documento.** Cada línea costó una medición real; repetirlas es tiempo
perdido.

<details>
<summary><b>Descartado por medición</b> — nueve callejones sin salida</summary>

<br>

| Idea | Por qué parecía buena | Qué pasó de verdad |
|---|---|---|
| **La iGPU Intel UHD 630** | está ahí, sin usar | **2,5× más lenta que la CPU**. `matrix cores: none`, `bf16: 0`. Y comparte el mismo bus, así que **ni siquiera suma ancho de banda**. PyTorch XPU/IPEX no soportan Gen9.5 |
| **Más hilos** | 6 núcleos, 12 hilos | **empeora**: 2 hilos 4,19 · 8 hilos 4,31 · **12 hilos 5,18 (24 % peor)**. Óptimo: 6 anclados. La forma de aprovechar los que sobran es [solapar etapas](#5--solapar-el-decodificador-acústico--las-dos-etapas-a-la-vez), no subir este número |
| **Bajar `cfg_scale` para acelerar** | CFG hace dos pasadas | **no afecta**: 1.5/1.3/1.0 → 3,92/4,02/4,20. Parchear el código para saltarse la incondicional tampoco (3,90) |
| **`torch.compile`** | fusiona operaciones | **1,00×**. Nada |
| **bf16** | mitad de bytes | Coffee Lake no tiene AVX512-BF16: sería emulado. Descartado sin medir |
| **Preasignar la caché KV** | evitar realojos | sin efecto medible |
| **Atajo para `cfg=1`** | saltarse media difusión | sin efecto (ver arriba) |
| **Decoder de OpenVINO en híbrido** | lo mejor de cada uno | rápido pero **ininteligible** |
| **int4 en la cabeza de difusión** | menos bytes aún | **sesga el fin de frase** (95 tokens frente a 84) y empata en RTF |
| **Cabeza de difusión en fp32** (hoy int8) | quitar el residuo de deriva que deja el int8 en OpenVINO | son 42 M parámetros × 6 pasadas × lote 2 **por fotograma**, y va limitada por memoria: 168 MB frente a 42 MB por pasada, ~+45 ms sobre un fotograma de 133. Descartado por cuenta, sin medir |
| **Sigmas de Karras en el solver** | mejor reparto de los pocos pasos | con el schedule **coseno** del modelo los timesteps salen degenerados: `999, 999, 998, 993, 837, 0`. `trailing` da lo mismo que `linspace`. Medido en el planificador |
| **Acelerar el WSOLA de `estirar.py`** | es un bucle en Python puro | cuesta **0,01 s por segundo de audio**. No es cuello |
| **Redondear en vez de truncar al pasar a PCM16** | 0,5 LSB de error frente a 0,25 | sin sesgo DC y a −96 dBFS: inaudible |
| **8 o 10 pasos de difusión en vez de 6** | con 6 la última evaluación de la red cae en t=166 y el solver salta a cero; con 8 es t=125 y con 10 t=100, y el coste es pequeño | **medido con umbral fijado de antemano y no lo pasa**: ver la tabla de abajo. 8 sube el UTMOS +0,046 de media (se pedía +0,10) en 22 de 36 clips (se pedían 24) y mete una alucinación (WER 211 %); 10 lo baja. Se queda en 6 |

**8 y 10 pasos, la tabla.** Banco emparejado en la VM (openvino, `sp-Spk1_man`, cfg 3,0, las 6 frases
de `fidelidad.py` × las semillas 11, 7, 3, 23, 42 y 101, mismas semillas en los tres bancos; whisper
de la VM; UTMOS22 como juez de naturalidad, comparado solo por diferencia clip a clip). El umbral se
escribió **antes** de ver un número: 8 se adoptaba si ΔUTMOS medio ≥ +0,10 con mejora en ≥ 24/36
clips, el WER medio no empeoraba más de 1 punto y RTF₈ ≤ 1,10 × RTF₆.

| pasos | WER medio | WER peor | exactos | UTMOS medio | UTMOS p10 | ΔUTMOS clip a clip | mejora en | RTF | ms/fotograma | cabeza |
|---|---|---|---|---|---|---|---|---|---|---|
| **6** | 11,8 % | 83,3 % | 23/36 | 3,547 | 3,126 | — | — | **1,057** | 125,1 | 6 × 2,60 = 15,6 ms |
| 8 | 11,2 % | **211,1 %** | 27/36 | 3,592 | 3,205 | **+0,046** | **22/36** | 1,081 | 128,8 | 8 × 2,56 = 20,5 ms |
| 10 | 10,7 % | 122,2 % | 23/36 | 3,523 | 3,116 | −0,024 | 19/36 | 1,137 | 134,2 | 10 × 2,54 = 25,4 ms |

Lo que sí deja claro la tabla es **dónde no está la calidad**: la diferencia entre 6 y 10 pasos es
de centésimas de UTMOS, mientras que entre semillas la misma frase va de 0 % a 41 % de WER (ver la
tabla de semillas en [plan-determinismo-calidad.md](plan-determinismo-calidad.md)). El coste sí es
el previsto: cada paso son 2,5-2,6 ms de cabeza por fotograma, o sea +2,3 % de RTF con 8 y +7,6 %
con 10. Y la primera pasada del banco a 6 pasos salió a RTF 1,267 por correr recién reiniciado el
servicio con las páginas del modelo aún en swap; se repitió (mismos 36 md5, bit a bit) y dio 1,057.
El RTF de la primera tanda tras un despliegue no vale.

**El corolario que ordena todo:** como el cuello es **leer pesos desde RAM**, lo que paga es **reducir
bytes de peso**, no reducir operaciones. Por eso int8 ganó y `torch.compile` no.

**Y el matiz que llegó después.** Ese corolario explica por qué fallan las ideas
de la tabla, pero **no dice que la máquina esté llena**. Lo prueba el
solapamiento: el decodificador entero cabe en el hueco que deja el bucle
esperando a la memoria. Lo que estaba saturado era *una etapa*, no el sistema.

</details>

<details>
<summary><b>⚠️ La trampa que se invierte según el motor</b></summary>

<br>

`OMP_PLACES=cores` (anclar hilos a núcleos) **acelera PyTorch un 3 % pero ralentiza OpenVINO un 118 %**:

| | Sin anclaje | Con anclaje |
|---|---|---|
| OpenVINO | **89 ms/llamada** | 195 ms/llamada |

Está documentado en el propio módulo. **Si algún día el motor cambia, ese ajuste hay que invertirlo.**

</details>

---

## Cómo medir sin engañarse

<details>
<summary><b>Las reglas del banco</b> — cada una salió de una medición falseada</summary>

<br>

- **Un modelo por proceso.** Cargar fp32 y su copia int8 a la vez pasa de 4,7 GB y el OOM mata el proceso.
- **Salida sin buffer** (`python -u`) o las filas se pierden al morir.
- **Siempre los mismos hilos**, o las comparaciones no valen:
  ```bash
  OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close
  ```
- **Guardar el `.wav` de cada variante.** El RTF sin calidad no significa nada: una variante 3× más rápida
  que suena mal no sirve.
- **`HF_HUB_OFFLINE=1`** — el modelo está en el store.
- **El prefijo de voz se muta en `generate()`**: recargar y `deepcopy` en cada medición, o la segunda
  salida no se parece a la primera.
- **Unificar formato antes de comparar texto** — ver la trampa de whisper con los números, arriba.

**Y una advertencia sobre el entorno:** una medida anterior de «~0,5 s de hueco en cada frontera de frase»
estaba **contaminada por otros procesos saturando la máquina**. En reposo no había huecos ni con búfer 0.
Mide en la máquina que importa, y en el estado que importa.

</details>

<details>
<summary><b>Medir el sistema en marcha</b> — sin instrumentar nada</summary>

<br>

La API devuelve sus propias métricas en cada respuesta:

```bash
# TTS: las cabeceras X-* traen la medición de esa síntesis
curl -s -X POST http://voz:8080/tts -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d '{"texto":"Prueba de rendimiento."}' -o /dev/null -D- | grep '^X-'
```

**Lánzala dos veces:** la primera incluye la carga del `.onnx`; la segunda es el coste real.

```bash
# STT: los campos del JSON
curl -s -X POST http://voz:8080/stt -H "Authorization: Bearer $TOKEN" -F "archivo=@muestra.ogg" | jq '{duracion_s, proceso_s, rtf}'

# Fidelidad: el circuito completo texto -> voz -> whisper -> texto
python scripts/fidelidad.py

# El A/B EMPAREJADO: mismas frases y mismas semillas en las dos variantes, para
# que la diferencia sea la del cambio y no la del sorteo. Deja un WAV por
# (frase, semilla) y clips.csv con una fila por clip.
python scripts/fidelidad.py --semillas 11,7,3,23,42,101 --pasos 8 --crono --audios banco/pasos8

# Y la naturalidad, que whisper no ve: UTMOS por clip y tabla emparejada entre bancos
python scripts/naturalidad.py puntuar --dir banco/pasos8
python scripts/naturalidad.py comparar --base banco/pasos6 --contra banco/pasos8 banco/pasos10
```

Y la **consola en el navegador** (`http://voz:8080/`) muestra el pipeline por estados y colores —gris
pendiente, azul segmentado, ámbar sintetizando, verde sonando— así que se ve dónde está el tiempo sin leer
un log.

</details>

---

## Qué queda sobre la mesa

<details>
<summary><b>La mejor inversión pendiente: 20 € de RAM</b></summary>

<br>

La máquina está en **single channel**: un módulo de 8 GB y el segundo zócalo **vacío**. Medido: **17,2 GB/s
de 21,3 teóricos (80,7 %)** — la CPU sola ya exprime el bus.

Un segundo módulo idéntico da **dual channel**: ~34 GB/s reales, **el doble del recurso que limita cada
inferencia**.

Y el techo de memoria ya bloqueó trabajo real cuatro veces: el banco de cuantización se quedó sin memoria,
el servidor de streaming murió tres veces, los grafos de OpenVINO no caben en el sandbox de Nix (piden
4,6 GB, y por eso los IR viven fuera del store), y el host de construcción va sin margen.

| Opción | Coste | Resultado |
|---|---|---|
| **+1× 8 GB DDR4-2667 SODIMM** | **~20 €** | 16 GB, dual channel |
| 2× 16 GB | ~60 € | 32 GB (el máximo), dual channel |

**El segundo módulo debe ser idéntico** o el dual channel puede no activarse.

**La GPU, en cambio, ya no hace falta:** el objetivo era tiempo real y se alcanzó en CPU. El detalle de por
qué una eGPU no compensa está en [hardware-y-portabilidad.md](hardware-y-portabilidad.md).

</details>

---

## Documentos relacionados

| Documento | Qué añade |
|---|---|
| [rendimiento.md](rendimiento.md) | las mediciones de Piper y whisper, y la comparativa entre motores |
| [hardware-y-portabilidad.md](hardware-y-portabilidad.md) | GPU por passthrough, RAM y llevar el stack al Mac |
| [arquitectura.md](arquitectura.md) | dónde vive cada una de estas piezas en el flake |
| [plan-determinismo-calidad.md](plan-determinismo-calidad.md) | el plan del 10-09-2026: estado por sesión, semilla por defecto, banco emparejado con UTMOS y 6/8/10 pasos, con los umbrales fijados antes de medir |
