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
| 8 | El bucle de difusión entero en un grafo (OpenVINO, VM) | **0,885**³ | 1,06× | 6 pasos de cabeza + guía + freno + solver en una llamada; ver [el detalle](#8--el-bucle-de-difusión-en-un-grafo--seis-llamadas-y-el-solver-en-una) |
| 7 | Subidas del decodificador sin convolución traspuesta (OpenVINO, VM) | **0,98**² | 1,1-1,3× | con k = 2s son un producto de matrices; ver [el detalle](#7--las-subidas-del-decodificador-sin-convolución-traspuesta--el-mismo-cálculo-un-tercio-del-tiempo) |

³ Mediana del banco A/B de 238 clips, frente a 0,941 con la difusión paso a paso en el mismo banco.

² Motor OpenVINO en la VM de producción, desde RTF 1,13-1,29 con el decodificador anterior. En la VM
el solapado (5) no se usa: con OpenVINO empeora.

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
<summary><b>7 · Las subidas del decodificador sin convolución traspuesta</b> — el mismo cálculo, un tercio del tiempo</summary>

<br>

**El hallazgo, con el perfilador y no a ojo.** `PERF_COUNT` de OpenVINO sobre el IR de producción
(i7-8700T, 6 hilos): el decodificador tarda 57 ms por llamada y **31 ms son las seis
convoluciones traspuestas** de las subidas, en `jit_gemm_f32`; solo `subidas.0` son 16,7 ms. El LM y
la cabeza, en cambio, ya van por `brgemm_avx2` en un 89-95 %: por ahí no queda nada.

**Por qué sobraba.** `SubidaTr` hacía la traspuesta sobre la ventana entera — las k-1 entradas del
estado más las T nuevas — y se quedaba con las T·s muestras del final: en `subidas.0` entran 16,
salen 136 y se usan 8. Con k = 2s cada muestra de salida suma exactamente **dos** entradas, la suya y
la anterior, así que basta con las últimas T+1. Y cada entrada aporta un bloque de k muestras que es
un producto de matrices: pesos `[C_in, C_out, k]` → matriz `[C_out·k, C_in]`, y el tramo j de salida
es la mitad izquierda del bloque j más la derecha del j-1.

Solo recortar la ventana **no gana** en OpenVINO (31,8 → 28,0 ms): la traspuesta sigue en `jit_gemm`.
El producto de matrices sí, porque cae en `brgemm`:

| subida | traspuesta | producto de matrices |
|---|---|---|
| `subidas.0` 2048→1024 ×8 | 23,7 ms | **4,9 ms** |
| las seis | 31,8 ms | **10,1 ms** |

**Es el mismo cálculo.** En torch con los pesos reales, 40 fotogramas encadenados: 122,7 dB de SNR
frente al original (diferencia máxima 7e-7). En OpenVINO, fp16 nuevo frente a fp16 viejo por el
camino del servicio: **119,9 dB**. Y como el int8 ya comprimía las traspuestas (sus pesos iban en
`u8`), la puerta se mide contra el fp16 y no contra el int8 viejo: los dos int8 quedan a **16,9 dB**
del fp16, el viejo y el nuevo, con las subidas en `u8` o en fp16. Ese error lo pone la cuantización
de las FFN y es el mismo antes y después.

**La trampa, que costó cinco bisecciones.** El primer IR sonaba parecido pero mal: −2,9 dB, el mismo
espectro (0,97) y desplazado. No era la matemática ni la conversión: **el estado de OpenVINO sobre este
grafo, en cuanto pasa por fichero, realimenta mal.** Medido frente a torch:

| camino | SNR |
|---|---|
| convertido y `make_stateful` en memoria | 118 dB |
| guardado con estado (fp16 o float32) y releído | −2,9 dB |
| guardado sin estado, `make_stateful` al releer (por nombre o por posición) | −2,9 dB |
| guardado sin estado, estado **explícito** como entradas y salidas | **71 dB**, lo mismo que el viejo |

La causa no está identificada — los nombres y el emparejamiento de los 34 estados salen idénticos al
releer —. El IR nuevo (`decoder_mm_*`) va **sin estado** y `motor.AcusticoOV` lleva las 34 colas en
Python: copiar 711 KB por llamada. Dos corrientes intercaladas cada 10 fotogramas salen bit a bit
iguales que a solas. Dos pistas que no eran: precalcular la matriz de pesos fuera del grafo (sale el
mismo IR, la constante se pliega) y emparejar el estado por posición.

**De extremo a extremo, en producción** (VM voz, desplegado con Nix, 8 frases × 2 rondas, semilla
101, `/crono` del propio servicio):

| | antes | después |
|---|---|---|
| RTF global | 1,288 | **0,983** |
| decodificador acústico | 76,3 ms/fotograma | **41,3 ms** |
| LM · cabeza · resto | 43,3 · 15,6 · 11,3 | 43,1 · 15,9 · 11,1 |
| duración de cada frase | — | **idéntica** en las 8 |
| SNR después frente a antes | — | 37-38 dB (el ruido del int8, el mismo que dio la puerta) |

Ese «antes» cayó en el extremo lento de la base: en pasadas alternadas en la misma VM la base iba de
1,13 a 1,27 y el decodificador nuevo de 1,02 a 1,06. **Un 10-24 % menos de RTF, y por primera vez por
debajo de tiempo real en esta VM.**

**El banco exhaustivo** (`scripts/banco_ab.py`, 13-09-2026): 7 voces — los cuatro clones y tres de
serie — × 17 frases (números, preguntas, exclamaciones, un párrafo de ~25 s) × 2 semillas, cfg 3,5,
238 parejas generadas por el servicio de verdad, whisper **large-v3** para la transcripción y las marcas
por palabra, UTMOS22, ECAPA y F0 fotograma a fotograma. Con un **control**: el mismo decodificador en
int4, un cambio de timbre conocido. Si el banco no lo separa, no vale para concluir.

| | nuevo frente a viejo | control int4 frente a viejo |
|---|---|---|
| RTF (238 clips) | 1,057 → **0,924** | → 0,906 |
| mismo largo | **238/238** | 222/238 |
| SNR, mediana | 38,9 dB | 18,7 dB |
| MCD / LSD, medianas | **1,16 / 0,41 dB** | 12,17 / 2,91 dB |
| coseno ECAPA con el clip base, mediana | **0,9999** | 0,9888 |
| identidad contra la huella de la voz, diferencia [IC 95 %] | **−0,000 [−0,000, +0,000]** | −0,000 [−0,001, +0,001] |
| desvío de F0 donde suenan las dos, mediana (p95) | **0 cents (6,7)** | 0 cents (50,4) |
| recorrido tonal, diferencia [IC] | +0,004 [−0,008, +0,016] st | **−0,040 [−0,072, −0,010] st** |
| final del habla, diferencia [IC] | −0,2 [−0,6, +0,2] ms | −6,8 [−11,6, −1,8] ms |
| desfase por palabra (marcas de whisper) | 0 ms | 0 ms |
| transcripción idéntica | 234/238 | 226/238 |
| WER | 3,80 → 3,84 % | 3,80 → 3,46 % |
| UTMOS, diferencia [IC] | −0,001 [−0,002, +0,000] | **−0,043 [−0,052, −0,035]** |

Por voz, los clones no se separan de las de serie: MCD 1,14-1,21 dB, identidad igual a la cuarta cifra
(andrés 0,7933 → 0,7927, juan 0,8105 → 0,8104) y UTMOS igual. Las cuatro transcripciones que cambian son
titubeos de whisper sobre audio que mide lo mismo («Ahora» / «Jora», «Lo siento» / «Y lo siento»).
Y el int4, que la tabla de `precisionAcustico` ya daba por cambio de timbre, **queda descartado con
números**: menos naturalidad, el recorrido tonal más plano y 16 clips que ni duran lo mismo — el recorte
de silencio y la cola se deciden sobre el audio, así que al cambiar la amplitud se mueven —, por un 2 %
de RTF.

</details>

<details>
<summary><b>8 · El bucle de difusión en un grafo</b> — seis llamadas y el solver, en una</summary>

<br>

**Donde estaba.** Por cada fotograma, `sample_speech_tokens` hacía 6 llamadas a la cabeza de difusión y,
entre cada dos, en torch: la guía (cfg), el freno de guía y un paso del solver DPM. Dentro del servicio
cada llamada costaba 2,6 ms frente a 1,9 aislada, y el solver aparecía en el muestreador de pila.

**Por qué se puede trazar.** Con los pasos fijos, lo que el solver hace en Python — índice del paso,
primer orden al principio y al final, segundo orden en medio — es aritmética sobre constantes. El bucle
entero cabe en un grafo: entradas `condition` [2,896], `speech` [2,64] (el ruido, que sigue saliendo de
`torch.randn` con la misma forma, así que la semilla consume igual), `cfg_scale` y `freno`; salida el
latente. `pkgs/vibevoice-ov/convertir_difusion.py`, int8 con la misma receta que la cabeza, y los pasos
en el nombre del fichero (`difusion_p6_int8.xml`): con otros pasos `voz_stream.py` sigue paso a paso.

| | paso a paso | un grafo |
|---|---|---|
| bucle de un fotograma, aislado | 19,8 ms | **13,9 ms** |
| `generate` por fotograma, dentro del servicio (`/crono`) | 107,7 ms | **102,9 ms** |
| `resto` (Python) por fotograma | 11,2 ms | **6,6 ms** |
| paridad en torch fp32 | — | **0,0** |
| int8 frente al camino de siempre, un fotograma | — | 67,9 dB |

**Lo que cambia de verdad, y por qué hizo falta el banco.** A diferencia del decodificador, **este latente
vuelve al LM**: un redondeo distinto hace que la locución tome otro camino igual de válido. 27 de 238
clips cambian de duración, así que las medidas fotograma a fotograma no dicen nada aquí. Se juzgó con el
banco A/B (238 parejas contra producción, whisper large-v3) y un criterio **escrito antes de ver un
número**:

| criterio | exigido | medido |
|---|---|---|
| WER, IC superior | ≤ +0,5 pts | **+0,405** (media −0,62: 3,84 → 3,21 %) |
| UTMOS, IC inferior | ≥ −0,02 | **−0,012** (media −0,002) |
| identidad, global y por clon | ±0,005 | +0,000 · andrés +0,0003 · isis −0,0023 · juan −0,0011 · santiago −0,0007 |
| tono medio / recorrido | ±0,05 st o IC con el 0 | −0,023 [−0,091, +0,044] / +0,003 [−0,063, +0,073] |
| final del habla / velocidad | ±20 ms / ±0,05 pal/s | +0,7 ms / +0,003 |
| **RTF (mediana del banco)** | — | **0,941 → 0,885** |

Lo que el criterio no cubría y se ve: **las pausas internas crecen un poco**, +0,06 por clip y +15 ms
de duración [IC +3, +29]. Y las transcripciones que cambian (13 de 238) van en los dos sentidos: donde la
difusión paso a paso sacaba «No, no, no, que es el olvido» la nueva dice «Vale, ahora mismo lo miro».

</details>

<details>
<summary><b>6 · Streaming</b> — primer sonido 23,21 s → 0,20 s</summary>

<br>

**No baja el RTF. Es lo que convierte esto en conversación.** Con RTF 0,8 sin streaming esperas 8 segundos
antes de oír nada; con streaming oyes en cientos de milisegundos aunque el RTF siga por encima de 1.

**Lo importante:** el audio es **bit a bit idéntico** al de la generación normal (mismo md5), quitando
los fotogramas callados de la entrada que ya no se emiten (ver más abajo). No es una
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
<summary><b>8b · Cargar solo lo que se usa</b> — pico de carga 4,4 → 1,95 GB, audio idéntico</summary>

<br>

**Donde estaba.** `motor.cargar()` hacía `from_pretrained(float32)` del checkpoint **entero**: 2 GB en
bf16 pasaban a ~4 GB en fp32 para soltar acto seguido el LM TTS, el decodificador, el codificador y la
cabeza, que sustituye OpenVINO. Ese pico (VmHWM 4,3-4,5 GB en una VM de 4,9) empujaba 860 MB al swap en
cada arranque, y de ahí salía la unidad `voz-stream-sin-swap`.

**Tres cambios en `pkgs/vibevoice-ov/motor.py`, bit a bit:**
- **A1:** el modelo se construye en `meta` (`accelerate.init_empty_weights`, con el config en float32
  como hace `from_pretrained`). Las piezas sustituidas se sueltan **antes** de leer nada, y del
  safetensors se leen con `safe_open` solo los tensores que siguen vivos.
- **A2:** la tabla de embeddings del LM de texto (151936 × 896) se sirve **por mmap** del safetensors en
  bf16 y se pasa a fp32 al consultarla (`EmbeddingMmap`). bf16 → fp32 es exacto, así que consultar y
  luego convertir da los mismos bytes.
- **A4:** `CabezaOV` se compila en su primera llamada. Con la difusión en un grafo no se llama nunca.

**La puerta, fijada antes de medir** ([plan-rendimiento.md](plan-rendimiento.md)):

| | base (carga vieja) | variante |
|---|---|---|
| huella de los 65 tensores torch | — | **idéntica** |
| md5 de 8 frases, semilla 101, 12 rondas en 4 procesos alternos | — | **idéntico en todo** |
| `ws_fidelidad.py` completo | — | **todo correcto** |
| VmHWM de voz-stream | 4326 / 4503 MB | **1952 / 1945 MB** |
| RSS tras el banco | 2168 / 2166 MB | **1841 / 1831 MB** |
| arranque (carga + calentamiento) | 22,4-25,5 s | **12,8-12,9 s** |
| RTF, mediana de las rondas válidas | 0,9490 | 0,9435 (en el ruido) |

**Desplegado el 14-09-2026** (`b67ee49`). En producción, voz-stream arranca con VmHWM **1867 MB** y
0 de swap, el md5 de las 8 frases es el de antes y `ws_fidelidad.py` sale «todo correcto». Con el
mismo despliegue, whisper pasa a 4 hilos con `Nice 10` y `CPUWeight 20`, después de comprobar que
transcribe exactamente lo mismo que a 6 (8/8 frases).

En el laboratorio (`scripts/lab_fase0.py carga`) el pico de la carga sola baja de 4321 a 1361 MB. De los
~1,35 GB que quedan, ~700 MB son el repack del decodificador int8 de OpenVINO (360 anónimos y 340 del
mmap del IR).

**Tres trampas del banco que costaron intentos**, antes de la primera cifra:
- **Swap desigual:** la base arrancaba con páginas en swap y la variante no, así que cada tanda
  devuelve el swap a RAM como `voz-stream-sin-swap`.
- **PATH vacío en `systemd-run`:** arranca sin PATH, y ni `pgrep` ni `systemctl` existían.
- **PATH de la unidad sin `curl`:** copiar el entorno de la unidad trae su `PATH` de NixOS, que no
  incluye `curl`.

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

<details>
<summary><b>11 · El aire de antes de la primera palabra</b> — un cuarto de cada relleno era silencio</summary>

<br>

**El síntoma.** Se generó un audio de prueba con una voz clonada para escucharlo, y antes de medirlo
nada se veía: ni recorte, ni deriva de volumen, ni chasquido, ni corte en seco. Lo que sí salió al
mirar los bordes es que **la locución tardaba en arrancar**.

**Lo medido** (11-09-2026, VM con OpenVINO, umbral de pico 0,03, el mismo con el que el respiro
distingue suelo de sala de voz):

| ruta | duración media | silencio delante |
|---|---|---|
| rellenos del asistente (`/tts/stream`) | 1,22 s | **0,33 s** (hasta 0,58) |
| frases largas (`/tts/stream`) | 2,84 s | 0,19 s |
| párrafo de 6 frases (sesión) | 20,50 s | **0,70 s** |

En un relleno de una palabra eso es **un cuarto del clip**. Y los rellenos son justo lo que el
asistente suelta para tapar la espera mientras piensa: se generan una vez, se guardan en caché y se
reproducen mil veces, así que ese tercio de segundo de silencio **se paga entero en cada
reproducción**. Ahí es donde está la ganancia.

**Donde NO está la ganancia es en el streaming en directo, y conviene decirlo porque parecía que
sí.** Se midió la misma narración de 35 s por sesión, con el recorte y sin él:

| | primer byte | primera sílaba | audio |
|---|---|---|---|
| sin recortar | 0,30 s | 0,58 s | 35,5 s |
| **con recorte** | 0,55 s | **0,56 s** | 35,2 s |

La primera sílaba suena **a la misma hora** (0,56 frente a 0,58 s, ruido). Es evidente en cuanto se
mira: el silencio de cabeza se «reproducía» mientras el modelo todavía estaba generando la primera
palabra, así que quitarlo no adelanta nada en directo. Lo que cambia es que el primer byte llega más
tarde —ya no se mandan trozos vacíos— y que **el byte que llega ya es voz**. En un fichero que se
guarda para después, en cambio, el silencio es retraso puro.

**El arreglo.** No emitir los fotogramas callados de antes del primer sonido, en las dos vías
(`RecorteEntrada`, por composición en los dos streamers, igual que `RemateEOS` hace con el final).
**Se tiran fotogramas enteros y no se corta dentro de uno**, por dos razones: el fotograma que trae
el ataque se emite completo, así que no hay forma de comerse el arranque de la palabra; y la rejilla
de 3200 muestras se conserva, que es de lo que depende la prueba con la que `ws_fidelidad.py`
comprueba que el respiro del servidor es exactamente el post-proceso documentado.

**El resultado**, A/B con el mismo binario cambiando solo el campo `recorte_entrada` de la petición:

| | antes | después |
|---|---|---|
| «Vale.» | 0,80 s | **0,40 s** |
| «Hecho.» | 0,80 s | **0,40 s** |
| «El backup de anoche terminó sin errores.» | 2,80 s | 2,53 s |
| silencio delante (media de 7 frases) | 0,35 s | **0,05 s** |
| los 25 rellenos del perfil general, seguidos | 54,0 s | **37,8 s** (−30 %) |
| párrafo de 6 frases por sesión | 20,5 s | 19,9 s |

**Y se comprueba lo fuerte, no lo cómodo:** en las 7 frases el audio recortado es **exactamente** el
de antes menos N fotogramas enteros de cabeza (`a[k:] == b`, con `k` múltiplo de 3200 muestras). Si
eso se cumple, el recorte no ha tocado el habla; no hace falta creerse nada más. La suite entera de
`ws_fidelidad.py` sigue en verde, respiro incluido.

**Lo que sí cambia es lo que whisper OYE, y por poco.** Transcribiendo los 62 rellenos del asistente
con el mismo juez, en tres versiones del mismo audio:

| versión | rellenos exactos de 62 |
|---|---|
| sin recortar | 59 |
| recorte entero (**lo que se queda**) | 58 |
| recortando pero dejando un fotograma de sala delante | 58 |

Un clip se tuerce, otro se arregla y un tercero pierde una tilde. **La tercera fila es la que cierra
el asunto**: si el problema fuera el arranque abrupto, dejar 133 ms de sala delante lo arreglaría, y
no lo hace (*«Sí, claro.»* sale *«¡Ciclar!»* con recorte y *«¡Cicler!»* con margen). Es ruido del
reconocedor sobre medio segundo de audio, no una propiedad del recorte, que ya sabemos que no toca
una muestra del habla.

> **Consecuencia para el banco:** al medir clips de una palabra conviene pedirlos con
> `"recorte_entrada": false`. El habla es idéntica y el juez tiene su pista de aterrizaje, así que
> las cifras salen comparables con las de antes del 12-09-2026.

```bash
# El A/B, sin reiniciar nada: mismo binario, un campo de la peticion
curl -s -X POST http://voz:8082/tts/stream -d '{"texto":"Vale.","recorte_entrada":false}' ...
```

</details>

---

## Lo que NO funcionó

**Esta es la sección más útil del documento.** Cada línea costó una medición real; repetirlas es tiempo
perdido.

<details>
<summary><b>Descartado por medición</b> — los callejones sin salida</summary>

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
| **Ajustes de hilos de OpenVINO y torch** (13-09-2026) | `/crono` daba 20-22 ms por pasada de LM dentro del servicio frente a 13,4 aislado, y OpenVINO 2025.4 trae `ENABLE_CPU_PINNING` activado | **nada que medir por encima del ruido, y el audio sale igual bit a bit en todos** (16/16 md5). Banco de 8 frases × 2 rondas en la VM, alternando con la base: base 1,131 / 1,154 / 1,270 · `ENABLE_CPU_PINNING=false` 1,289 · `OMP_WAIT_POLICY=PASSIVE` 1,128 · los dos 1,147 · torch a 2 hilos 1,195 · `CPU_DENORMALS_OPTIMIZATION` 1,137. Y en un microbanco del bucle de un fotograma, 107-127 ms en todas las variantes (un `ov.Core` compartido, sin torch, pasivo). Las tres bases se separan más que cualquier variante de su base |
| **El decodificador en int4** (medido con el banco exhaustivo, 13-09-2026) | 0,924 → 0,906 de RTF | UTMOS −0,043 [IC −0,052, −0,035], recorrido tonal −0,040 st, 16 de 238 clips con otro largo, 12 transcripciones distintas; ver la tabla de la subida sin convolución traspuesta. Un 2 % de RTF no lo paga |
| **Buscar capas en implementación de referencia** | fue lo que destapó las depthwise en torch (106×) | con `PERF_COUNT`, el LM y la cabeza van por `brgemm_avx2` en un 89-95 %; lo que está en `ref` suma 1,2-1,7 ms por llamada. Donde sí había que mirar era el decodificador: ver la subida sin convolución traspuesta |
| **Cuantización dinámica de activaciones y precisión de la caché KV** (`DYNAMIC_QUANTIZATION_GROUP_SIZE` 0/64/128, `KV_CACHE_PRECISION` f16/f32) | la caché KV del LM figuraba como `u8` y el LM se encarece con el contexto (13,7 ms/paso con 400 tokens, 18-21 con 1400) | **no se aplican en esta CPU**: en las 14 combinaciones, LM, cabeza y decodificador dan la salida **bit a bit igual** que la base y los tiempos quedan en el ruido. La KV no va cuantizada: esa propiedad solo actúa sobre la atención fusionada con caché, que aquí no existe (fila siguiente) |
| **Fusionar la atención del LM con la caché KV** (`ScaledDotProductAttentionWithKVCache`) | copiaría la caché en su sitio en vez de concatenarla en cada paso y usaría un kernel de atención | **no hay nodo de atención en AVX2**: el plugin de CPU descompone incluso un modelo que es SOLO `scaled_dot_product_attention`, estático o dinámico, en `Subgraph` de MatMul+Softmax. Ni `enable_gqa`, ni GQA a mano, ni quitar la máscara cambian el grafo compilado (probado con un LM de juguete y paridad 1e-8). Es cosa del hardware (sin AVX-512 ni AMX) |
| **Las dos pasadas del LM (condicional y negativa del CFG) en paralelo** (14-09-2026, plan de rendimiento B2) | la negativa no depende de la condicional, y una pasada a 3 hilos cuesta solo 1,14× la de 6 | **no ganan**: medianas de 6 rondas en la VM, con voz-stream parado. El par a la vez (2 streams de OpenVINO, 3 hilos cada uno) cuesta 0,925× dos pasadas seguidas con 500 posiciones y 1,022× con 1500; con dos modelos compilados, peor. La puerta pedía ≤ 0,80 en las dos longitudes. Dos etapas de cómputo no caben juntas en 6 núcleos, la misma pared que el solapado del decodificador. Detalle en [plan-rendimiento.md](plan-rendimiento.md) |
| **Enseñar a la VM su topología SMT real** (14-09-2026, B1: `args: -smp 12,sockets=1,cores=6,threads=2`) | el guest veía 12 núcleos físicos y podía poner dos hilos de OpenVINO en hermanos SMT; explicaría la varianza de base | **no gana**: 4 tandas alternas con la VM arrancada desde cero en cada una. El guest vio de verdad 2 hilos por núcleo, md5 idéntico y RTF mediano 0,8976 → 0,9028 (+0,6 %, la puerta pedía −3 %), sin menos varianza. Se deja la topología de antes |
| **Buscar el coste en Python** | `resto` son 11-13 ms por fotograma en `/crono` | con un muestreador de pila (cada 2 ms, 24.929 muestras en el banco de 8 frases): **el 88 % de `generate()` está dentro de `infer()`** de OpenVINO — LM 38,9 %, decodificador 35,2 %, cabeza 13,9 % —, y el 12 % de Python está repartido (conector en torch 1,7 %, solver DPM ~2 %, `sample_speech_tokens` ~2 %). No hay un punto caliente que rehacer |

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

**El corolario que ordenaba todo, y que ya no vale tal cual:** cuando la RAM iba en **canal único**, el
cuello era **leer pesos**, así que lo que pagaba era **reducir bytes**, no reducir operaciones. Eso
explica por qué int8 ganó y `torch.compile` no, y por qué fallan las ideas de la tabla de arriba. Con
el segundo módulo puesto **manda el cómputo** — las tres pruebas están en [Qué queda sobre la
mesa](#qué-queda-sobre-la-mesa) —, así que una idea nueva no se puede descartar citando este corolario:
hay que medirla.

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
- **Al medir clips de una palabra, pedirlos sin el recorte de entrada** (`"recorte_entrada": false`).
  El habla es la misma con recorte y sin él —está verificado byte a byte—, pero whisper acierta menos
  sobre medio segundo de audio que empieza de golpe, y esa diferencia es del juez, no del audio.
- **Y unificar también las grafías que no se pueden oír.** En castellano la **h es muda** y **b y v
  son el mismo fonema**: «hecho»/«echo» y «borrada»/«vorrada» suenan igual, así que cuál de las dos
  escribe whisper lo decide su modelo de lenguaje y no la voz. Con una frase larga acierta por
  contexto; con un relleno de media palabra, no. Medido en el banco de rellenos del asistente (450
  clips de 1,1 s): «Hecho.» sale como *«¡Echo!»* con 5 de 18 semillas y «Borrada.» como
  *«¡Vorrada!»* con 6 de 18 — 11 clips contados como error sin serlo, y una semilla que parecía
  24/25 era en realidad **25/25**. Lo arregla `comparable()` en `scripts/fidelidad.py`, que protege
  la «ch» antes de quitar las haches porque «echo» y «eco» sí suenan distinto. **En frases largas no
  cambia ni un número** (comprobado sobre los tres bancos de pasos y el de 18 semillas).

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
<summary><b>La semilla por defecto: 40 puntos de WER entre la mejor y la peor</b></summary>

<br>

Medido el 10 de septiembre de 2026 en la VM con 18 semillas × las 6 frases del banco
(`sp-Spk1_man`, cfg 3,0, 6 pasos; tabla completa en
[plan-determinismo-calidad.md](plan-determinismo-calidad.md)): las semillas 101 y 17 aciertan las
6 frases; la 42 falla 5 de 6 (40,7 % de WER) y la 37 tiene 29,6 %. Entre 6 y 10 pasos, en cambio,
la diferencia es de un punto. Sorteando, una de cada seis peticiones caía en una semilla mala.
**Decidido el mismo día: la VM `voz` lleva `services.voz-stream.semilla = 101`** y el compose
`VIBEVOICE_SEMILLA=101`. El mismo texto con la misma voz da ahora el mismo audio byte a byte, y un
cliente que quiera variedad manda `"semilla": null`. El defecto del módulo sigue en `null` porque la
semilla buena es de la voz: quien cambie `vozDefecto` tiene que repetir el banco
(`fidelidad.py --semillas … ` y `naturalidad.py`) antes de fijar otra. Lo que queda sobre la mesa es
ese banco con las frases reales del asistente y con las voces clonadas.

</details>

<details>
<summary><b>✅ Los 20 € de RAM ya están puestos — y con ellos cambió el corolario</b></summary>

<br>

Era «la mejor inversión pendiente» de este documento durante meses. **Ya está hecha.** El host tiene
hoy dos módulos de 8 GB, uno por canal, confirmado con `dmidecode` en el M920q:

```
Locator: ChannelA-DIMM0   Size: 8 GB   Speed: 2667 MT/s   Configured: 2400 MT/s
Locator: ChannelB-DIMM0   Size: 8 GB   Speed: 2400 MT/s   Configured: 2400 MT/s
```

Un detalle que contradice lo que decía este mismo apartado: **no hicieron falta módulos idénticos**.
Tienen frecuencia nominal distinta (2667 y 2400) y el dual channel se activó igual; los dos corren a
2400, que es lo que manda el más lento.

**Lo que cambió no es la velocidad, es el diagnóstico.** El corolario que ordenaba todo el documento
—«el cuello es leer pesos desde RAM, así que lo que paga es reducir bytes»— era cierto **en canal
único**. Con los dos módulos, las tres pruebas que lo confirmarían salen que no (medidas en la VM y
documentadas en [`pkgs/vibevoice-ov/motor.py`](../pkgs/vibevoice-ov/motor.py)):

| Prueba | Si mandara la memoria | Lo medido |
|---|---|---|
| Pasada de backbone de 2 tokens frente a 1 | ~1,0× (mismos 156 MB de pesos) | **1,77×** |
| Decodificador con 6 latentes por llamada frente a 1 | gana (344 MB leídos una vez, no seis) | **no gana** |
| Decodificador de int8 (344 MB) a int4 (212 MB) | −38 % por los bytes | **−7 %** |

**Manda el cómputo**, en seis núcleos a 2,4 GHz con AVX2 y sin VNNI. Lo que queda por ganar está en
hacer menos trabajo, no en mover menos bytes.

**Y desaparecieron los cuatro bloqueos** que la falta de memoria causaba: el banco de cuantización se
quedaba sin memoria, el servidor de streaming murió tres veces, el host de construcción iba sin margen
y los grafos de OpenVINO no cabían. Queda **una** consecuencia por revisar: los IR siguen generándose
fuera del store por aquel límite de 2560 MB del contenedor constructor, y con 16 GB puede que ya
quepan. Sería el único artefacto derivado del proyecto que dejaría de estar fuera de Nix.

Lo que **no** se ha vuelto a medir es el ancho de banda en sí: los 17,2 GB/s de 21,3 que aparecen más
arriba son la cifra de canal único, y sigue sin repetirse la prueba con los dos módulos. No cambia
ninguna decisión —el diagnóstico ya lo dan las tres pruebas de arriba— pero la cifra que se cita en
este documento es la vieja.

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
