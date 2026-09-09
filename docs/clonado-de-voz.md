# Clonado de voz

Resumen para quien tenga prisa:

- **Fabricar prefijos de voz `.pt` para el 0.5B: RESUELTO.** El formato estaba
  sin documentar y aquí queda descrito y verificado. Un prefijo fabricado a
  mano reproduce el oficial con coseno ≥ 0,9993 en las cuatro ramas, y
  sintetiza con la misma calidad (medida abajo).
- **Convertir un audio nuevo en esos latentes: RESUELTO** (2026-08-24), con un
  codificador de terceros auditado aquí. Ver [§7](#7-el-codificador-que-faltaba).
  Clonando las seis voces españolas desde 15,5 s de referencia, la huella ECAPA
  del clon contra su propia voz da **0,850 de media** (mínimo 0,803) frente a
  **0,176** contra otra voz — sin solape, y con WER 0,000.
- Lo que **no** funciona sigue documentado abajo: el codificador del 1.5B no
  vale tal cual (§4) y el adaptador lineal tiene su techo en R² ≈ 0,59 (§4.3).

Es decir: las dos mitades del problema están hechas. El formato del prefijo se
documentó aquí, y el codificador que Microsoft no publicó existe fuera y
funciona — pero **solo con la receta de prefijo de §3**, no con la que trae.

---

## 1. Por qué el 0.5B no puede clonar

`model.safetensors` del `VibeVoice-Realtime-0.5B` tiene 604 tensores. Del
tokenizador acústico hay **276 claves, todas del decoder y ninguna del
encoder**. Lo que `transformers` instancia como encoder son pesos aleatorios:
al cargar el modelo, la propia librería lo avisa listando las 276 claves
`model.acoustic_tokenizer.encoder.*` como *newly initialized*.

Desglose de pesos (MB):

| componente | Realtime-0.5B | VibeVoice-1.5B |
|---|---:|---:|
| language_model | 391,6 | 3087,4 |
| tts_language_model | 596,5 | — |
| acoustic_tokenizer / decoder | 687,4 | 687,4 |
| acoustic_tokenizer / **encoder** | **ausente** | **687,4** |
| semantic_tokenizer | — | 689,2 |
| prediction_head | 84,2 | 246,6 |
| conectores y varios | 3,3 | 10,0 |
| **total** | **1763,0** | **5408,0** |

Sin encoder no hay forma de convertir audio en latentes, y Microsoft no
publica herramienta para generarlos: `VibeVoiceStreamingProcessor` solo
**consume** prefijos (`process_input_with_cached_prompt` recibe
`cached_prompt` y de él únicamente lee las longitudes para fabricar
`input_ids` de relleno).

Nota sobre los modelos grandes: en Hugging Face hoy solo están
`VibeVoice-1.5B`, `VibeVoice-ASR` (7B), `VibeVoice-ASR-BitNet`,
`VibeVoice-ASR-HF`, `VibeVoice-Realtime-0.5B` y `VibeVoice-AcousticTokenizer`.
Los identificadores `microsoft/VibeVoice-Large`, `-Large-7B` y `-7B` devuelven
HTTP 401: ese modelo ya no está publicado. El 1.5B ocupa 5,4 GB, pero **del
repo solo hacen falta 687 MB** (el encoder acústico), que se pueden bajar por
rangos HTTP sin traer el resto.

---

## 2. Arquitectura: realtime frente a no-realtime

**VibeVoice-1.5B / ASR (no realtime).** Un solo `language_model` (Qwen2 de 28
capas, hidden 1536 en el 1.5B; 3584 en el ASR), dos tokenizadores
(`acoustic_tokenizer` con encoder+decoder y `semantic_tokenizer` solo
encoder), sus dos conectores, y la cabeza de difusión. El prompt de voz se
construye en tiempo de inferencia: `VibeVoiceProcessor._create_voice_prompt`
intercala en el texto un bloque por hablante

```
 Speaker N: <|vision_start|> <|vision_pad|> × ceil(muestras/3200) <|vision_end|>\n
```

y marca esas posiciones en `speech_input_mask`; el audio crudo viaja aparte en
`speech_tensors`. Ya dentro del modelo,
`VibeVoiceForConditionalGeneration.forward_speech_features` hace el trabajo
real:

```python
frames = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))[0][0]
audio_tokens = frames.sample(self.model.acoustic_tokenizer.std_dist_type)[0]
audio_features = (audio_tokens + self.model.speech_bias_factor) * self.model.speech_scaling_factor
connect_features = self.model.acoustic_connector(audio_features)
```

y sustituye los embeddings de los `<|vision_pad|>` por `connect_features`. Ahí
está el encoder acústico: es la pieza que el 0.5B no tiene.

**VibeVoice-Realtime-0.5B.** Parte el Qwen2 de 24 capas en dos módulos: un
`language_model` de 4 capas (con `norm` puesta a `Identity`) que solo procesa
texto, y un `tts_language_model` de 20 capas que recibe la salida del primero y
genera habla. Cada posición del `tts_language_model` lleva sumado un embedding
de tipo, `tts_input_types` (2 × 896): tipo 1 para texto, tipo 0 para habla. En
generación se alternan ventanas de texto y ventanas de latentes de difusión, y
cada latente se decodifica a audio al vuelo con caché acústica. Como el prompt
de voz no se puede recalcular en cada frase sin arruinar la latencia, viene
**precalculado en disco**: los `.pt` son cachés KV ya rellenadas.

---

## 3. Formato del prefijo `.pt` (documentado por ingeniería inversa)

Los 61 ficheros de `~/.cache/vibevoice-nix/voces` son un `dict` con cuatro
ramas, cada una un `BaseModelOutputWithPast` (bfloat16):

| rama | posiciones | capas KV | contenido |
|---|---|---|---|
| `lm` | M | 4 | prefill del `language_model` con los tokens de la transcripción |
| `tts_lm` | N + M | 20 | prefill del `tts_language_model` |
| `neg_lm` | 1 | 4 | prefill con `<|image_pad|>` (id 151655) |
| `neg_tts_lm` | 1 | 20 | rama negativa para la guía CFG |

Con N = número de latentes acústicos (3200 muestras = 1/7,5 s cada uno) y M =
número de tokens de texto.

**La receta exacta**, verificada componente a componente:

```python
# 1. rama de texto: prefill directo, SIN plantilla ni tokens especiales
lm_out = language_model(inputs_embeds=embed_tokens(ids_transcripcion))

# 2. rama TTS: primero los latentes, DESPUES el texto, y TODO con tipo 0
entrada = cat([acoustic_connector(z) + tts_input_types[0],      # N posiciones
               lm_out.last_hidden_state + tts_input_types[0]])  # M posiciones
tts_out = tts_language_model(inputs_embeds=entrada)

# 3. rama negativa de texto
neg_lm_out = language_model(inputs_embeds=embed_tokens([[151655]]))

# 4. rama negativa TTS: la entrada es la SALIDA del lm negativo, con tipo 1
neg_tts_out = tts_language_model(
    inputs_embeds=neg_lm_out.last_hidden_state + tts_input_types[1])
```

Tres detalles que no son evidentes y que costaron encontrar:

1. En `tts_lm` los latentes van **antes** que el texto, no después.
2. El texto dentro de `tts_lm` lleva **tipo 0**, no tipo 1, aunque sea texto
   (durante la generación el texto nuevo sí entra con tipo 1). Probado: con
   tipo 1 el residuo es 1,19; con tipo 0 es 0,0025.
3. La rama `neg_tts_lm` es la única que usa tipo 1, y su entrada no es el
   embedding de `<|image_pad|>` sino la salida del `language_model` sobre él
   (residuo 0,0048 frente a 4,54 con el embedding crudo).

### Cómo se verificó

La caché KV es invertible. En la capa 0, `v = W_v·RMSNorm(h) + b_v` sin RoPE, y
`k` lleva RoPE, que se deshace con la rotación inversa según la posición. Con
esas 256 ecuaciones por posición y 64 incógnitas se recupera la entrada:

- **Rama `lm`**: se proyecta todo el vocabulario (151 936 embeddings) y se
  busca el vecino más próximo. Los 130 tokens de `sp-Spk1_man` salen con
  residuo relativo máximo **0,0044**, y al decodificarlos aparece la
  transcripción literal del audio de referencia: *"¿Cómo están? Bienvenidos a
  un episodio más del podcast de..."*. Ningún token especial: el prompt es
  texto plano.
- **Rama `tts_lm`**: las últimas M posiciones encajan con
  `lm.last_hidden_state + tipo0` (130/130 con residuo medio 0,0025); las N
  primeras se resuelven por descenso de gradiente sobre `z`, llegando a residuo
  medio **0,0047** en V y **0,0023** en K.
- Decodificando esa `z` con el decoder del 0.5B sale el audio de referencia
  original: transcrito por whisper da **WER 0,083** contra el texto que se
  había recuperado de los tokens por un camino completamente independiente.

Y la vuelta completa: refabricando el prefijo a partir de (tokens, z)
recuperados, contra el oficial:

| rama | coseno de `last_hidden_state` | peor coseno de KV |
|---|---:|---:|
| `lm` | 1,00019 | 0,99988 |
| `tts_lm` | 0,99982 | 0,99928 |
| `neg_lm` | 1,00000 | 0,99986 |
| `neg_tts_lm` | 0,99998 | 0,99972 |

---

## 4. ¿Sirve el encoder del modelo grande? Compatibilidad de espacios

### 4.1 Las configuraciones coinciden; los pesos no

`acoustic_tokenizer_config` es idéntica en 0.5B, 1.5B y ASR: ratios
8·5·5·4·2·2 (3200 muestras por latente), `vae_dim` 64, `fix_std` 0.5,
`encoder_depths` 3-3-3-3-3-3-8, RMSNorm, causal. Misma arquitectura, mismas
formas de tensor.

Los **pesos** son otra historia. Comparando 12 tensores del decoder repartidos
por toda la red:

| comparación | tensores idénticos byte a byte |
|---|---|
| 0.5B vs VibeVoice-1.5B | **0 / 12** |
| 0.5B vs VibeVoice-ASR | **0 / 12** |
| 0.5B vs VibeVoice-AcousticTokenizer | **0 / 12** |
| 1.5B vs ASR vs AcousticTokenizer entre sí | **idénticos** (md5 iguales) |

Las diferencias no son de precisión: coseno entre 0,0073 y 0,80 según la capa,
con diferencia relativa de 0,78 a 2,21 (el bias de la cabeza sale con coseno
−1,0000, es decir, signo contrario). **Los tres modelos grandes comparten un
tokenizador acústico; el realtime tiene el suyo, entrenado por separado.** Los
factores de escala tampoco coinciden: 0.5B usa escala 0,2334 y sesgo −0,0703;
el 1.5B usa 0,1963 y −0,0493.

### 4.2 La prueba con verdad de terreno

Como los latentes `z` de una voz oficial ya se habían recuperado (sección 3),
hay verdad de terreno exacta. El ciclo es: `z` → decoder(0.5B) → audio →
encoder(1.5B) → `z'`. Si el espacio fuese el mismo, `z' ≈ z`.

| prueba | resultado |
|---|---|
| control positivo: enc(1.5B) → dec(1.5B) | corr. de onda **+0,9799** |
| control negativo: encoder aleatorio del 0.5B | coseno por posición +0,0110 |
| **enc(1.5B) contra la `z` verdadera del 0.5B** | coseno por posición **+0,0158** |
| enc(1.5B) → dec(0.5B) | corr. de onda +0,0063; RMS 0,0083 frente a 0,0591 |

El control positivo (0,98) demuestra que el encoder del 1.5B está bien cargado
y funciona. Y aun así, sus latentes puestos en el decoder del 0.5B dan
prácticamente silencio, y se parecen a los verdaderos tanto como un encoder de
pesos aleatorios. **La hipótesis de partida —que el espacio latente es
compartido y basta con enchufar el encoder grande— queda desmentida.**

### 4.3 Pero los espacios están relacionados por una transformación lineal

La magnitud de los latentes sí es compatible (std 5,74 frente a 4,58), lo que
sugería una base distinta más que un espacio distinto. Ajustando una matriz de
64×64 más término independiente sobre pares (`z` del 1.5B, `z` del 0.5B), con
validación cruzada dejando una voz fuera:

| voz excluida | entreno cos / R² | prueba cos / R² |
|---|---|---|
| sp-Spk1_man | +0,808 / 0,658 | **+0,758 / 0,539** |
| sp-Spk0_woman | +0,810 / 0,661 | **+0,755 / 0,527** |
| sp-Spk3_man | +0,828 / 0,687 | **+0,692 / 0,445** |
| sp-Spk4_woman | +0,812 / 0,666 | **+0,736 / 0,525** |

Generaliza a voces que no ha visto: los dos espacios son el mismo, en bases
distintas, y una transformación lineal recupera algo más de la mitad de la
varianza. Con 10 voces de entrenamiento (2031 muestras) el resultado no mejora
—cos +0,767 y +0,748 en las dos voces de prueba—, así que **el techo del
modelo lineal está en R² ≈ 0,59, y no es cuestión de más datos**.

Reconstruyendo audio con ese adaptador (siempre con la voz excluida del
ajuste):

| voz | WER frente a la referencia | tono referencia → adaptado |
|---|---:|---|
| sp-Spk1_man | 0,126 | 115,9 → 130,1 Hz |
| sp-Spk0_woman | 0,171 | 240,0 → 244,9 Hz |
| sp-Spk3_man | 0,186 | 99,6 → 106,7 Hz |
| sp-Spk4_woman | 0,045 | 208,7 → 224,3 Hz |

El habla es inteligible y el tono se conserva razonablemente. La cadena
completa, sin embargo, no aguanta.

---

## 5. Prueba de concepto de extremo a extremo

Cadena de clonado: wav → whisper (transcripción) → encoder(1.5B) → adaptador
lineal → prefijo `.pt` → `generate()`. Voces de prueba **excluidas** del ajuste
del adaptador. Tres frases, mismas para todos, 10 pasos de difusión, cfg 1,5;
tono y recorrido con `tono()` de `scripts/sondeo_voz.py`, WER con
`scripts/fidelidad.py`.

El **oráculo** es el control clave: prefijo fabricado con los latentes
verdaderos (recuperados del `.pt` oficial) y la transcripción de whisper. Mide
la cadena entera *menos* el adaptador.

### sp-Spk5_man — referencia 135,6 Hz, 8,5 semitonos

| prefijo | tono | recorrido | WER |
|---|---:|---:|---:|
| oficial de Microsoft | 127,9 Hz | 11,5 st | 0,139 |
| **oráculo** (z real + texto de whisper) | 138,1 Hz | 10,9 st | 0,139 |
| **clon** (encoder 1.5B + adaptador) | 135,1 Hz | 19,4 st | 0,139 |

### sp-Spk2_woman — referencia 179,1 Hz, 7,6 semitonos

| prefijo | tono | recorrido | WER |
|---|---:|---:|---:|
| oficial de Microsoft | 181,8 Hz | 9,0 st | 0,198 |
| **oráculo** (z real + texto de whisper) | 189,0 Hz | 6,7 st | 0,394 |
| **clon** (encoder 1.5B + adaptador) | **152,1 Hz** | 15,7 st | 0,389 |

Lectura de la tabla:

- **El oráculo valida la receta de fabricación.** Reproduce el tono del
  oficial con 10 Hz de margen en ambas voces (138,1 frente a 127,9; 189,0
  frente a 181,8) y un recorrido tonal igual o mejor. Un prefijo fabricado a
  mano es tan bueno como uno de Microsoft, siempre que los latentes sean
  correctos.
- **El adaptador es el que falla.** En la voz masculina acierta el tono
  (135,1) porque esa voz cae cerca de la media del conjunto; en la femenina se
  queda en 152,1 Hz cuando debería estar en 182, es decir **cinco semitonos
  por debajo**: arrastra la voz hacia el promedio en vez de conservar su
  identidad.
- **La prosodia se desestabiliza siempre**: el recorrido tonal pasa de ~9-11
  semitonos a 15,7-19,4. La entonación se vuelve errática, que es lo que cabe
  esperar de un latente con un 40% de varianza sin explicar.
- La inteligibilidad no distingue clon de oráculo (0,139 y 0,389 idénticos):
  el WER alto de Spk2 viene de otra parte de la cadena, no del adaptador.

**Veredicto: no es una clonación utilizable.** Para una voz cercana a la media
pasa por buena en tono, pero no reproduce identidades alejadas y degrada la
entonación en todos los casos.

---

## 6. Qué se puede hacer hoy, y qué haría falta

### Se puede hoy

- **Fabricar prefijos con latentes conocidos** (sección 3). Utilidad
  inmediata: reetiquetar, recortar o concatenar voces existentes, cambiar la
  transcripción asociada, o construir prefijos más cortos —los oficiales
  gastan 22-33 s de audio (166-287 latentes), y acortarlos reduce el KV que se
  arrastra en cada frase.
- **Recuperar los latentes de cualquier `.pt` oficial** y su transcripción
  exacta, con los residuos de la sección 3.

### Ya se puede (desde 2026-08-24)

- **Clonar una voz desde un audio arbitrario**, con `scripts/clonar_voz.py`.
  Ver [§7](#7-el-codificador-que-faltaba) para las cifras y las trampas.

### La vía que se planteó antes de que apareciera el codificador

El obstáculo es exclusivamente el codificador. Y hay una propiedad que lo hace
tratable: **el decoder del 0.5B es diferenciable y está disponible**, así que
los latentes correctos de cualquier audio se pueden obtener por optimización
—descenso de gradiente sobre `z` minimizando la distancia entre
`decoder(z)` y el audio objetivo (mejor sobre mel-espectrograma que sobre la
forma de onda, para no penalizar desfases). Es lento, pero da pares
(audio, `z` correcta) **ilimitados** a partir de audio real cualquiera.

Con ese conjunto hay dos destinos posibles, en orden de coste:

1. **Adaptador no lineal** sobre el encoder del 1.5B (un MLP con contexto
   temporal, no un lineal por fotograma). El lineal ya llega a R² 0,59 sin
   ver contexto; el margen que queda es justo el que un modelo con ventana
   temporal debería capturar.
2. **Destilar un encoder propio** para el 0.5B, con la arquitectura que ya
   define su `acoustic_tokenizer_config` (276 tensores, 687 MB), inicializado
   desde el del 1.5B. Es lo mismo que hizo Microsoft y no publicó.

Ambos caminos son entrenamiento, no ingeniería inversa: ya no hay nada más que
descubrir sobre el formato. Antes de meterse ahí conviene medir cuánto cuesta
la optimización directa de `z` por audio, porque si sale barata (segundos por
segundo de audio) **puede ser ella misma el codificador**, sin entrenar nada:
se paga una vez por voz, al crear el prefijo, y no en cada síntesis.

---

## 7. El codificador que faltaba

Un tercero publicó el checkpoint oficial **más un codificador acústico**:
[`mohammed-bahumaish/vibevoice-realtime-0.5b-with-encoder`](https://huggingface.co/mohammed-bahumaish/vibevoice-realtime-0.5b-with-encoder)
(MIT). Aquí está auditado con `scripts/auditar_encoder.py`, y sirve.

### 7.1 No es el codificador del 1.5B con otro nombre

Los 276 tensores del codificador, comparados uno a uno contra los del 1.5B:

| | |
|---|---:|
| parámetros | 343,7 M |
| idénticos byte a byte | **0 / 276** |
| diferencia relativa (mediana) | 0,285 |
| diferencia relativa (ponderada por parámetros) | **0,472** |
| parámetros que se movieron menos del 30 % | **0,4 %** |

Es un reentrenamiento partiendo del 1.5B —justo lo que §6 proponía como vía 2—,
no una copia. Va en **F32** mientras el resto del checkpoint es BF16, y ocupa
los primeros 1311 MB del safetensors sin ningún otro tensor intercalado: se baja
entero con **una sola petición `Range`**, sin traer los 3,18 GB.

### 7.2 Vive en el espacio latente del 0.5B

Ciclo `audio → z → audio` con el decoder del 0.5B, sobre las seis voces de
`ejemplos-voces/`. El brazo positivo reproduce el 0,9799 de §4.2, que es lo que
da derecho a creerse los otros dos:

| brazo | corr. de espectro | identidad ECAPA | corr. de onda |
|---|---:|---:|---:|
| **comunitario → dec(0.5B)** | **0,961–0,971** | **0,9696** | −0,94 * |
| positivo `enc(1.5B) → dec(1.5B)` | 0,955–0,965 | 0,9788 | 0,9821 |
| negativo `enc(1.5B) → dec(0.5B)` | 0,089–0,187 | −0,0060 | 0,0026 |

\* **Trampa de medición.** La salida del comunitario viene con la **polaridad
invertida**, que es inaudible, así que a desfase cero la correlación de onda sale
en −0,94 y parece un fracaso. RMS y tono coinciden al 1–3 %. Hay que medir en
espectro logarítmico, o negando la señal.

### 7.3 Clonado de extremo a extremo: las seis voces

Con la voz oficial se genera una referencia desde un texto **conocido**, se
fabrica un prefijo de ese par y se dice la misma frase con los dos. El lazo se
cierra sobre sí mismo, así que no hace falta transcribir nada y la
transcripción es exacta por construcción.

| voz | tono oficial → clon | recorrido | ECAPA |
|---|---|---|---:|
| sp-Spk0_woman | 244,9 → 235,3 Hz | 11,8 → 10,0 st | **0,8415** |
| sp-Spk1_man | 111,1 → 128,3 Hz | 14,1 → 14,6 st | **0,8027** |
| sp-Spk2_woman | 184,6 → 175,2 Hz | 8,7 → 9,3 st | **0,8591** |
| sp-Spk3_man | 100,0 → 118,2 Hz | 11,5 → 11,2 st | **0,8413** |
| sp-Spk4_woman | 184,6 → 186,0 Hz | 11,5 → 13,3 st | **0,8774** |
| sp-Spk5_man | 134,8 → 142,0 Hz | 11,0 → 9,7 st | **0,8792** |

Contra los umbrales de `scripts/oido.py`, medidos sobre estas mismas voces
(mismo locutor ≥ 0,626, distinto ≤ 0,446):

    clon contra SU voz oficial     media 0,850   mínimo 0,803
    clon contra OTRA voz           media 0,176   máximo 0,404

No hay solape. WER 0,000 en la frase de prueba.

**Lo que no es perfecto:** las tres voces masculinas suben de tono (+2,5 a +2,9
semitonos en las dos más graves) y las femeninas se mueven mucho menos. El
volumen del clon sale entre 1,05× y 1,55× el del original, sin recorte.

### 7.4 Los dos marcadores que cuestan la identidad

El `make_voice_prompt.py` que acompaña al codificador construye la rama
`tts_lm` con un `<|vision_start|>` delante y un `<|vision_end|>` detrás del
bloque de latentes: **N+2+M** posiciones. La receta de §3 de este documento,
recuperada invirtiendo la caché KV de las voces oficiales, es **N+M sin
marcadores**.

Medido sobre el mismo audio, la misma frase y la misma semilla:

| receta | WER | tono | recorrido | **ECAPA** |
|---|---:|---:|---:|---:|
| §3 de este doc (N+M) | 0,000 | 240,0 Hz | 10,0 st | **0,8638** |
| comunitaria (N+2+M) | 0,000 | 233,0 Hz | 8,0 st | **0,4098** |
| *oráculo, la voz oficial* | *0,000* | *244,9 Hz* | *11,8 st* | *1,0000* |

**Dos posiciones de más y deja de ser la misma persona**, sin que el WER ni el
tono lo delaten. Es exactamente el tipo de fallo que no se ve sin una métrica de
identidad: suena bien, se entiende, y es otro locutor.

### 7.5 Con una grabación real de móvil

Todo lo anterior usa audio generado por el propio modelo, que es el caso
favorable. La prueba de verdad es una nota de voz de WhatsApp: **9,9 s de opus a
17 kbps**, de los que solo **6,8 s son voz** (31,7 % de silencio), con el pico
en 1,000 —o sea **recortada**— y el 99,9 % de la energía por debajo de 5,9 kHz.
Hablante masculino grave, 98,6 Hz.

Aquí no hay oráculo, así que la calibración sale de la propia grabación:
partirla por la mitad y medir una mitad contra la otra da el techo realista en
esas condiciones.

| | ECAPA |
|---|---:|
| **calibración**: sus dos mitades entre sí (mismo locutor, mismas condiciones) | 0,7170 |
| **control**: su voz contra las seis voces oficiales (distinto locutor) | máx. 0,4138 |
| clon, frase de 5,5 s | 0,6464 |
| clon, frase de 6,5 s | 0,7399 |
| clon, frase de 10,0 s | **0,8718** |
| **clon, media** | **0,7527** |

La media del clon **supera la auto-similitud de la grabación consigo misma**.
El tono del clon va de 97,2 a 111,1 Hz frente a los 98,6 Hz del original, y el
recorrido (15,7–18,8 st frente a 16,5) queda en el mismo orden.

Dos matices honestos: la cifra mejora con la duración de la frase generada, y
buena parte de eso es que **ECAPA es más ruidoso con 5 s que con 10**; y el
sesgo de subir el tono en voces graves vuelve a aparecer en las frases cortas.

Conclusión: funciona con audio de móvil comprimido, recortado y con menos de la
mitad del material recomendado.

### 7.6 Usarlo

```bash
python scripts/clonar_voz.py \
    --audio mi_voz.wav \
    --transcripcion "la transcripcion literal de ese audio" \
    --salida ~/.cache/vibevoice-nix/voces/mi_voz.pt

VIBEVOICE_VOZ=mi_voz ./scripts/voz-stream-mac.sh
```

El `.pt` sale de 4,7 MB y se carga como cualquier voz oficial. **El motor de
producción no necesita el codificador**: solo hace falta para fabricar el
prefijo, una vez por voz.

### 7.7 Lo que queda sin comprobar

- Solo español, y un puñado de frases.
- La curva de §7.8 está medida sobre voces oficiales sintéticas. Con grabaciones
  reales de micrófono la forma debería ser la misma, pero no está comprobado.
- Una sola voz real, y en condiciones malas a propósito. Falta una grabación
  limpia de 24 kHz de verdad, que debería ir mejor.
- Nadie ha comparado la `z` del codificador contra la `z` **verdadera**
  recuperada en §3. Es la única verdad de terreno que queda sin usar, y diría
  cuánto margen queda.

### 7.8 Cuánto audio hace falta, medido

`clonar_voz.py` admite varias muestras de la misma voz: `--audio` y
`--transcripcion` se repiten emparejados. Cada clip se codifica **por separado**
y se concatenan los latentes; pegar las ondas primero metería un salto artificial
en cada empalme que el encoder convolucional se llevaría a los latentes de
alrededor.

La curva se midió sin pedirle más grabaciones a nadie, con una voz oficial de
oráculo (`scripts/banco_duracion.py`): se generan cuatro clips de referencia
desde textos conocidos, se fabrican prefijos con 1, 2, 3 y 4, y todo se compara
contra la huella ECAPA de la referencia completa. El propio oráculo diciendo las
frases de prueba da el techo.

| | segundos | posiciones | ECAPA | ± |
|---|---:|---:|---:|---:|
| **sp-Spk4_woman** — techo | — | — | **0,8856** | 0,020 |
| 1 clip | 11,7 | 137 | 0,8538 | 0,028 |
| 2 clips | 23,1 | 276 | **0,8854** | 0,025 |
| 3 clips | 35,2 | 424 | 0,8854 | 0,025 |
| 4 clips | 48,1 | 588 | 0,8931 | 0,018 |
| **sp-Spk3_man** — techo | — | — | **0,8814** | 0,006 |
| 1 clip | 11,7 | 137 | 0,8339 | 0,015 |
| 2 clips | 22,0 | 268 | 0,8484 | 0,009 |
| 3 clips | 34,3 | 417 | 0,8509 | 0,014 |
| 4 clips | 48,0 | 587 | 0,8530 | 0,027 |

**Más audio sí mejora, y el salto está entre 12 y 23 segundos.** En la voz
femenina ese tramo vale +0,032 y ya toca el techo; de ahí en adelante la curva es
plana dentro del ruido. En la masculina el mismo tramo vale +0,015 y luego
+0,003 y +0,002: rendimientos decrecientes bruscos.

**Y hay una asimetría que no se puede ignorar.** La voz femenina alcanza su techo
con 23 s; la masculina no lo alcanza ni con 48, se queda 0,028 por debajo. Es la
misma dirección del sesgo que ya aparecía en §7.3 y §7.5, donde las voces graves
clonadas suben de tono y las agudas no. Con las voces graves hay algo que el
prefijo no termina de capturar, y no se arregla con más material.

Lo caro no es fabricar el prefijo, es usarlo: **el prefijo entero viaja en cada
generación**. Pasar de 137 a 588 posiciones cuadruplica la caché KV que arrastra
cada frase, para ganar 0,008 en la voz femenina y 0,005 en la masculina más allá
de los 23 s. No compensa.

> **La recomendación operativa: 25-30 s.** Con menos de 15 s se está en la parte
> empinada de la curva; con más de 30 se paga caché sin ganar nada.

#### El matiz que casi cuesta caro: homogéneos, no cualesquiera

La curva de arriba se midió con clips generados por el mismo modelo en las mismas
condiciones. Con material real dispar, **más audio empeora el clon**.

Medido sobre una voz con tres notas de voz de WhatsApp de distinta sala,
distancia y registro. Lo primero que llama la atención es que entre sí solo dan
0,51–0,59 de ECAPA —zona gris—, cuando las dos mitades del clip largo dan 0,854 y
contra otro hombre da 0,213: es la misma persona grabada de tres maneras
distintas.

| prefijo | segundos | ECAPA del clon |
|---|---:|---:|
| **solo el clip largo** | 31,4 | **0,6837** ±0,029 |
| los tres juntos | 41,1 | 0,5915 ±0,084 |
| solo los dos cortos | 9,6 | 0,4600 ±0,049 |

Diez segundos **más** de material y el clon pierde 0,09. Y el techo explica por
qué: los propios clips cortos, audio real de la persona, puntúan 0,681 y 0,605
contra la referencia conjunta. Son tan distintos que ni él se parece a sí mismo
ahí. El prefijo, al juntarlos, promedia condiciones de grabación en vez de
acumular información de la voz.

`clonar_voz.py` avisa cuando detecta clips que no suenan entre sí a la misma
grabación, y recomienda quedarse con el mejor. **La regla completa es: 25-30 s de
la misma sesión, mismo micro y misma distancia.** Si solo hay una toma buena y
varias malas, la buena sola gana.


### 7.9 Dónde corre cada cosa, medido en la VM

Fabricar un prefijo **no cabe en la VM**, y no es una estimación. Probado en `voz`
(4.909 MB de RAM, 12 núcleos, con los tres servicios en marcha y 1.671 MB
libres), cargando solo el modelo en fp32 dentro de un *scope* con tope de
memoria para no arriesgar producción:

```
Memory cgroup out of memory: Killed process 8857 (python3.12)
  total-vm:3222508kB, anon-rss:1627672kB
run-p8857-i8858.scope: Failed with result 'oom-kill'
```

Murió al llegar a 1,63 GB, y eso es **solo el modelo**: el codificador de la
comunidad son 1,3 GB más en F32. Con el pico de 4,1 GB que ya estaba medido para
el fp32 completo, el total ronda los 5,4 GB — más de los 4,9 GB que tiene la
máquina entera, así que tampoco cabría parando `voz-stream`.

El reparto que sí funciona:

| pieza | dónde | por qué |
|---|---|---|
| `clonar_voz.py` | estación de trabajo | necesita modelo + codificador, ~5,4 GB |
| el `.pt` resultante | la VM, en `VIBEVOICE_VOCES` | 2,6–8,4 MB, se carga como cualquier voz oficial |
| `voz-stream` | la VM | **no necesita el codificador**: solo hace falta para fabricar el prefijo |
| `prosodia.py`, `espectro.py` | cualquiera de los dos | numpy puro; probados en la VM sobre audio que ella misma generó |
| `banco_clonado.py`, `banco_duracion.py` | estación de trabajo | generan cientos de locuciones |

Es la misma lógica que el resto del repositorio: lo caro se hace fuera y a la
máquina llega el artefacto pequeño. Un prefijo es a una voz lo que un `.onnx` de
Piper es a las suyas.

### 7.10 El techo de la voz: medirlo antes de clonar

Toda la sección 7 mide clones contra su referencia, y §7.8 ya usaba un número
que merece nombre propio: la referencia partida en dos mitades, una contra la
otra, con la misma huella ECAPA. Es el **techo** de esa grabación, lo máximo que
puede sacar cualquier motor con ese audio, y desde septiembre de 2026 se calcula
antes de fabricar cada voz.

Por qué importa quedó demostrado a base de perder tres doblajes
([comparativa-motores.md §7](comparativa-motores.md)): dos voces del mismo
vídeo, cortadas de la anotación humana, y dos destinos opuestos.

| voz | material | techo | mejor clon (VibeVoice, es) | % del techo |
|---|---|---|---|---|
| Laura | 30,0 s | **0,946** | 0,65 | 61 % |
| Juan Pablo | 19,3 s | **0,598** | 0,52 | 87 %, y aun así en la zona gris |
| voz de trabajo (§2 de la comparativa) | 28,0 s | 0,764 | 0,54 | 71 % |

Juan Pablo no llega al umbral de "misma persona" (0,626) **consigo mismo**: sus
dos mitades no se reconocen. Ningún banco limpio, ninguna semilla y ningún motor
lo subieron, porque el límite lo ponía la grabación. Sin este número se
confunden dos problemas con arreglos opuestos: "el clon es malo" (más
referencia, otra semilla, otro cfg) y "esta voz no se deja clonar" (grabar
mejor, y nada más).

Cómo se usa:

```bash
# solo medir: una grabacion, o varias de la misma voz
python scripts/techo.py nota.opus
# elegir referencia dentro de una charla: el mejor tramo de 25 s, no el mas largo
python scripts/techo.py charla.wav --tramos 25 --salto 5
```

`clonar_voz.py` y `clonar_voz_qwen.py` lo imprimen **antes de cargar el modelo**
y dejan una ficha `<voz>.json` junto al `.pt` (o al `.wav`/`.bin`) con el techo,
los segundos y las fuentes. `--techo-minimo 0.70` se niega a fabricar por
debajo; sin él solo avisa. El veredicto:

| techo | qué esperar |
|---|---|
| ≥ 0,85 | buena: el clon puede pasar de 0,60 |
| 0,70 – 0,85 | aceptable: la zona de las voces de trabajo |
| 0,626 – 0,70 | floja: el clon queda en la zona gris; mejor regrabar |
| < 0,626 | no se reconoce a sí misma: no hay clon que valga |

Los dos motores probados rinden un porcentaje parecido del techo de cada voz
(46-61 %). Con dos voces es indicio, no ley, pero apunta a que la palanca más
grande de calidad, y la única que no cuesta CPU, es grabar referencias con
techo alto: 25-30 s seguidos, micro cerca, sin solapes ni música.

**Comprobación de la herramienta** (9 de septiembre de 2026, mismo juez): sobre
la referencia de la comparativa da **0,764**, el mismo número que salió allí.
Sobre los cuatro hablantes del vídeo de Laura y Juan Pablo, cortados con
`banco_motores.py referencia` a partir de la anotación humana:

| hablante | segmentos en orden, con solapes | sin solapes, los más largos primero (regla de `dobla`) |
|---|---|---|
| 1 (Laura) | **0,019** (23,6 s) | **0,890** (32,7 s) |
| 2 (Juan Pablo) | sin segmentos limpios | 0,608 (11,7 s, un clip con 5,1 s de otra voz) |
| 0 | 0,644 | 0,655 |
| 3 | 0,653 | 0,653 |

La primera columna es la lección: **un segmento con otra voz encima no baja el
techo, lo destruye**. De 0,019 a 0,890 con el mismo material, solo quitando los
tramos donde hablan a la vez. Por eso `referencia` aplica ahora la regla de
pureza de `dobla` (contaminación < 0,3 s, los más largos primero, un clip sucio
solo si no hay ni 4 s limpios). Los 0,890 y 0,608 cuadran con los 0,946 y 0,598
que midió `dobla` con sus propios cortes de 30 y 19 s: la diferencia es qué
segundos entran, no la métrica.

### 7.11 La semilla del clonado: elegirla midiendo

El clonado tenía un sorteo escondido. El codificador acústico devuelve una
distribución y `clonar_voz.py` la muestrea (`e.sample`, `torch.randn`) sin
fijar semilla: dos prefijos de la **misma** referencia salían distintos, y
nadie lo medía porque cada `.pt` se fabricaba una vez. Se vio en el doblaje
del vídeo de 4 voces: con las mismas referencias, la identidad QC de una voz
dio 0,53 en una corrida y 0,39 en la siguiente. Esa varianza tapa cualquier
mejora pequeña y, peor, decide por sorteo cómo suena una persona.

Desde el 9 de septiembre de 2026:

- `clonar_voz.py --semilla N` (por defecto 11; `-1` deja el sorteo libre)
  fija el muestreo, así que la misma referencia da el mismo `.pt`. Con
  `--lote`, cada voz puede traer su `semilla`. La ficha `<voz>.json` la guarda.
- `scripts/banco_semillas.py` la **elige midiendo**: fabrica el prefijo con
  varias semillas (el modelo cargado una vez), sintetiza las mismas frases con
  las mismas semillas de síntesis, y mide cada clon contra la referencia:
  ECAPA (media y mínimo), `sesgo_st` (el tono), `car/s` (el ritmo) y WER.
  Ordena por identidad, desempata por tono y WER, deja `<voz>.pt` con la
  ganadora y `<voz>.json` con la tabla entera. La tabla es lo importante: son
  todos los tonos que el clonado puede sacar de esa grabación, y cuánto vale
  cada uno.

```bash
python scripts/banco_semillas.py --audio ref.wav --transcripcion "..." \
    --nombre laura --salida voces/         # 5 semillas x 4 frases x 2 sintesis
```

Medido el 9 de septiembre sobre las dos voces del banco de `dobla` (referencias
del vídeo de 4 voces, 5 semillas de clonado × 4 frases × 2 semillas de
síntesis, cfg 3, 6 pasos, MPS). El tono es la mediana por semilla con el
detector acotado a la banda de la voz (ver el aviso de abajo):

| voz | semilla | puntuación | ECAPA | mín | es | en | tono | WER |
|---|---|---|---|---|---|---|---|---|
| Laura (techo 0,814, f0 247 Hz) | **5** | 0,646 | **0,655** | 0,428 | 0,715 | 0,476 | +0,05 st | 8,6 % |
| | 1 | 0,636 | 0,648 | 0,496 | 0,695 | 0,507 | +0,77 st | **4,7 %** |
| | 2 | 0,627 | 0,639 | 0,421 | 0,705 | 0,443 | +0,41 st | 8,0 % |
| | 3 | 0,622 | 0,635 | 0,426 | 0,701 | 0,440 | +0,22 st | 11,3 % |
| | 4 | 0,613 | 0,626 | 0,363 | 0,691 | 0,430 | +0,23 st | 10,6 % |
| Juan Pablo (techo 0,51, f0 171 Hz) | **4** | 0,471 | **0,499** | 0,397 | 0,529 | 0,407 | −1,65 st | **11,3 %** |
| | 2 | 0,436 | 0,465 | 0,372 | 0,495 | 0,372 | −1,13 st | 17,9 % |
| | 5 | 0,425 | 0,459 | 0,245 | 0,505 | 0,322 | −1,42 st | 19,8 % |
| | 1 | 0,419 | 0,458 | 0,350 | 0,476 | 0,405 | −1,71 st | 21,7 % |
| | 3 | 0,416 | 0,451 | 0,360 | 0,479 | 0,367 | −1,93 st | 16,0 % |

Lo que dice la tabla:

- **La semilla mueve poco la identidad de una voz buena y bastante la de
  una mala.** Laura: 0,626-0,655 (0,03 de rango). Juan Pablo: 0,451-0,499, y
  el WER del 11 % al 22 %: la mejor semilla le saca un 10 % de identidad y la
  mitad de errores a la peor. Con techo 0,51 sigue siendo una voz inclonable,
  pero el sorteo decidía si quedaba en 0,45 o en 0,50.
- **El tono no depende de la semilla.** Laura sale entre +0,05 y +0,77 st de
  su f0 real con cualquier semilla; Juan Pablo entre −1,1 y −1,9. El sesgo
  es de la voz (grave, hacia abajo; §7.8 medía +2,5 st hacia arriba en otras
  voces graves con otra referencia), no del sorteo.
- **La puntuación es compuesta**: `ECAPA − 0,01·|tono en st| − 0,1·WER`
  (0,01 de ECAPA equivale a 1 st o a 10 % de WER). En Laura decide entre la
  5 (más identidad) y la 1 (mejor WER y mejor mínimo); en Juan Pablo la 4
  gana en todo.

> **Aviso sobre medir el tono.** La primera versión de esta tabla decía que
> la semilla movía el tono hasta 4 semitonos (Laura de −1,9 a −6,2 st). Era el
> detector: la autocorrelación con la banda entera (60-400 Hz) caía en el
> subarmónico en algunos clips y daba −12 st, una octava, y la media
> arrastraba. Acotando la búsqueda a [f0/1,5, f0·1,5] de la voz (un error de
> octava necesita un factor 2) el sesgo real es el de arriba. `prosodia.
> contorno` admite ahora `f_min`/`f_max`, y `tono.sesgo_st` los usa.

El banco escribe `voces/<id>/semilla.json` para el banco de identidades de
`dobla`, y el doblaje la usa (`--semilla-clon` es el valor por defecto para
las voces sin ficha).

### 7.12 Corregir el tono a la salida: medido, y no compensa

Con el sesgo de tono medido por voz (§7.11), lo natural era corregirlo a la
salida: desplazar el clon los semitonos que le faltan, sin tocar la
duración. `scripts/tono.py` lo hace de dos maneras, solo con numpy, y
`scripts/banco_tono.py` mide antes y después con el mismo juez sobre los
clips de la semilla elegida (8 por voz):

| voz | versión | ECAPA | mín | tono medio | WER |
|---|---|---|---|---|---|
| Laura | sin corregir | **0,648** | 0,496 | +0,54 st | **4,5 %** |
| | remuestreo + WSOLA, por voz | 0,331 | 0,129 | (mal medido) | 5,5 % |
| | PSOLA, por voz (−0,77 st) | 0,545 | 0,310 | +0,70 st | 5,2 % |
| Juan Pablo | sin corregir | **0,499** | 0,397 | −1,49 st | **11,3 %** |
| | remuestreo + WSOLA, por voz | 0,150 | 0,105 | (mal medido) | 16,1 % |
| | PSOLA, por voz (+1,65 st) | 0,406 | 0,212 | −0,39 st | 20,7 % |
| | PSOLA, por clip (el tope) | 0,410 | 0,219 | −0,37 st | 17,6 % |

- **Remuestreo + WSOLA** (el mismo WSOLA de `estirar.py`) mueve el tono y las
  formantes con él: el juez de identidad lo ve como otra persona (0,648 →
  0,331). Descartado.
- **TD-PSOLA** (marcas de periodo alineadas al pico, ventana de dos periodos,
  reubicadas a T/r) conserva las formantes y clava el tono (Juan Pablo de
  −1,5 a −0,4 st), pero cuesta **0,10 de identidad y sube el WER** (11 → 21 %
  en Juan Pablo). Incluso corrigiendo cada clip con su sesgo exacto (el tope
  teórico) el resultado es peor que sin tocar.
- **Y el sesgo real es pequeño**: +0,5 st en Laura, −1,5 en Juan Pablo. La
  cifra de 2,5 st de §7.8 sigue siendo cierta para aquella referencia, pero no
  es general.

Decisión: **no se corrige el tono a la salida.** `tono.py` queda como
herramienta (`--st`, `--referencia`) y `banco_tono.py` como el banco que hay
que superar si alguien vuelve a intentarlo: una corrección que no baje la
identidad ni suba el WER sobre esos mismos clips. La palanca del tono, si
hace falta, está en la referencia (grabar a la persona en su registro) y en
la semilla, no en el posprocesado.

---

> **Sobre clonar voces ajenas.** Esto convierte 15 segundos de audio en una voz
> reutilizable. Es la capacidad por la que Microsoft no publicó el codificador.
> Clonar a alguien sin su consentimiento no es un uso de este repositorio.

---

## Reproducir

Intérprete: `pkgs/vibevoice/.venv/bin/python`. Modelo en
`~/.cache/vibevoice-nix/modelo`, voces en `~/.cache/vibevoice-nix/voces`.

El encoder del 1.5B se baja sin traer los 5,4 GB del repo, leyendo la cabecera
del safetensors y pidiendo por `Range` solo los 276 tensores del encoder
(687 MB); las claves del 1.5B tienen exactamente los nombres que espera el
código instalado, así que el `state_dict` entra con `strict=True` en
`modelo.model.acoustic_tokenizer.encoder`.

Las medidas de esta nota salen de `tono()` en `scripts/sondeo_voz.py` y de
`transcribir`/`normalizar`/`wer` en `scripts/fidelidad.py`, con el whisper del
stack en el 8080.
