# Clonado de voz

Resumen para quien tenga prisa:

- **Fabricar prefijos de voz `.pt` para el 0.5B: RESUELTO.** El formato estaba
  sin documentar y aquí queda descrito y verificado. Un prefijo fabricado a
  mano reproduce el oficial con coseno ≥ 0,9993 en las cuatro ramas, y
  sintetiza con la misma calidad (medida abajo).
- **Convertir un audio nuevo en esos latentes: NO RESUELTO.** El 0.5B no trae
  codificador acústico y el del 1.5B **no vale tal cual**: vive en otra base
  del mismo espacio de 64 dimensiones. Un adaptador lineal recupera parte
  (R² ≈ 0,59) y produce habla inteligible con timbre aproximado, pero pierde
  la identidad en voces alejadas de la media y desestabiliza la entonación.

Es decir: la mitad difícil del problema (el formato del prefijo) está hecha;
lo que falta es un codificador para el 0.5B, y la vía para conseguirlo queda
descrita al final con la evidencia de por qué debería funcionar.

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

### No se puede hoy

- Clonar una voz a partir de un audio arbitrario con fidelidad aceptable.

### La vía que sí debería funcionar

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
