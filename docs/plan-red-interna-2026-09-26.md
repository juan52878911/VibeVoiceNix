# Plan: sondear y manipular la red por dentro. Qué se hereda del 1.5B y qué más se saca del 0.5B (2026-09-26)

Informe de lectura hecho por un agente sobre el repositorio (documentos, bancos, código, historial).
No se ha medido nada nuevo ni se ha tocado código. Etiquetas: **(M)** medido en el proyecto, **(E)**
estimado con la cuenta a la vista, **(S)** suposición sin medir. Complementa el
[plan de emoción e intención](plan-emocion-intencion-2026-09-26.md): allí está el producto (el mando
en la API) y las puertas; aquí está **cómo mirar dentro de la red para encontrar los mandos**, sobre
qué red hacerlo, y qué otras cosas nuevas se pueden sacar de los pasos que el modelo ya da.

Las tres preguntas de Juan, con la respuesta corta:

1. **¿Sobre el 0.5B original o sobre uno más grande, para heredar capacidades?** Sobre el 0.5B, que
   es el único que habla español, el único que corre en la VM y el único con codificador para clonar.
   El 7B («Large») ya no está publicado (HTTP 401, `clonado-de-voz.md` §1) (M). El 1.5B solo habla
   inglés y chino, cuesta del orden de 3× por fotograma (E) y su espacio latente es otro (M). Sirve
   como **maestro fuera de línea** (genera habla expresiva en inglés que el codificador comunitario
   convierte en datos del 0.5B) y, si algún día pasa la puerta de `comparativa-motores.md` §7, como
   motor de lote para dobla en inglés. Nunca como base de producción. §2.
2. **¿Qué aprovechamos del repo?** Casi todo el laboratorio ya existe: la pasada forzada que abre la
   red en una llamada, el bucle LoRA, los jueces, el banco de producción y los puntos de enganche en
   `voz_stream.py` y `motor.py`. §3.
3. **¿Qué más se puede hacer dentro del modelo?** Ocho usos nuevos de pasos que el modelo ya da,
   sin pasadas extra: la rama negativa como mando de estilo, el arranque como selector de registro,
   el énfasis desde el embedding del texto, la caché KV bifurcada para repetir solo una frase, la
   verosimilitud del propio modelo como control de calidad y detector de descarrilamiento, la atención
   como alineador de palabras, los latentes como códec de voz, y una sonda de frontera para el respiro. §5.

---

**Estado (26-09):** herramientas escritas en `scripts/red/` y probadas; lo que se pudo medir sin los pesos (los 25
prefijos como estados internos: la identidad tiene forma de U por capa, mínima en las 9-11; el sexo es lineal y
generaliza) y lo que queda para el Mac está en
[bancos/2026-09-26-red-interna-entorno.md](bancos/2026-09-26-red-interna-entorno.md).

## 1. La familia VibeVoice, vista desde este proyecto

| | Realtime-0.5B (producción) | VibeVoice-1.5B | VibeVoice-ASR (7B) y ASR-BitNet | AcousticTokenizer |
|---|---|---|---|---|
| Idiomas | en, zh y **es experimental** (M) | en, zh (M) | reconocimiento, no síntesis | — |
| LM | `language_model` 4 capas + `tts_language_model` 20 capas, 896 (M) | un `language_model` de 28 capas, 1536 (M) | 3584 | — |
| Pesos del LM | 392 + 596 MB (M) | 3087 MB (M) | ~7B | — |
| Cabeza de difusión | 84 MB (M) | 247 MB (M) | — | — |
| Codificador acústico | **ausente en el oficial**; el comunitario (MIT, 344 M, reentrenado desde el 1.5B) vive en el espacio del 0.5B: identidad 0,97 en ida y vuelta (M) | sí, y **comparte** tokenizador con ASR y AcousticTokenizer (M) | sí | sí |
| Espacio latente frente al 0.5B | — | **otro**: un adaptador lineal 64×64 recupera R² ≈ 0,59 y la cadena completa no aguanta (M) | el mismo que el 1.5B (M) | el mismo que el 1.5B (M) |
| Tokenizador semántico | no | sí, 689 MB, solo codificador (M) | sí | — |
| Prefijo de voz | caché KV precalculada (`.pt`) (M) | se calcula en inferencia desde audio crudo (M) | — | — |
| Streaming | sí, decodificador causal con caché (M) | bucle reconstruido en `scripts/deriva/gen15b.py`; decodifica en bloque (M) | — | — |
| Coste por fotograma en la VM | ~108 ms (M) | ≈ 3× en LM y cabeza por el tamaño de los pesos (E): fuera del tiempo real | GPU | — |

**Lo que se puede heredar del 1.5B y lo que no:**

- **Sí, como datos.** El 1.5B genera diálogos de varios hablantes, largos y con una expresividad que
  el 0.5B no tiene por sí solo (S, es la afirmación del informe técnico; aquí no está medida). Su
  audio, pasado por el codificador comunitario, son latentes del 0.5B: el camino normal de `datos.py`,
  sin adaptador entre espacios. Con eso se destila estilo al 0.5B con el LoRA de siempre. Es inglés,
  pero F7 mostró que el estilo aprendido en un idioma **se aplica al clon en español** (M).
- **Sí, como juez y como maestro de sondas.** Las mismas sondas de §4 valen en el 1.5B; si un atributo
  se lee mejor allí, su dirección se puede usar de objetivo al entrenar el 0.5B (S).
- **No, como base.** Sustituir el `tts_lm` por uno mayor obliga a reentrenar conector, cabeza y
  clasificador de fin, y a refabricar las 61 voces: es entrenar otro modelo. Fuera.
- **No, en la VM.** Ni por RTF ni por memoria (VmHWM ~2 GB con 5 GB de VM).
- **A medir, como motor de lote en inglés para dobla** (c8a/m8a o GPU): solo si supera la tabla de
  `comparativa-motores.md` §7 con las mismas voces y el mismo juez, que es la regla del proyecto.

---

## 2. Qué se aprovecha del repo: el laboratorio ya existe

| Pieza | Ruta | Para qué sirve aquí |
|---|---|---|
| **Pasada forzada** diferenciable (coseno 1,000000 con `generate()`) | `scripts/lora/forzado.py` | abre la red: para audio real con su texto da la condición de cada fotograma, los estados de las 20 capas y la rama negativa en **una** llamada. Es el instrumento de todo §4 |
| Condiciones de audio real dentro de `generate()` | `scripts/fase3_condiciones.py` | lo mismo con el bucle de producción, fotograma a fotograma; guarda `cond [T, 896]` |
| Bucle LoRA completo (LoRA sin peft, datos CML-TTS/LibriTTS-R, entrenamiento, evaluación generando, comparación pareada con IC) | `scripts/lora/` | entrenar cualquier cosa sobre los LM; puertas 0 y 1 ya validadas |
| Datos con consentimiento por pausas y marcas por palabra | `scripts/lora/datos_voz.py` | pares reales de la persona; alineación palabra a palabra para las sondas de énfasis |
| Jueces: WER, ECAPA, UTMOS, PER, AST, estilo frente a la persona, historia de un vídeo | `scripts/juez_lote.py`, `juez_acento.py`, `juez_sonidos.py`, `lora/juzgar_estilo.py`, `lora/juzgar_historia.py` | las puertas |
| Descriptores por clip: tono, recorrido, microvariación, sílabas/s, pausas/min, energía, inclinación, HNR; contorno | `scripts/perfil_vocal.py`, `scripts/prosodia.py` | **las etiquetas gratis de las sondas**: se calculan sobre cualquier audio |
| Banco de producción (238 parejas) y prueba de despliegue bit a bit | `scripts/banco_ab.py`, `scripts/ws_fidelidad.py` | nada entra sin pasarlos |
| Elección de semilla y de ruido de arranque por voz contra el motor real | `scripts/banco_semillas.py`, `scripts/elegir_arranque.py` | el «arranque por emoción» de §5.2 es el mismo barrido con otro juez |
| Fabricar prefijos (positivo y negativo) desde audio | `scripts/clonar_voz.py`, receta en `clonado-de-voz.md` §3 | prefijos emocionales y **prefijos negativos** (§5.1) |
| Puntos de enganche en el servicio: `sample_speech_tokens` sustituido, `muestrear_reforzado`, `instrumentar_latentes`, `foto_generacion`/`reponer_generacion` | `pkgs/vibevoice-cli/voz_stream.py` | donde se suma una dirección a la condición, donde se cambia el ruido de arranque, y cómo viaja un estado con la sesión |
| Grafos OpenVINO y sus conversores | `pkgs/vibevoice-ov/motor.py`, `convertir_*.py` | añadir entradas al IR (dirección, fuerza α), leer estados intermedios |
| Corpus congelado | `scripts/corpus_mejora.json` | se amplía, no se cambia |
| Bucle del 1.5B reconstruido y control de decodificación en bloque | `scripts/deriva/gen15b.py`, `redecodificar.py`, `sonda_eps.py` | generar con el 1.5B como maestro; la sonda del lazo CFG ya mide `std_cond`, `inflado` y `norma_cond` por paso |
| Modo taller de la VM (clon con 11 GB) y entorno de GPU | `scripts/modo_taller.sh`, `crear_taller.sh`, `lora/gpu_entorno.sh` | CPU para sondas y síntesis; g4dn para forzar corpus grandes |
| Archivo para escuchar (web estática del NAS) | `scripts/historial_audios.py` | oír lo que hace cada componente principal de §4.4 |

**Lo que no hay y hay que escribir:** el guardado de los residuales por capa en la pasada forzada
(hoy solo devuelve la condición), la sonda ridge por capa, el parcheo de activaciones, la lectura de
atenciones (en torch; el IR no las expone), y una entrada `direccion`/`alfa` en el IR del LM.

---

## 3. Dónde mirar: el mapa del modelo

```
texto ──► language_model (4 capas) ──► h_lm (896 por ficha)         ◄── §5.3 énfasis por ficha
                                            │ + tipo texto
prefijo .pt (KV: latentes de la voz + su texto)                     ◄── §5.1 prefijo negativo, E3 prefijo emocional
                                            ▼
                    tts_language_model (20 capas, residual 896)     ◄── §4 sondas y direcciones por capa
                       │ estado de salida = condición (896)         ◄── E2 dirección sobre la condición
                       │ clasificador de fin                         ◄── §5.8 sonda de frontera
                       ▼
     cabeza de difusión (6 pasos; guía cfg 3,0 = cond + 3·(cond − neg); freno 0,75)
                       │ ruido inicial (semilla; 6 primeros fotogramas con semilla propia)  ◄── §5.2 arranque
                       ▼
                 latente 64 (7,5 por segundo) ──► conector ──► vuelve al tts_lm     ◄── §5.4 bifurcar la KV
                       │                                                             ◄── §5.5 verosimilitud
                       ▼
        decodificador σ-VAE causal ──► audio                                         ◄── §5.7 códec
```

La rama negativa del CFG es un `tts_lm` paralelo que arranca de `<|image_pad|>` y solo ve los
latentes ya generados: **sostiene la voz** (quitarla cuesta −0,04 de identidad, M) y cuesta lo mismo que
la positiva (M).

---

## 4. Programa de sondeo y manipulación (sobre el 0.5B, sin GPU salvo forzar corpus grandes)

Principio: no hace falta un corpus de emociones para encontrar los mandos. Cada fotograma de audio
real se etiqueta con sus descriptores (gratis), se busca en qué capa cada descriptor es **linealmente
legible**, y se interviene ahí. La emoción se compone después de mandos bajos (§4.4) o se aprende de
pares cuando los haya (plan de emoción, §3.2).

### 4.1 I0 · Instrumentar la pasada forzada (Mac, medio día)

`forzado.py` devuelve además el residual de cada capa del `tts_lm` en las posiciones de voz, la
condición, la salida de la rama negativa y el logit de fin. Sobre CML-TTS/LibriTTS-R (en, es, de,
fr; los ~2.400 ejemplos de la guía destilada ya forzados: 183.000 fotogramas) y sobre las grabaciones
con consentimiento. En fp16, 20 capas × 896 × 183.000 fotogramas son ~6,5 GB: se guarda cada 2 capas
o se muestrea 1 de cada 3 fotogramas.

Etiquetas por fotograma (todas de código existente): F0 en semitonos respecto a la mediana del
hablante, energía respecto a la media del clip, sílabas/s locales, «es pausa», «quedan k fotogramas
para la pausa siguiente», «quedan k para el fin», «la frase es pregunta», «esta ficha lleva énfasis»
(energía y F0 de la palabra por encima de sus vecinas, con las marcas de whisper).

### 4.2 I1 · Sondas lineales por capa (Mac, 1 día)

Ridge por capa y por etiqueta, con las medias por hablante restadas (para que la dirección sea
**intra**hablante y no lleve identidad), validación por hablante apartado. Salida: R² por capa y
etiqueta, y una dirección unitaria por (capa, etiqueta).

**Puerta:** R² de validación > 0,3 en alguna capa para tono, energía y pausa inminente; si ningún
atributo prosódico es lineal en ninguna capa, la manipulación por direcciones se cierra y queda solo
el LoRA (plan de emoción, E4). Este mapa sustituye la división «LM = ritmo, difusión = entonación»
que el plan de personalidad supuso a mano.

### 4.3 I2 · Intervención causal: sumar, quitar y parchear (Mac + VM, 2-3 días)

Tres operaciones sobre la capa que la sonda señale, medidas **generando** con el motor de producción:

1. **Sumar**: `r' = r + λ · rms(r) · d`, λ por fotograma. El mando.
2. **Proyectar fuera**: `r' = r − (r·d) d`. Quita un atributo (por ejemplo, la subida final de
   pregunta) y dice si la dirección es causal o solo correlada.
3. **Parchear**: copiar el residual de la capa L de una locución A (aguda, rápida) en la locución B
   en las mismas posiciones y ver qué atributo viaja. Separa «aquí se lee» de «aquí se decide».

Dos decisiones que se miden, no se suponen: en qué rama (solo la condicional, que la guía amplifica
×3, o las dos) y en qué sitio (condición final: dos líneas en `sample_speech_tokens`; residual de una
capa: una entrada más en el IR del LM). La 3b enseñó que mover la condición un 29 % rompe el habla:
barrido de λ en 0,02-0,20 de la norma y el WER decide.

**Puerta, por atributo y por voz (3 clones con consentimiento + 2 de serie, semillas 11 y 101):** el
descriptor objetivo se mueve en la dirección pedida con IC > 0 y **monótono en λ**; ECAPA ≥ −0,005;
UTMOS IC inferior ≥ −0,02; WER ≤ +0,5 puntos; ningún clip con WER > 25 % si la base tenía ≤ 10 %.
Pasa la λ máxima que cumpla todo, y es la que va a la ficha de la voz.

### 4.4 I3 · Componer mandos altos y descubrir mandos sin etiquetas (Mac, 1 día)

- **Composición:** alegría ≈ tono alto + energía + velocidad; tristeza ≈ lo contrario con más pausas;
  calma ≈ velocidad baja y energía plana. Se compara con la dirección aprendida de pares (plan de
  emoción, E2) cuando exista: si coinciden (coseno > 0,7) la composición basta y no hacen falta
  corpus de emociones; si no, gana la aprendida.
- **PCA/ICA** sobre las condiciones de miles de fotogramas: sumar cada componente principal con λ
  pequeña, generar 4 frases con 2 voces, escuchar y medir descriptores. Lo que se mueva y no rompa
  el WER se nombra y entra al catálogo. Es la forma más barata de encontrar mandos que nadie pensó.

### 4.5 I4 · Atención: qué cabezas miran el texto y cuáles el prefijo (Mac, 1 día)

En torch, con salida de atenciones, sobre 50 locuciones: masa de atención de cada cabeza (20 capas ×
14 cabezas) hacia las fichas de texto, hacia los latentes del prefijo y hacia los latentes generados.
Salen tres cosas: el **alineador** fotograma → palabra (§5.6), las cabezas que **sostienen la
identidad** (las que no se pueden tocar al dirigir) y, si alguna cabeza sigue la puntuación, un mando
de intención sin dirección alguna.

### 4.6 I5 · Cirugía puntual de pesos (Mac, medio día cada una)

Solo dos sitios con sentido y baratos de deshacer: el **sesgo del clasificador de fin** (cuánto alarga
o corta; el bloque COLA FINAL ya lo demora con código) y los dos embeddings de tipo
`tts_input_types` (la única etiqueta por posición que la red recibe: escalarlos o añadirles una
dirección dice si sirven de mando global). Puerta: la de I2.

---

## 5. Aprovechar un paso del modelo para hacer algo nuevo

Todos sin pasadas extra del backbone; los marcados con RTF dicen lo que cuestan.

### 5.1 La rama negativa como mando de estilo (0 en RTF)

Hoy la rama negativa arranca de `<|image_pad|>` y ve solo lo generado. La guía empuja desde ella:
`eps = neg + 3·(cond − neg)`. Si la rama negativa arranca de un **prefijo de la misma persona hablando
plano** (un clip monótono real, o habla generada con λ de «plano» de §4.3), la guía empuja **lejos de
lo plano** en cada fotograma: expresividad por contraste, con la voz sostenida porque el negativo es
la misma voz. Cuesta fabricar un `.pt` negativo con `clonar_voz.py` (receta §3 del doc de clonado,
pasos 3 y 4) y nada en tiempo de ejecución. Variantes: negativo «triste» para pedir alegría,
negativo «sin pausas» para pedir pausas. Riesgo (S): el modelo nunca vio un negativo con voz; el freno
de guía y la rampa de arranque están escritos sobre las dos ramas y hay que remedirlos. Puerta: la de I2.

### 5.2 El arranque como selector de registro (0 en RTF)

Está medido que los 6 primeros fotogramas deciden el registro de toda la locución: el ruido inicial
elige la música inventada y la realimentación del LM la mantiene (M). Dos usos:

- **Semilla de arranque por emoción y por voz:** el barrido de `elegir_arranque.py` con el juez de
  emoción como criterio. Cero coste; una entrada más en la ficha de la voz.
- **Arranque con latentes reales:** los 4-6 primeros latentes salen del clip emocional de la persona
  (codificado) en vez de la difusión, y el LM continúa en ese registro. Es el mismo mecanismo que el
  ruido de arranque con otra fuente. Riesgo (S): la costura entre lo real y lo generado.

### 5.3 Énfasis por ficha, desde el embedding del texto (0 en RTF)

El énfasis local no necesita alinear fotogramas: la rama de texto da un estado `h_lm` **por ficha** y
el `tts_lm` los consume por ventanas de 5. Escalar o sumar una dirección a las fichas de la palabra
marcada (`{énfasis}palabra{/énfasis}`, quitado por el servidor) actúa sobre esa palabra por
construcción. La dirección sale de la sonda de énfasis de §4.2 sobre las posiciones de texto, o de la
diferencia media `h_lm` de fichas enfatizadas frente a no enfatizadas en el corpus con marcas. Puerta:
el énfasis cae en la palabra pedida (juez de E0) en ≥ 80 % de los clips, WER ≤ +0,5. Si funciona, es
el control local en vivo sin retardo que calibrar.

### 5.4 Bifurcar la caché KV: repetir solo la frase que falló (ahorro en dobla)

La generación es determinista por semilla y el estado entero de una locución es la caché KV más las
colas del decodificador (`foto_generacion` ya las recoge; en OpenVINO los estados se leen y escriben
con `query_state`). Guardar una foto en cada frontera de frase permite **volver a generar desde la
frase k** con otra semilla, otra λ o una palabra cambiada, sin rehacer lo anterior. En dobla, las
re-tiradas del QC costaron 1.268 s en el vídeo de 74 min (M) y hoy repiten el segmento entero; con la
foto se repite solo el trozo malo. En el asistente permite «dilo otra vez, más alegre» sobre la
última frase. Coste: memoria de la foto (20 capas × 2 × 896 × L en fp16, unos 36 KB por posición) y
un `copy` por frontera.

### 5.5 La verosimilitud del propio modelo como QC y detector de descarrilamiento

El modelo da dos señales por fotograma que hoy se tiran: la **pérdida de difusión** de la cabeza sobre
el latente que acaba de generar (con el mismo ruido, cuánto le sorprende) y el **logit de fin**. En
lote (dobla), pasar cada segmento generado por la pasada forzada da una sorpresa media por segmento;
si correlaciona con el WER de whisper (a medir sobre los 216 segmentos de F1 con sus dos semillas),
puede ordenar las re-tiradas antes de transcribir: el QC es el 47 % del coste del job (M). En vivo,
una racha de fotogramas con sorpresa alta es el aviso temprano de música inventada o de alucinación:
`LocucionDescarrilada` hoy se dispara tarde, por duración. Coste en vivo: una pasada más de la
cabeza (2,6 ms, M) si se quiere la sorpresa del latente elegido; en lote, cero.

### 5.6 La atención como alineador de palabras (juez y datos)

Las cabezas de I4 que miran el texto dan, por fotograma, la ficha que se está diciendo: marcas por
palabra sin whisper. Usos: alinear las grabaciones de `datos_voz.py` con el propio modelo (whisper
large-v3 hoy), medir el retardo lectura-habla que el respiro no conoce, y el énfasis en vivo del plan
de emoción (E5) si §5.3 no basta. Solo en torch (el IR no expone atenciones): es herramienta de
laboratorio y de lote, no de la VM.

### 5.7 Los latentes como códec de voz y como archivo (E)

Un latente son 64 números por 133 ms: 480 valores por segundo, ~1 KB/s en fp16 y ~4 kbit/s en int8
(E), frente a 32 kbit/s del Opus de las notas de voz. El codificador comunitario ida y vuelta conserva
identidad 0,97 (M). Usos: guardar los rellenos del asistente y los segmentos de dobla como latentes
(re-decodificables con cualquier decodificador futuro, por ejemplo el podado de R2, sin resintetizar),
transmitir voz por el túnel a 4 kbit/s con el decodificador en el receptor, y **conversión de voz en
el espacio del modelo**: sustituir cada latente de una fuente por su vecino más cercano en un banco de
latentes de la persona (la idea de kNN-VC sin WavLM, con el decodificador que ya está cargado). Lo
último es lo más incierto (S): los latentes acústicos mezclan contenido y timbre, al revés que WavLM;
se mide con la puerta de F4 antes de creerlo.

### 5.8 Una sonda de frontera para el respiro

El respiro solo puede alargar donde el modelo ya pausó, porque no hay ancla de frontera: el
clasificador de fin es plano en las fronteras internas (M). La etiqueta «quedan k fotogramas para la
pausa siguiente» de §4.1 busca esa ancla en los residuales, donde nadie ha mirado. Si una capa la lee
con R² > 0,3, el respiro y `forma` ganan el anclaje que les faltaba, y las pausas de la persona se
pueden poner donde ella las pondría, no solo donde el modelo se calló.

### 5.9 El 1.5B como generador de datos de estilo (GPU, ~4 USD)

Diálogos y monólogos expresivos en inglés generados con `gen15b.py` en una g4dn (RTF sin medir;
E 0,3-0,5), codificados con el comunitario y añadidos a `datos.py` como hablantes sintéticos. Sirven
para las sondas (más variedad de registro que los audiolibros) y para un LoRA de expresividad. Solo
después de que I1 diga qué atributos faltan en el corpus real. Riesgo: heredar también los artefactos
del 1.5B; los jueces lo verían.

---

## 6. Orden, coste y lo que puede salir mal

| Semana | Qué | Máquinas | USD (E) |
|---|---|---|---|
| 1 | I0 instrumentar; I1 sondas; §5.2 semilla de arranque por emoción (barrido, gratis) | Mac, VM | 0 |
| 2 | I2 intervención causal; §5.1 prefijo negativo; §5.3 énfasis por ficha | Mac, VM | 0 |
| 2-3 | I3 composición y PCA; I4 atención; §5.8 sonda de frontera | Mac | 0 |
| 3 | §5.5 sorpresa frente a WER en los 216 segmentos de F1; §5.4 bifurcación en sesiones (diseño y prueba con `ws_fidelidad`) | Mac, VM | 0 |
| 4+ | §5.9 datos del 1.5B y LoRA de expresividad, solo si I1 muestra huecos | g4dn | ≤ 4 |

Todo lo de las semanas 1-3 es CPU. Lo que sale de aquí alimenta el plan de emoción: las direcciones
de I2 son su E2, el énfasis de §5.3 es su E5, y lo que ninguna dirección mueva es lo que justifica el
LoRA con α en el grafo (E4).

Riesgos, todos con precedente medido en el repo:

- **Zona segura estrecha** de los estados (3b): fuerzas pequeñas, WER en cada puerta.
- **Identidad dentro de la dirección**: por eso las sondas restan la media por hablante y ECAPA vigila.
- **La guía amplifica**: cualquier cambio solo en la rama condicional se multiplica por 3.
- **No linealidad**: puede que un atributo no sea lineal en ninguna capa; I1 lo dice antes de gastar.
- **Herramientas de laboratorio frente a producción**: atenciones y parcheo son de torch; a la VM solo
  llega una suma por fotograma o una entrada más en el IR, y siempre con md5 idéntico apagado.
- **El 1.5B no es un atajo**: ni español, ni tiempo real, ni el mismo espacio latente. Es un maestro.
