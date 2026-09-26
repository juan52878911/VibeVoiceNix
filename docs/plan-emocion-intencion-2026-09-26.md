# Plan: emoción e intención en la voz, controlables al generar y en tiempo real (2026-09-26)

Informe de lectura hecho por un agente sobre el repositorio entero (documentos, bancos, código y
historial de git). No se ha medido nada nuevo ni se ha tocado código. Las etiquetas son las de
siempre: **(M)** medido en el proyecto, **(E)** estimado con la cuenta a la vista, **(S)** suposición
sin medir. Fuentes abreviadas: `pm` = [plan-mejora-modelo.md](plan-mejora-modelo.md), `pp` =
[plan-personalidad-voz.md](plan-personalidad-voz.md), `po` =
[plan-optimizacion-2026-09-23.md](plan-optimizacion-2026-09-23.md), `f1..f8` = `docs/bancos/2026-09-2*`.

Lo que pide Juan: **manipular la emoción y la intención de la voz, controlarlas al generar y también
en tiempo real**, y seguir ganando eficiencia y calidad. Este plan dice dónde vive cada cosa en este
modelo, qué palancas quedan abiertas después de todo lo medido, y en qué orden y con qué puertas
probarlas.

Restricciones que manda el proyecto y que se respetan:

- Producción sigue en CPU (i7-8700T, VM de 5 GB): RTF < 1, VmHWM ~1,9-2,0 GB. Un control que cueste
  pasadas extra del backbone no es de tiempo real aquí.
- GPU solo para entrenar (g4dn.xlarge, 0,526 $/h). Gasto del plan de mejora: 9,74 de 20 USD (`pm`).
  Este plan se autoimpone **8 USD**.
- Solo identidades con consentimiento; nada de `.pt`, huellas ni clips en el repo.
- Puerta fijada antes de medir; una puerta que no pasa cierra la palanca.
- Todo control va **apagado por defecto** y, apagado, el audio queda **bit a bit** el de hoy
  (`ws_fidelidad.py` es la puerta de despliegue, como con `forma`).

---

## 1. Qué se ha decidido ya y qué cierra (lectura del historial)

### 1.1 Lo que pasó y está en producción o listo

| Palanca | Resultado | Fuente |
|---|---|---|
| cfg 3,0 con freno de guía 0,75 y rampa de arranque 4,5 en 6 fotogramas | WER 13,6 → 3,6 %; sin deriva de volumen; primera palabra sin masticar | `voz_stream.py` (`frenar_guia`, `CFG_ARRANQUE`) (M) |
| Semilla fija 101 por voz; ruido de arranque 1 | quita la lotería (40 puntos de WER entre semillas) y la música inventada | `plan-determinismo-calidad.md`; `optimizacion.md` (M) |
| OpenVINO int4/int8, decodificador solapado, RAM −54 % | RTF 0,885 mediana; fotograma de 133 ms cuesta ~108 ms: LM ~48 (dos pasadas), cabeza ~16, decodificador ~37, resto ~7 | `plan-rendimiento.md` (M) |
| `forma` (re-durar las pausas con el perfil de la persona) | el único posproceso que pasa: no toca los tramos con voz | `pp` fase 4c (M) |
| Referencia corta repetida ×2 (F1) | identidad +0,02 a +0,05, UTMOS +0,09 a +0,22 | `f1` (M) |
| Normalizador de texto en voz-stream (F2) | WER real −4 puntos, inglés 13 → 6 % | `f2` (M) |
| Conversión de voz kNN-VC solo para cambio de idioma (F4) | identidad 96-109 % del clon directo con minutos de audio real | `f4` (M) |
| **F8, LoRA por voz** en los dos LM, al 75 % de fuerza, mezclado con lectura | identidad +0,049, estilo −0,21, UTMOS +0,08, WER igual; sobrevive a int4; la **fuerza es una palanca continua** | `f8` (M) |

### 1.2 Lo que se midió y se cerró (no repetir)

| Callejón | Por qué cierra | Fuente |
|---|---|---|
| Marcas de texto para risa, duda, suspiro («(risas)», «[inhala]», «Mmm…») | el modelo **lee** las marcas; solo «Uf…» sopla y cuesta −0,47 de UTMOS | `f2/f5` (M) |
| Referencia elegida por estilo (espontánea frente a neutra) | el prefijo **no transfiere el ritmo** de forma medible; no generaliza entre personas | `f1` (M) |
| Mezclar audio de otra voz en el prefijo | identidad −0,17 a −0,48 desde el primer segundo ajeno | `f3` (M) |
| Adaptador FiLM sobre la condición de la difusión, ajustado por gradiente a una voz | acerca la melodía pero WER +15,7, UTMOS −0,34: la condición sale de la zona que la cabeza conoce | `pp` 3b (M) |
| WSOLA (velocidad) y PSOLA (tono) sobre la señal | UTMOS −0,14 a −0,25; ECAPA −0,10 y WER +10 | `pp` 4b; `clonado-de-voz.md` §7.12 (M) |
| Cortar en trozos o `\n` para forzar pausas | inventa, repite la referencia o se come el final | `pp` fase 4 (M) |
| LoRA multilingüe (F7) | WER no baja, identidad −0,01; sí cambia el **estilo** (acento) y lo aplica al clon | `f7` (M) |
| Guía destilada (quitar la rama negativa) | identidad −0,04 en dos corridas; queda apagada como posible modo rápido | `guía` (M) |
| Decodificador destilado a mitad de canales, desde cero, sin discriminador | UTMOS 3,34 → 1,24 | `decodificador` (M) |
| Cambiar de modelo por etiquetas de emoción (Orpheus, Dia, CSM) | inglés y GPU; F5-TTS/Fish no comerciales; Qwen3-TTS pierde identidad cruzada | `pm` §5; `comparativa-motores.md` (M) |

### 1.3 Lo que se fusionó y lo que quedó por decidir

- Ramas del plan de mejora fusionadas en `main` hasta el 25-09 (normalizador, referencia ×2, jueces,
  bucle LoRA completo en `scripts/lora/`, código de la guía destilada apagado).
- **Decisión de Juan (24-09): no se hacen versiones por voz por ahora** (F8): el LoRA fundido obliga a
  un IR del backbone por voz (~150-300 MB) y un backbone por voz en memoria en dobla. Este plan
  vuelve sobre ese bloqueo en §3.4, porque el mecanismo que hace falta para la emoción **lo resuelve
  de paso**.
- Pendientes de decisión: ambientes CC0 por mezcla (F6), banco de no-verbales reales por montaje
  (vía 3 de `pm` §2e), prefijo corto ×2 como defecto en voz-stream (`po` #1).

---

## 2. Dónde vive la emoción en este modelo

No hay token de estilo, de emoción ni de idioma (`pm` §3.1). Lo que decide cómo suena un fotograma:

| Pieza | Qué decide | Coste por fotograma (M) | Dónde se puede tocar sin pasadas extra |
|---|---|---|---|
| `tts_language_model` (20 capas, 896) | ritmo, pausas, énfasis, cuándo parar | ~48 ms (condicional + negativa) | el **residual** de cada capa y el estado de salida, que es la **condición** de la difusión |
| cabeza de difusión (6 pasos, guía cfg 3,0 con freno) | entonación, energía, expresividad | ~16 ms | la condición que recibe (896 números por fotograma), la escala de la guía y el ruido |
| decodificador acústico σ-VAE | timbre, textura; **no crea melodía** | ~37 ms | nada útil (sumidero) |
| el prefijo `.pt` | timbre y, en parte, la forma de hablar | 0 (precalculado) | qué audio y qué transcripción entran |
| el texto | contenido; puntuación | 0 | puntuación, ortografía, normalizador |

Consecuencia: **un control de emoción en tiempo real tiene que actuar sobre la condición (896-d) o
sobre el residual del LM, fotograma a fotograma, sin añadir pasadas.** Eso es exactamente lo que la
fase 3b intentó y rompió el habla, con una diferencia importante: allí el ajuste se **aprendió por
gradiente sobre una sola voz** (movió la condición un 29 % para bajar la pérdida un 2,5 %). Aquí las
direcciones salen de **contrastes reales de la misma persona en dos emociones**, se aplican con una
fuerza acotada y se miden generando, no forzando.

La otra vía, con pesos, ya está validada por F8: un LoRA sobre los dos LM cambia **cómo** habla el
clon manteniendo **quién** es, y su fuerza es un mando continuo (25 % → 100 %) que sobrevive a int4.
Lo que F8 no resolvió es servirlo sin un IR por voz; §3.4 lo resuelve con el LoRA **sin fundir** y la
fuerza como entrada del grafo.

Tiempo real quiere decir aquí: el mando cambia en el **siguiente fotograma** (133 ms). Las sesiones
vivas (`SesionViva`, websocket `/tts/sesion/ws`) ya mantienen una `generate()` abierta y una foto de
su estado (`foto_generacion`); solo hace falta que el estado del control viaje con ella.

---

## 3. Palancas, en orden de valor por coste

### 3.1 E0 · Jueces de emoción e intención, y corpus congelado (Mac, 1 día, 0 USD)

Sin juez no hay puerta. Hoy hay jueces de WER, identidad (ECAPA), naturalidad (UTMOS), pronunciación
(PER), sonidos (AST) y estilo frente a la persona (`juzgar_estilo.py`, 10 descriptores de
`perfil_vocal.py` + contorno de `prosodia.py`). Falta uno de **emoción** y otro de **intención**.

- **Activación y valencia por descriptores** (ya existe casi todo): F0 medio y recorrido, rango de
  energía, sílabas/s, inclinación espectral, HNR. La activación (alegría/enfado frente a calma/tristeza)
  se ve en esos números; la valencia (alegría frente a enfado) mucho peor. Cero dependencias nuevas.
- **Clasificador de emoción (SER)** como segundo juez. Candidatos: `emotion2vec+` (código Apache-2.0;
  comprobar la licencia de los pesos), `speechbrain/emotion-recognition-wav2vec2-IEMOCAP`
  (4 clases, inglés) y el dimensional de audEERING `wav2vec2-large-robust-12-ft-emotion-msp-dim`
  (activación/valencia/dominancia; **CC BY-NC-SA: solo como juez, nunca para pesos**, como se hizo con
  L2-ARCTIC). Un juez entrenado en inglés sobre habla española mide activación razonablemente y
  valencia peor (S): por eso va siempre junto a los descriptores.
- **Intención**: pregunta frente a afirmación por el contorno final (subida en la última sílaba
  tónica, `prosodia.py`); énfasis por palabra con las marcas por palabra de whisper (energía y F0
  de esa palabra frente a sus vecinas). La ironía y la duda no tienen juez barato: **fuera** del plan.
- **Corpus congelado**, añadido a `corpus_mejora.json` sin tocar lo que hay: 8 frases neutras de
  contenido (que admitan cualquier emoción), 4 preguntas, 4 exclamaciones, 4 frases con una palabra
  marcada para énfasis. Emociones objetivo: **neutra, alegre, triste, enfadada, calmada** (las cinco
  que tienen datos apareados; el susurro se queda fuera, ver §5).
- **Calibración con controles reales**: audio emocional real (§3.2) y las grabaciones con
  consentimiento del guion de emociones (§3.5) si llegan a tiempo.

**Puerta:** el juez separa las emociones del audio real con IC 95 % disjuntos al menos en activación
(alegre/enfadada frente a triste/calmada), y sobre clips sintéticos neutros de producción no se mueve
con la semilla (desviación entre 6 semillas menor que un cuarto de la distancia entre emociones).
Si no separa, se cambia de juez antes de E2.

### 3.2 Datos apareados: misma persona, misma frase, dos emociones (Mac + GPU, medio día, ~0,3 USD)

Todo lo que sigue necesita **pares (neutro, emocional) del mismo hablante**, porque es lo que
cancela la identidad y el texto y deja solo la emoción. Fuentes, por orden:

| Fuente | Qué da | Licencia | Uso |
|---|---|---|---|
| **CREMA-D** (91 actores × 12 frases × 6 emociones, inglés, 7.442 clips) | pares perfectos: mismo actor, misma frase | ODbL: **revisar si el share-alike alcanza a los pesos** (misma cautela que con Google Crowdsourced en `pm`) | direcciones (E2) y LoRA (E4) si la licencia lo permite; juez en todo caso |
| ESD (10 hablantes en inglés × 5 emociones × 350 frases paralelas) | pares paralelos | por comprobar (uso de investigación en su página) | juez y calibración; pesos solo si la licencia lo permite |
| RAVDESS, Expresso (Meta), MSP-Podcast | emoción actuada o espontánea | CC BY-NC / académica | **solo juez** |
| **Minado por activación de CML-TTS y LibriTTS-R** (ya en `datos.py`, CC BY 4.0) | segmentos de un mismo lector con activación alta y baja según el juez de E0 | libre | direcciones y LoRA, **la única fuente en español** hoy |
| **Guion de emociones con consentimiento** (§3.5) | pares reales de Carlos, Liliana, Juan | consentimiento expreso | direcciones propias, evaluación |

El español actuado con licencia libre es escaso (MESD, EmoMatchSpanishDB: comprobar licencia y
calidad; son palabras sueltas o pocos hablantes). La apuesta es que la **prosodia de la emoción
cruza el idioma** en este modelo: F7 lo mostró para el estilo de lectura (el LoRA aprendió «cómo
suena un lector nativo» y lo aplicó al clon español) (M). Se mide en E2 antes de gastar en E4.

**La cadena de extracción ya existe:** `forzado.py` da, para audio real con su texto y una
referencia del mismo hablante, la condición de cada fotograma y los residuales del `tts_lm` en una
pasada (puerta 0: coseno 1,000000 con `generate()`). Con la referencia **neutra** del actor y el
objetivo **emocional** sale justo el par que se busca.

### 3.3 E2 · Direcciones de emoción sobre la condición (sin pesos, tiempo real, 0 en RTF)

Cómo encontrar estas direcciones sin corpus de emociones, sondeando la red capa a capa, y qué otros
mandos salen de mirar dentro (rama negativa, arranque, énfasis por ficha, caché bifurcada): [plan de
la red por dentro](plan-red-interna-2026-09-26.md).

**Idea.** Para cada emoción E, la dirección `d_E` = media sobre pares de (condición del fotograma
emocional − condición del fotograma neutro del mismo actor y frase), normalizada. Al generar:

```
c' = c + λ_E · rms(c) · d_E        # una suma por fotograma: cero coste
```

con `λ_E` un mando continuo por emoción, que puede cambiar en cada fotograma. Es la forma más simple
de «dirigir activaciones»; no hay gradiente ni una voz concreta de por medio, así que **no puede
sobreajustarse a Carlos** como la 3b.

Tres variantes que hay que medir, no suponer:

1. **Dónde se suma.** Solo a la rama condicional (la guía cfg 3,0 amplifica la diferencia ×3) o a
   las dos ramas (la guía no la ve). Lo segundo es más suave y probablemente más seguro (S).
2. **En qué sitio.** Sobre la condición (896-d, entrada de la cabeza: mueve entonación y energía) o
   sobre el residual de una capa intermedia del `tts_lm` (mueve también ritmo y fin de frase, y
   entra en la caché KV de los fotogramas siguientes). Lo primero es cambiar dos líneas en
   `sample_speech_tokens`; lo segundo pide una entrada más en el IR del LM (`convertir_lm_estado.py`).
3. **Cuánta fuerza.** Barrido de λ en 0,02-0,20 de la norma de la condición. La 3b rompió el habla
   moviéndola un 29 %: el techo útil estará muy por debajo (S). La puerta la pone el WER.

**Experimento.** Carlos, Liliana, Juan (clones de producción) y dos voces de serie; corpus de E0;
semillas 11 y 101; motor de producción (OpenVINO) a los dos lados. Por emoción: λ en 4 valores.
Además, la prueba de **tiempo real**: locución de 20 s con λ = 0 los primeros 10 s y λ_E después,
frente a la misma locución generada entera con λ_E; se mide la costura (WER y UTMOS en la ventana de
±2 s) y el juez en cada mitad.

**Puerta (fijada antes de medir), por emoción y por voz:** el juez de E0 se mueve en la dirección
pedida con IC > 0 y **monótono en λ**; ECAPA contra el audio real ≥ −0,005; UTMOS con IC inferior
≥ −0,02; WER ≤ +0,5 puntos; ningún clip con WER > 25 % si la base tenía ≤ 10 %; la costura del cambio
en vivo no sube el WER de la ventana más de 1 punto. Pasa la λ máxima que cumpla todo. Si ninguna
emoción pasa en ninguna voz con el juez moviéndose, la vía sin pesos se cierra y se va a E4.

**Coste:** extracción en la g4dn ~30 min (~0,3 USD, compartido con E4); síntesis en la VM ~2 h
(5 voces × 20 frases × 5 emociones × 4 λ × 2 semillas = 4.000 clips: se recorta a 2 λ tras el primer
barrido); juicio en la g4dn ~1 h (~0,5 USD). **Abandono** al primer barrido si el WER sube con
cualquier λ que mueva el juez.

### 3.4 E4 · LoRA de emoción **con la fuerza como entrada del grafo** (con pesos, ≤ 4 USD)

Es la vía robusta, y F8 ya demostró las tres cosas que la sostienen: el LoRA en los dos LM cambia el
estilo sin perder identidad, la fuerza es un mando continuo, y sobrevive a int4 (M).

**Entrenamiento** (mismo `entrenar.py`, mismos datos que §3.2): un LoRA de rango 16 **por emoción**,
con la referencia neutra del actor como prefijo y el enunciado emocional como objetivo, mezclado un
tercio con lectura general (la mezcla es lo que permitió subir la fuerza al 75 % sin WER en F8).
600 pasos, ~14 min de g4dn por LoRA: 5 emociones × 2 corridas ≈ 2,5 h ≈ 1,3 USD. Solo los LM,
nunca la cabeza (F8: la cabeza entrenada cuesta −0,25 de UTMOS).

**Producción: el LoRA no se funde.** Cada `nn.Linear` del `tts_lm` pasa a `W x + α · B A x` con
`α` como **tensor de entrada del IR** (un escalar por emoción, `[n_emociones]`), y `A`, `B` como
constantes del grafo (`convertir_lm_estado.py`). Consecuencias:

- Un solo IR para todas las voces y todas las emociones; `α` cambia por fotograma: **tiempo real**.
- Varias emociones a la vez (α por emoción, suma de LoRAs) y transiciones suaves.
- **Desbloquea F8:** el LoRA de identidad por voz va por el mismo mecanismo, con su `α` por voz, sin
  un IR por voz ni un backbone por voz en dobla. Es la objeción que llevó a la decisión del 24-09.
- Coste: rango 16 × 7 matrices × 20 capas son 140 pares de MatMul pequeños por pasada. En FLOPs es
  despreciable (< 1 %); el riesgo es el **despacho de nodos en OpenVINO/AVX2**, ~2-5 ms por fotograma
  (E). Con `α = 0` los nodos siguen ejecutándose: hay que medirlo antes de que decida nada.

**Puertas, en orden:** (1) md5 idéntico con α = 0 en `ws_fidelidad.py` (la puerta 1 de F7, ahora en
el grafo); (2) RTF en la VM ≤ 1,03 × la base con α = 0 (si no pasa, se buscan solo q/v y rango 8, o se
funden los LoRA en fp32 en memoria al abrir la sesión, que no vale para int4); (3) la misma puerta de
E2 por emoción (juez con IC > 0 y monótono en α, ECAPA ≥ −0,005, UTMOS IC inf ≥ −0,02, WER ≤ +0,5),
con la cuantización de producción simulada como en F8; (4) `banco_ab.py` (238 parejas) con α = 0
frente a producción: UTMOS IC inf ≥ −0,02, identidad ±0,005.

**Abandono:** a la segunda corrida sin pasar (3) en al menos dos emociones. El coste humano manda:
grafo con `α`, refabricar nada (el backbone positivo no cambia; los `.pt` valen).

### 3.5 E3 · Guion de emociones con consentimiento y referencia emocional (Mac + VM, 1 día + grabar)

La palanca más grande medida tres veces es la grabación (`pm` §2f). Pedir a las personas con
consentimiento **30-60 s por emoción** leyendo el mismo guion (neutro, alegre, triste, enfadado,
calmado) da tres cosas a la vez: controles reales para el juez (E0), pares propios para direcciones
personales (E2: `d_E` de esa persona en vez de la media de actores) y un prefijo por emoción.

Sobre el **prefijo emocional**: F1 dice que el prefijo no transfiere el ritmo y F3 que mezclar otra
voz hunde la identidad. Aquí es la misma persona, así que la identidad no debería caer (S), y lo que
se busca es timbre y calidad de voz de la emoción (tensión, brillo), que es justo lo que el prefijo sí
lleva. Es un experimento de una tarde: clon neutro frente a clon con prefijo emocional (F1 ya tiene
`f1_referencias.py`), misma puerta que E2 sin la costura.

Legal: el consentimiento tiene que cubrir **modelar la voz con emociones**; queda fuera del repo, como
todo lo demás.

### 3.6 E1 · Lo que ya controla el texto: puntuación, y las pausas por emoción (VM, medio día, 0 USD)

Barato y primero, porque acota lo que las demás palancas tienen que aportar:

- **Puntuación como intención.** Medir qué hace el modelo con `¿…?`, `¡…!`, `…`, comas, guiones y
  MAYÚSCULAS en una palabra: contorno final (pregunta), energía y F0 en la palabra (énfasis). Los
  marcadores de F5 fallaron porque no están en el corpus; la puntuación sí está. 3 voces × 12 frases
  × 2 semillas. Puerta: el juez de intención se mueve con IC > 0 y WER ≤ +0,5; lo que pase queda
  documentado como «lo que se puede pedir por texto» y entra en la guía del asistente.
- **`forma` por emoción.** Cada emoción lleva su distribución de pausas (excitación: más cortas;
  tristeza y calma: más largas y más frecuentes en la persona real). Ya es un campo por petición
  (`pausas`) y no toca la voz. Se aplica en tiempo real (`ColaAudioSesion`). Puerta: la de la 4c.
- **Normalizador y velocidad** no se tocan: la cabecera de velocidad mueve el tono (±10 % de margen)
  y se queda como está.

### 3.7 E5 · Control local: énfasis e intención por palabra

El mando global (E2/E4) sirve para «esta frase, alegre». El énfasis pide «**esta palabra**». Tres
grados, de barato a caro:

1. **Por puntuación y ortografía** (E1): si «¡…!» o las mayúsculas mueven la palabra, ya está.
2. **Dos pasadas en lote (dobla).** La generación es determinista por semilla: los fotogramas anteriores
   al cambio son idénticos. Se genera, se alinean las palabras con whisper (marcas por palabra, ya en
   `datos_voz.py`), y se vuelve a generar aplicando λ solo en los fotogramas de la palabra. Cuesta una
   síntesis más por segmento marcado; en dobla el QC ya transcribe, así que la alineación es gratis.
3. **En vivo, con retardo calibrado.** El modelo lee el texto en ventanas de 5 fichas por 6 fotogramas
   (`forzado.py`), así que la posición de lectura se conoce; el habla va por detrás «una cantidad
   variable» (`voz_stream.py`, bloque RESPIRO). Se mide ese retardo por voz con whisper (mediana y
   p90 en fotogramas) y se aplica λ en una ventana que cubra el p90. Puerta: el énfasis cae en la
   palabra pedida en ≥ 80 % de los clips (juez de E0) sin subir el WER. Si el retardo es demasiado
   disperso, el énfasis en vivo se queda en «desde aquí en adelante» y el de palabra solo en lote.
   Alinear por la atención del `tts_lm` sobre las fichas de texto sería lo exacto, pero el IR no
   expone atenciones y en AVX2 no hay nodo SDPA que tocar: queda como investigación, no como fase.

### 3.8 E6 · El producto: el mando en la API y en la sesión viva (código, 2-3 días, 0 USD)

Independiente de qué palanca pase; se construye sobre E2 y crece con E4:

- **Estado de control por generación:** `{emocion: {alegre: 0.6, calma: 0.2}, enfasis: [...]}`; viaja
  en `foto_generacion`/`reponer_generacion` para que una sesión parada lo recupere al reanudar.
- **Websocket:** un marco nuevo de control (`MARCO_CONTROL`), aplicado en la frontera del siguiente
  fotograma; acuse con el fotograma en que entró. Un cliente puede mover un deslizador mientras habla.
- **HTTP:** campo `emocion` en `/tts/stream` y en las sesiones; marcas en línea del tipo
  `{alegre}…{/alegre}` que el **servidor** quita antes de tokenizar (el modelo no las ve: F5) y
  traduce a ventanas de fichas.
- **Ficha de la voz** (`<voz>.json`): λ máxima por emoción que pasó la puerta para esa voz (la
  fuerza segura es de la voz, como la semilla), direcciones propias si existen (E3) y pausas por
  emoción (E1).
- **Puerta de despliegue:** `ws_fidelidad.py` con md5 idéntico sin control, y una prueba con control
  encendido que compruebe que el cambio entra en el fotograma acusado.
- **`prueba.html`:** deslizadores por emoción y un botón de énfasis, para oírlo desde el móvil.

---

## 4. Eficiencia: lo que queda con recorrido

El plan de rendimiento y `po` cerraron con puerta casi todo lo barato (hilos, cuantización, pasos,
iGPU, `neg_cada`, guía destilada). Manda el cómputo, no la memoria (M). Lo que queda:

| # | Palanca | Ganancia | Coste | Riesgo | Puerta |
|---|---|---|---|---|---|
| R1 | **Prefijo corto ×2 por defecto** en voz-stream (10-12 s × 2 en vez de 30 s) | calidad (M, `f1`); RTF −5-10 % (E) porque el LM crece con el contexto (13,7 → 18-21 ms de 400 a 1400 fichas, M) y menos KV | refabricar clones; 3 h de `banco_ab` | bajo | identidad ±0,005 por clon, UTMOS IC inf ≥ −0,02, WER IC sup ≤ +0,5, RTF ≤ base |
| R2 | **Decodificador podado desde el maestro con discriminador**: quitar la mitad de los 8 bloques de la etapa 0 (268 de 340 M parámetros), pesos del maestro, ajuste corto con la pérdida espectral **más** un discriminador multiperiodo/multiescala (HiFi-GAN, MIT) | el decodificador son 37 de 108 ms: −12-15 ms por fotograma, **−10-14 % de RTF** (E); la única de dos dígitos que queda | 2-4 h g4dn (~2 USD) | medio: el intento anterior falló por partir de cero y sin discriminador, no por la idea | la del decodificador destilado: WER ≤ +0,5, UTMOS ≥ −0,02, ECAPA ±0,005, RTF ≤ 0,9 × base; abandono a la 2.ª corrida |
| R3 | Destilación de pasos de la cabeza (6 → 3) | cabeza 16 → 8 ms: −7 % (E) | 2-3 h g4dn | medio: 8/10 pasos no pasaron; 4 nunca se juzgó | `banco_ab` |
| R4 | Coste del LoRA en el grafo (E4) medido con α = 0 | 0 o negativo | incluido en E4 | — | RTF ≤ 1,03 × base |

R1 va antes que nada (es gratis y puede comerse parte de lo que ganen las otras). R3 solo si R2 pasa
y se quiere apurar. Un mando de emoción por E2 o por `forma` cuesta 0 en RTF; por E4, lo que diga R4.

---

## 5. Riesgos y lo que no merece la pena

- **Identidad.** Cualquier mando que la mueva más de 0,005 no pasa: la emoción de otra persona no
  sirve. Por eso los pares son del mismo hablante y la puerta lleva ECAPA en todas las fases.
- **Inteligibilidad.** La lección de la 3b: la condición tiene una zona segura estrecha. Fuerzas
  pequeñas, WER en cada puerta, y ningún clip catastrófico.
- **El juez engañado.** El de personalidad subió +0,22 con audio peor (`pp` 3b). El juez de emoción
  va siempre con WER, UTMOS e identidad, nunca solo.
- **Idioma de los datos.** Casi todo lo apareado es inglés. Si E2 muestra que las direcciones no
  cruzan al español, E4 depende del minado de CML-TTS y del guion con consentimiento.
- **Licencias.** NC solo para jueces; ODbL de CREMA-D por revisar antes de entrenar pesos con él.
- **Costura en vivo.** Cambiar α o λ a mitad deja la caché KV con el estado anterior. Se mide la
  costura (E2); si molesta, el cambio se aplica en la siguiente frontera de frase (el respiro ya la
  detecta) en vez de en el siguiente fotograma.
- **Susurro** y otras calidades de voz sin F0: sin datos apareados y probablemente fuera de lo que
  el σ-VAE ha visto (S). Fuera del plan hasta que haya datos.
- **Sobreajuste a Carlos** de nuevo: E3 da direcciones propias, pero la puerta exige que E2/E4 pasen en
  al menos dos voces con consentimiento y una de serie.

**No repetir:** marcas textuales de no-verbales, posproceso sobre la voz, FiLM por gradiente sobre
una voz, prefijo con audio ajeno, más pasadas del backbone en producción, cambiar de modelo.

---

## 6. Orden, calendario y presupuesto

| Semana | Fase | Máquinas | USD (E) |
|---|---|---|---|
| 1 | **E0** jueces y corpus; **E1** puntuación y pausas por emoción; §3.2 extracción de pares; pedir el **guion de emociones** (E3) | Mac, VM, g4dn 30 min | 0,3 |
| 1-2 | **E2** direcciones sobre la condición, con la prueba de tiempo real | Mac, VM, g4dn 1 h | 0,5 |
| 2 | **R1** prefijo corto por defecto (banco de 3 h) | VM, LXC 204 | 0 |
| 2-3 | **E6** el mando en la API y la sesión (sobre lo que pase de E1/E2) | código | 0 |
| 3 | **E3** referencia emocional y direcciones propias, si llegaron las grabaciones | Mac, VM | 0 |
| 3-4 | **E4** LoRA de emoción con α en el grafo (primero R4: coste con α = 0) | g4dn 3-4 h, VM, LXC 204 | 2-3 |
| 4-5 | **R2** decodificador podado con discriminador | g4dn 2-4 h | ~2 |
| — | **E5** énfasis por palabra: 1 y 2 con E6; 3 después de medir el retardo | VM | 0 |

Tope autoimpuesto: **8 USD** de los 10,26 que quedan del plan de mejora. Cada fase se cierra con su
informe en `docs/bancos/` y una fila en la tabla de estado de `pm`.

Lo que decide Juan antes de empezar: (1) si CREMA-D vale para pesos o solo para jueces; (2) pedir el
guion de emociones a las personas con consentimiento; (3) si el mecanismo de LoRA con α en el grafo
reabre las versiones por voz de F8, que quedaron aparcadas por el coste de un IR por voz.
