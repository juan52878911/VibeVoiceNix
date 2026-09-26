# Plan: contenido automático con voces realistas (historias, explicaciones y más) (2026-09-26)

Informe de lectura hecho por un agente sobre el repositorio y sus bancos. No se ha medido nada nuevo.
Etiquetas: **(M)** medido en el proyecto, **(E)** estimado con la cuenta a la vista, **(S)** suposición.

La pregunta de Juan: ¿se puede montar una fábrica de contenido hablado (relatos, explicaciones, episodios)
con lo que hay? Respuesta corta: **sí para narración y explicación con un narrador o varios, hoy mismo,
con las piezas que ya existen; no todavía para relatos con emoción dirigida y personajes que rían o
susurren**, porque eso es justo lo que persiguen los planes de
[emoción e intención](plan-emocion-intencion-2026-09-26.md) y de
[la red por dentro](plan-red-interna-2026-09-26.md). El camino es construir la cadena con lo maduro y
enchufar la emoción cuando pase su puerta.

---

## 1. Lo que ya existe y sirve tal cual

| Pieza | Dónde | Qué aporta a la fábrica |
|---|---|---|
| Guion por LLM en streaming | `scripts/asistente.py` (MiniMax por endpoint compatible con Anthropic, o Ollama local), `scripts/conversacion.py` (salida estructurada JSON con Ollama) | redactar el texto; ya se filtra el `<think>` y el markdown; ya hay salida estructurada |
| Narración de texto que llega a trozos | `scripts/narrador.py`, sesiones y websocket de voz-stream | leer un guion largo sin cortes ni huecos; búfer de 1,5 s medido |
| Motor en producción | voz-stream en la VM: RTF 0,885 (M), primer sonido 0,20 s, una locución a la vez | 1 h de audio ≈ 1 h de VM |
| Determinismo y calidad por defecto | semilla por voz, ruido de arranque 1, cfg 3,0 con freno, rampa de arranque, normalizador de texto, `forma` | WER 0,3-2,9 % en clones buenos (M); sin música inventada; sin deriva de volumen; números y siglas bien leídos |
| Lote con control de calidad y reintento | dobla: QC por segmento (whisper, ECAPA), resíntesis si WER > 0,15, montaje, ambientes por mezcla, ducking, m8a spot | −27 % de WER por +21 % de síntesis (M); 0,96 h de máquina por hora de vídeo, ~0,10 USD (M) |
| Voces | 25 de serie (sp-Spk0/1 en español) y clones con consentimiento (Carlos, Liliana, Juan) | narradores distintos por formato; identidad 60-80 % del techo en clones (M) |
| Jueces | `scripts/juez_lote.py` (WER, ECAPA, UTMOS), `perfil_vocal.py`, `juez_sonidos.py` | la puerta de cada lote sin oír nada |
| Archivo para escuchar | `scripts/historial_audios.py` | web estática con los audios y sus medidas |
| Coste por carácter | `docs/ec2-y-coste.md`: casa ≈ 0,08-0,12 $/M (E); c7i spot 2,7 $/M (M); Polly Neural 16 | la nube es ráfaga y disponibilidad, no coste |

## 2. Qué contenido, con qué dificultad

| Formato | Estado | Qué lo limita | Qué lo desbloquea |
|---|---|---|---|
| **Explicaciones y lecciones con un narrador** (tutoriales, resúmenes, documentación en voz) | **Listo** | frases que fallan con casi cualquier semilla (2 de 12 conocidas, M); lectura plana en clones (UTMOS 1,9-3,1 frente a 3,0-3,6 de serie, M) | reintento con otra semilla y, si falla dos veces, **pedir al LLM que reformule la frase** (barato, no tocado aún); narrar con voces de serie o clones que hayan pasado la puerta |
| **Noticias, boletines, «el estado del homelab»** | Listo | igual | el asistente ya lo hace con rellenos cacheados |
| **Relatos y cuentos con un narrador** | Listo, sin emoción dirigida | la emoción sale del texto y de la puntuación, no de un mando; el ritmo del narrador es el del modelo | E1 (puntuación medida), `forma` con perfil por «tono» (pausas más largas en calma, más cortas en tensión), semilla por personaje; después E2/E4 |
| **Diálogos y podcasts de dos o más voces** | Listo por montaje | cada segmento es una locución; no hay cambio de voz dentro de una (F3: mezclar voces en el prefijo las funde); sin solapes | dobla ya monta varios hablantes con pausas de unión; un guion en JSON con `personaje` por segmento basta |
| **Personajes con emociones marcadas, risas, susurros, suspiros** | **No** | F5: el modelo lee las marcas («(risas)», «[inhala]»); el susurro no tiene datos; la emoción no tiene mando | los no verbales, por montaje de fragmentos reales de la persona (vía 3 de `pm` §2e); la emoción, cuando E2 o E4 pasen |
| **Ambiente, música de fondo, efectos** | Listo por mezcla, pendiente de decisión | el modelo no debe generarlo (coste por fotograma igual al habla, sin control); F6: los ambientes procedurales no convencen | librería CC0 escuchada antes de adoptarla; dobla ya tiene la mezcla, la sala y el ducking |
| **Canto** | No | fuera de todos los planes | otro modelo |
| **Otros idiomas** | inglés sí; fr/de/it/pt con acento y WER alto (M) | prefijo español manda la fonética | kNN-VC solo para cambio de idioma (F4); guion en inglés con voces inglesas de serie |

## 3. La cadena propuesta

```
tema / fuente ──► guion (LLM) ──► JSON de segmentos ──► normalizador ──► síntesis por segmento
                                  {personaje, texto,                      (voz, semilla, ruido de
                                   tono, pausa}                            arranque, forma por tono)
                                                                                 │
publicación ◄── montaje ◄── QC por segmento ◄────────────────────────────────────┘
(ogg/mp3, RSS,  (pausas de unión,   (WER whisper; UTMOS; identidad si es clon;
 vídeo con       ambiente CC0,       reintento con otra semilla; a la 2.ª,
 imagen fija,    ducking, volumen)   reformular con el LLM; a la 3.ª, marcar)
 historial_audios)
```

Decisiones que ya están tomadas por lo medido y que la cadena respeta:

- **Un segmento es una frase o un párrafo corto que cierra frase** (tope de 220 caracteres): los trozos
  cortos sin contexto inventan (M); los saltos de línea disparan el fin de locución (M).
- **La semilla es de la voz y del corpus** (M): cada narrador lleva su semilla buena en su ficha; la
  variedad entre episodios se busca cambiando de narrador o de guion, no sorteando.
- **El reintento paga**: umbral de WER en torno a 0,15, como en dobla (M). Nuevo y barato: la tercera
  vía es reformular la frase con el LLM (sinónimos, partir la frase), porque las frases que fallan lo hacen
  con casi cualquier semilla (M) y son un problema del texto para este modelo.
- **El ambiente y los no verbales van por montaje**, nunca por el modelo (M, F5 y F6).
- **El guion lo redacta el LLM con salida estructurada** (ya se usa con Ollama; MiniMax por el endpoint
  compatible con Anthropic lleva herramientas nativas): una plantilla por formato (lección, relato,
  boletín, diálogo) que devuelve segmentos con personaje y tono. Para explicaciones, el LLM alucina:
  hay que darle la fuente (documento, notas) y revisar el primer lote a mano.

## 4. Coste y tiempo por episodio

Con el candado de una locución a la vez y RTF 0,885 (M):

| Episodio de 10 min de habla (~9.600 caracteres) | En casa (VM) | c8a.2xlarge spot |
|---|---|---|
| Síntesis | ~9 min | ~6 min (E, RTF ~0,5 bf16 medido en dobla; en f32 ~8) |
| QC con whisper `small` y reintentos (+21 %) | ~5 min (E; en dobla el QC es el 47 % del job, M) | ~3 min |
| Montaje y codificación | < 1 min | < 1 min |
| **Total** | **~15 min de VM, ~0,005 USD de luz** | **~10 min, ~0,03 USD** (E) |
| Cien episodios al mes | ~25 h de VM, ~0,5 USD | ~3 USD |

La VM de casa sirve para todo lo que no sea ráfaga; el spot, para lotes grandes o para no ocupar la VM
del asistente. Ningún formato de los de §2 necesita GPU.

## 5. Puertas de calidad (fijadas antes del primer lote)

- **Inteligibilidad:** WER por segmento ≤ 0,10 tras reintentos y ningún segmento > 0,25; el que no pase
  se marca en el archivo y no se publica solo.
- **Naturalidad:** UTMOS medio del episodio ≥ 3,0 con voces de serie; con clones, el que tenga su ficha
  (semilla y ruido de arranque elegidos contra el motor de producción).
- **Cobertura:** palabras transcritas entre 0,85 y 1,15 veces las del guion por segmento (la puerta de la
  fase 4 del plan de personalidad), para que no se coma ni repita nada.
- **Escucha humana del primer episodio de cada formato** con `historial_audios.py`, antes de automatizar
  su publicación. Los jueces no oyen monotonía ni ritmo cansino: eso lo decide una persona la primera vez.
- **Contenido:** para explicaciones, la fuente va en el prompt y el guion se revisa la primera vez.

## 6. Legal y de producto

- Voces de serie: las de Microsoft, con la licencia del modelo (comprobar el uso comercial antes de
  publicar contenido con ánimo de lucro). Clones: **solo con consentimiento expreso para ese uso**, y
  nunca para hacer decir a la persona algo que no aprobó; los `.pt` no salen del NAS ni del repo.
- Marcar el contenido como sintético donde se publique. Nada que imite a una persona real que no sea
  la que dio el consentimiento.
- El coste real del producto no es el cómputo (§4): es el guion (revisión) y el oído (primer lote).

## 7. Fases

| Fase | Qué | Con qué | Puerta | Tiempo |
|---|---|---|---|---|
| C0 · un narrador, una explicación | guion por LLM desde un documento del repo → JSON → `sintetizar_lote.py` → `juez_lote.py` → montaje con pausas de unión | todo existe; falta el guion en JSON y el montador (dobla lo tiene) | las de §5 en un episodio de 5 min; escucha | 1-2 días |
| C1 · reintento y reformulación | reintento por semilla, reformulación por LLM a la segunda, marca a la tercera | dobla (QC) + una llamada al LLM | WER medio del episodio −30 % frente a C0 con ≤ +25 % de síntesis | 1 día |
| C2 · varias voces y tonos por `forma` | personaje por segmento; perfil de pausas por tono (calma, tensión) como campo `pausas` | montaje de dobla; `forma` | cobertura y WER de C0; escucha del diálogo | 2 días |
| C3 · publicación | ogg/mp3, RSS, vídeo con imagen fija por ffmpeg, `historial_audios.py` con portada por episodio | scripts nuevos | un episodio publicado de punta a punta sin tocar nada a mano | 1-2 días |
| C4 · ambiente por mezcla | librería CC0 escuchada; nivel −25 a −35 dB bajo la voz; sala compartida | mezcla de dobla | decisión humana de qué fondos; UTMOS no decide aquí (F6) | 1 día + escucha |
| C5 · emoción dirigida | el mando de E2 o E4 por segmento (`emocion` en la API) | plan de emoción | la puerta de E2/E4 | cuando pase |
| C6 · no verbales reales | banco de risas, respiraciones y «eh» de cada persona, insertados en las pausas | vía 3 de `pm` §2e; AST para detectarlos | ECAPA ≥ −0,005, escucha | después de C5 |

C0-C3 son una semana de trabajo sin GPU y sin tocar el modelo. C5 y C6 son el salto de «leído bien» a
«actuado», y dependen de que las direcciones o el LoRA de emoción pasen su puerta.

## 8. Riesgos y lo que no hay que intentar

- **Monotonía en relatos largos:** el modelo lee; sin mando de emoción, un cuento de 20 min cansa (S).
  Mitigar con guion (frases cortas, preguntas, cambios de narrador) y con `forma`; medir con escucha, no
  con UTMOS.
- **Frases imposibles:** hay frases que este modelo no dice con ninguna semilla (M). La reformulación por
  LLM es la salida; sin ella, el 5-10 % de los segmentos se marca (E).
- **Clones para narrar:** solo si su ficha existe y pasan la puerta; un clon con referencia floja da
  WER catastróficos (M).
- **No intentar:** emoción por marcas en el texto (F5), ambiente generado por el modelo, varias voces en
  un mismo prefijo (F3), cambiar de modelo por los no verbales (inglés, GPU, licencias; `pm` §5), y
  publicar sin la escucha del primer episodio de cada formato.
