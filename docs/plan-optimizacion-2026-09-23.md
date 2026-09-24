# Qué optimizar ahora y si hace falta entrenar (23-09-2026)

Análisis de solo lectura hecho por un agente (Fable) sobre toda la documentación, los bancos y la memoria del proyecto, tras cerrar F7.

Solo lectura. **(M)** medido, **(E)** estimado. Fuentes abreviadas: `opt` = `docs/optimizacion.md`, `pr` = `docs/plan-rendimiento.md`, `pm` = `docs/plan-mejora-modelo.md`, `f1..f7` = `docs/bancos/2026-09-23-f*.md`, `mem/*` = memoria del proyecto, `dobla/README` = `/Users/juanbedoya/Documents/GitHub/dobla/README.md`, `fase1/README` = `/Users/juanbedoya/Documents/GitHub/dobla-fase1/README.md`.

## 1. Dónde se va hoy el tiempo, el dinero y la calidad

### Motor (VM voz, i7-8700T, OpenVINO int4/int8, 6 pasos)

| Qué | Cifra (M) | Fuente |
|---|---|---|
| Fotograma de 133 ms: LM TTS ~48 ms (2 pasadas: condicional + negativa del CFG), cabeza ~16 ms (6 × 2,6), decodificador ~37 ms, resto ~6,9 ms ≈ 108 ms | RTF 0,885 mediana banco 238 clips; 0,942 en producción tras la fase 1 | `pr` L481-515; `mem/decodificador-subidas-matmul` |
| 88 % del tiempo dentro de `infer()` (LM 38,9 %, decodificador 35,2 %, cabeza 13,9 %); no hay punto caliente en Python | — | `opt` tabla "NO funcionó" |
| Manda el cómputo, no la memoria (pasada de 2 tokens = 1,77× la de 1; decodificador int8→int4 solo −7 %) | PL1 35 W, 6 núcleos AVX2 sin VNNI | `opt` "Qué queda sobre la mesa" |
| El LM se encarece con el contexto: 13,7 → 18-21 ms/paso entre 400 y 1400 tokens; prefijo de 137 → 588 posiciones cuadruplica la KV | — | `opt`; `docs/clonado-de-voz.md` L538-541 |
| Memoria: VmHWM 1,87-2,0 GB; arranque 13 s | — | `pr` L503-515 |
| Clonado: 1,4 s/voz en caliente (M4), 0,05 s por s de audio; carga del codificador ~12-40 s | — | `mem/metricas-demo-clonado-vivo`; `fase1/README` |

**Callejones ya cerrados con puerta (no reproponer):** hilos/anclaje, cfg bajo, `torch.compile`, KV preasignada, cabeza fp32/int4, decodificador int4, LM int8 (C3), 8/10 pasos, Karras, `neg_cada > 1` (−2,6 % RTF por +2,5 pts WER), atención fusionada (no hay nodo SDPA en AVX2), cuantización dinámica/KV u8 (no aplican en esta CPU), B1 topología SMT, B2 pasadas en paralelo, C1/C2 iGPU, solapado del decodificador en OpenVINO, C5 LM de texto en OV (≤ 1,5 %), C4 ventana deslizante (riesgo puro), semillas/cfg/pasos/arranque para identidad (banco de reconstrucción, 22 tandas). (`opt`, `pr` L548-570, `mem/banco-reconstruccion`).

### Servicio en streaming

Primer sonido 0,20 s en la VM (`docs/rendimiento.md`), 0,52-0,65 s en M4 con clon de 168 posiciones frente a 0,9-1,0 s con la voz oficial de prefijo largo (RTF 0,70-0,81 frente a 1,12-1,14): **un prefijo corto genera más rápido** (M, `mem/metricas-demo-clonado-vivo`). Un candado, una locución a la vez. Semilla 101 fija; ruido de arranque 1; rampa cfg 4,5 en 6 fotogramas (`voz_stream.py` L897-970). El normalizador de texto **no está en el servidor** (0 apariciones de `normalizar_texto`/`num2words` en `voz_stream.py`).

### Calidad (los límites reales)

- WER: clones buenos 0,3-2,9 %; base multilingüe 4,4 % (es 1,4 %, en 2,0 %) (`f7`). La semilla mueve 40 puntos, los pasos 1 (`opt`). Dos frases fallan con casi cualquier semilla (`docs/plan-determinismo-calidad.md` L320-323).
- Identidad: techo del códec 0,742-0,835; los clones rinden 60-80 % de ese techo; ningún parámetro lo mueve; manda la grabación y la referencia (`mem/banco-reconstruccion`, `mem/techo-de-voz`). Al cambiar de idioma cae 0,1-0,2 (`pm` §1.2).
- Naturalidad: clones UTMOS 1,9-3,1 frente a 3,0-3,6 de serie; falta ritmo/pausas/dinámica, no tono (`mem/personalidad-voz`).

### Doblaje (dobla, m8a.xlarge)

| Qué | Cifra | Fuente |
|---|---|---|
| Vídeo de 74 min: QC 47 %, síntesis 33 %, cobertura 9,5 %, clonado 3 %; 0,96 h de máquina/h de vídeo; ~0,10 USD spot | M | `dobla/README` L565-577 |
| QC: whisper small residente 2,01 s/segmento; ECAPA 0,08 s; 208 re-tiradas = 1.268 s; **57 de 127 segmentos siguieron fallando tras 2 re-tiradas** | M | `dobla/docs/qc-reduccion.md` |
| Tras dropout=corte + WER normalizado: QC 134,5 → 56,8 s en el vídeo de 2 min; QC −30 % (E) en el de 74 | M/E | ídem |
| bf16 RTF 0,50 frente a f32 0,64 (c8a); bf16 no pasa identidad (−0,011) → opción, no defecto | M | `dobla/README` L582-600 |
| whisper literal `medium` 2,66× `small`; demucs por bloques de 10 min | M | `doblar_video.py` L336-350, L2521 |
| Demo 91 s: en voz propia 4,5 min / 0,009 USD; en nativo (kNN-VC) 10,1 min / 0,017 USD; it 17,9 min | M | `fase1/README` "Demo multilingüe" |
| kNN-VC RTF 0,095 en M4; i7 "2-3× más lento" sin medir; WavLM-large 1,3 GB | M/E | `f4`; `pipeline/acento.py` |
| Cada chunk re-elige referencias (1-2 min); QC y cobertura fuera del checkpoint | M | `dobla/README` L676-690 |

## 2. Optimizaciones sin entrenamiento, priorizadas

Orden por (ganancia × probabilidad) / esfuerzo.

| # | Palanca | Ganancia | Riesgo | Cómo medir / puerta |
|---|---|---|---|---|
| 1 | **Prefijo corto repetido por defecto en voz-stream y en dobla** (10-12 s × 2 en vez de 30 s; hoy dobla solo repite si < 10 s, `REPETIR_BAJO_S`) | Calidad **M**: Liliana 5 s×2 = 30 s en identidad y +0,22 UTMOS; Carlos codo ≤ 12 s (`f1`). Velocidad **E** −5-10 % de RTF y menos KV (el LM crece con el contexto, M) | Bajo: F1 ya lo midió en dos hablantes; falta en las voces de serie (sus prefijos son sintéticos y ahí la curva pide 23 s) | `banco_ab.py` con los clones refabricados (12 s×2) frente a producción: identidad ±0,005 por clon, UTMOS IC inf ≥ −0,02, WER IC sup ≤ +0,5, RTF ≤ base. Refabricar solo clones, no las 61 voces oficiales |
| 2 | **Normalizador en `voz_stream.py`** (campo `normalizar`, apagado por defecto; empaquetar `num2words` en el env de Nix o escribirlo sin dependencias) | WER real −4 pts; inglés 13 → 6 % (M, `f2`) | Bajo; en español "en observación" (un clip peor) | `fidelidad.py` sobre `corpus_mejora.json` grupos `*_numeros`; md5 idéntico con el campo apagado (`ws_fidelidad.py`) |
| 3 | **Re-tirada dirigida en el QC de dobla**: no re-tirar por identidad cuando el techo de la voz < 0,70 (ya lo avisa el log) ni por WER cuando el original es ininteligible (whisper literal con baja confianza) | 57/127 segmentos no se arreglan nunca (M): **E** −20-25 % del QC, que es el 47 % del job | Bajo (se sigue juzgando; solo se deja de re-sintetizar lo inarreglable) | Tramo de 74 min con `pruebas/tramos.py` + `batch_codigo.sh`: WER e identidad medianos iguales (±0,01) con −X s de QC; `calibracion.py` cuando haya valoraciones humanas |
| 4 | **Amortizar el coste fijo del acento nativo** (voz de serie en la imagen en vez de S3, banco de timbre y WavLM una sola vez por job ya están; medir el reparto real de `manifiesto.json`) | La demo pasó de 4,5 a 10,1 min por 91 s (M): casi todo es fijo. **E** −3-5 min por job corto | Nulo en calidad | Reparto por etapa del manifiesto en el mismo vídeo con/sin `--acento nativo` |
| 5 | **Referencias del `preparar` heredadas por los chunks** y clonado reutilizado por identidad conocida (`voces/<id>/`) | 1-2 min por chunk (M) y ~37 s de clonado en el vídeo de 2 min (M) | Bajo | Manifiesto: `clonar` y `referencias` por chunk; identidad QC igual |
| 6 | **Modo "menos coste" bf16 explícito en la web** (ya implementado como opción) | RTF 0,50 frente a 0,64 (M); ×5 en job real | Identidad −0,011 (no pasa): decisión de producto, no técnica | Ya medido (banco v2); solo falta decidir |
| 7 | Graviton c8g.xlarge para dobla | ~1 USD/M (E) frente a 0,96 medido en c8a f32: **la ganancia ya se cobró con c8a/m8a** | fp16 y cuantizados en simulación en ARM | 4 h, 0,32 USD; baja prioridad |

Lo que el plan de rendimiento dejó como "reabrible solo en AVX-512/AMX" (KV u8, cuantización dinámica de activaciones) aplica a dobla en m8a, no a la VM; pasa por `banco_ab.py` (`docs/ec2-y-coste.md` §8).

## 3. ¿Hace falta entrenamiento?

Lección de F7: el bucle funciona, la pérdida baja un 3 %, y generando **no baja el WER y cuesta −0,01 de identidad** (`f7`); la base ya lee al 4,4 %. Lo que queda por calidad (identidad, ritmo) lo pone la grabación, no los pesos. Por tanto, **la única razón honesta para entrenar hoy es velocidad**, y solo una opción tiene tamaño suficiente.

| Opción | Qué ganaría | Datos / GPU / USD (E) | Riesgo | Puerta | Veredicto |
|---|---|---|---|---|---|
| **Destilación de la guía (CFG) en la cabeza**: entrenar la cabeza (o un LoRA suyo) para que con solo la condición positiva prediga la `eps` guiada (cfg 3,0 fijo, o cfg como entrada por la rampa de 4,5); se suprime la pasada negativa del LM | LM 48 → ~24 ms/fotograma: **E −20 % de RTF** (108 → ~85 ms) y menos KV (sin rama negativa). Es la única palanca de dos dígitos que queda | Latentes de `datos.py` (ya existen); maestro = el propio modelo forzado (sin exposure bias en el maestro); 4-8 h g4dn ≈ 2-4 USD + 3 h de `banco_ab` | Medio-alto: el CFG es lo que bajó el WER de 13,6 a 3,6 %; el freno de guía y el ruido de arranque están escritos sobre las dos ramas; nunca medido aquí | (1) md5 intacto con la cabeza original; (2) `fidelidad.py` 18 semillas × 12 frases WER ≤ +0,5; (3) `banco_ab.py` 238: UTMOS IC inf ≥ −0,02, identidad ±0,005, tono ±0,03 st; (4) RTF VM ≤ 0,85× base. **Abandono** a la 2.ª corrida sin pasar (2) | **La única que recomiendo**, tope 6 USD, y después del #1 de la tabla anterior (que es gratis y puede comerse parte de la ganancia) |
| Destilación de pasos de la cabeza (6 → 2-3) | Cabeza 16 → 5-8 ms: **E −7-10 % RTF** máximo (Amdahl) | Igual que arriba, 3-5 h, 2-3 USD | Medio: 8 y 10 pasos ya no pasaron por calidad; 4 nunca se juzgó | `banco_ab.py` | Solo si la anterior pasa y se quiere apurar; sola no compensa |
| Poda o destilación de capas del `tts_lm` (20 capas) | ~2,4 ms por capa y pasada: −4 capas ≈ −10 % RTF (E) | Horas de audio + destilación, 10-20 h GPU, 5-10 USD | Alto: invalida los 61 `.pt` y todos los clones, refabricar y recuantizar; F7 mueve identidad con un LoRA, una poda más | `banco_ab` + `evaluar_clones` | **No** |
| Entrenamiento consciente de cuantización | Recuperar el decodificador int4 (UTMOS −0,043 por −2 % RTF) o la cabeza int4 | 2-4 h GPU | Bajo-medio | `banco_ab` | **No**: 2 % no paga el banco de 3 h; LM int4 ya pasa |
| LoRA con pérdida de identidad o pares es/en del mismo lector | Acento nativo sin perder identidad (F7: UTMOS +0,2-0,5, PER −0,02-0,05, identidad −0,08 en Juan) | No hay pares cruzados en CML-TTS; pérdida ECAPA exige decodificar audio en el bucle (caro) | Alto | Puertas de F7 | **No ahora**: kNN-VC ya da 96-109 % de identidad; reabrir solo si kNN-VC no cabe en CPU |
| LoRA por voz (F8, Carlos, LM y no cabeza) | Ritmo y pausas del propio hablante | 8,8-11,6 min de Carlos; ~1 h g4dn, ~1 USD; juez de personalidad perdido (rehacer, `f0`) | Alto: 3a/3b fallaron; exposure bias | Puerta de la 3b (juez IC>0, 8 descriptores, ECAPA ≥ −0,005, UTMOS ≥ −0,02, WER ≤ +0,5) | Pendiente de decisión humana; solo es producto con ≥ 10 min por persona |
| Afinar el clasificador de fin | Un clip fr de 340 s en la base; c1 con `--fin` lo arregla | Gratis dentro de otra corrida | Bajo | `banco_ab` final ±20 ms | Solo acompañando a otra corrida; solo no |
| Nada | 0 | 0 | 0 | — | Opción válida para calidad: todo lo que la mueve está fuera de los pesos |

**Recomendación:** (a) no entrenar por calidad; (b) un único experimento acotado de destilación de la guía cuando el prefijo corto esté medido; (c) no tocar capas, cuantización, LoRA multilingüe ni token de acento. Presupuesto: quedan 16,31 USD del tope (3,69 gastados, `~/Documents/mejora-modelo/aws_gasto.json`).

## 4. Qué falta respecto a lo que pidió Juan

| Petición | Estado | Bloqueo |
|---|---|---|
| WER mínimo | Hecho en dobla (normalizador en inglés, `wer_norm`, umbral 0,15, siglas). **Pendiente de implementar** en voz-stream (#2). Medido y descartado: LoRA F7, pasos, cfg, `neg_cada` | `num2words` en el entorno de la VM; decisión de push e imagen de dobla |
| Capa de personalidad | Medido y descartado: FiLM (3a/3b), referencia por estilo (F1), WSOLA/PSOLA; hecho: `forma` (apagado por defecto). **Pendiente de decisión**: F8 LoRA por voz | Juez de personalidad perdido (rehacer ~30 min VM taller); datos ≥ 10 min solo de Carlos |
| Clonar con menos audio | Hecho (repetir ×2, en dobla bajo 10 s). **Pendiente**: hacerlo el defecto y llevarlo a `clonar_voz.py`/voz-stream (#1) | Banco con clones refabricados |
| Acento nativo al cambiar de idioma | Hecho en dobla (`--acento nativo`, kNN-VC). Pendientes: RTF en el i7/m8a sin medir; identidad cae con < 1,5 min de audio; PER no llega al nativo | Muestra pequeña (4 frases); coste fijo por job (#4) |
| Ruido ambiental / sonidos / canto | Medido y descartado: marcas en el texto (F5), ambientes procedurales (F6), generación por el modelo. **Pendiente de decisión humana**: librería CC0 con escucha; banco de no-verbales reales por montaje (vía 3, sin implementar) | Elección de librería y consentimiento; canto no está en ningún plan |
| Eficiente | Hecho: RAM −54 %, disco, c8a/m8a, QC −30 %, reanudable. Pendiente: #1, #3, #4, #5; bf16 es decisión de producto | — |
| Llevar a producción el plan | Ramas `mejora-modelo` y `mejoras-modelo` sin push ni imagen | Decisión de Juan |

## 5. ¿Está listo el LoRA?

**El bucle, sí.** `scripts/lora/` tiene forzado con coseno 1,000000 frente a `generate()` (puerta 0), LoRA a cero con md5 idéntico (puerta 1), datos CML-TTS/LibriTTS-R, entrenamiento con validación por lector, evaluación generando y comparación pareada con IC; las incidencias (carrera en `evaluacion.json`, clips desbocados en el juez) están corregidas. Límites conocidos: dos procesos grandes por T4, `datos.py` de pt se muere a 9,6 GB de RAM, y no cubre `--cabeza` en producción (los `.pt` se invalidan y hay que reconvertir IR y refabricar voces).

**El modelo resultante, no.** Las tres corridas (c1, c1p500, c2, en `~/Documents/mejora-modelo/lora/gpu/`) no pasan la puerta: WER +3 a +11 % (IC cruza 0), identidad −0,008 a −0,012. No hay nada fundido, ni IR, ni nada desplegable; los pesos solo sirven como punto de partida si algún día se reabre el acento por pesos.

### Critical Files for Implementation
- /Users/juanbedoya/Documents/GitHub/VibeVoiceNix/.claude/worktrees/jolly-moser-5946b8/pkgs/vibevoice-cli/voz_stream.py (normalizador, prefijo, rampa/CFG, candado)
- /Users/juanbedoya/Documents/GitHub/VibeVoiceNix/.claude/worktrees/jolly-moser-5946b8/scripts/lora/forzado.py (maestro para la destilación de la guía; pérdida v-prediction y rama negativa)
- /Users/juanbedoya/Documents/GitHub/VibeVoiceNix/.claude/worktrees/jolly-moser-5946b8/scripts/banco_ab.py (la puerta de todo cambio de motor)
- /Users/juanbedoya/Documents/GitHub/dobla-fase1/pipeline/doblar_video.py (QC, re-tiradas, `REPETIR_BAJO_S`, acento)
- /Users/juanbedoya/Documents/GitHub/dobla-fase1/pipeline/acento.py (coste fijo del kNN-VC)