# Preferencias por recompensa relativa (Diffusion-DPO) · 2026-09-25

**Resultado: NO PASA. Plan cerrado en D3.** D0 pasó su puerta (hay varianza entre semillas), pero el LoRA
entrenado con Diffusion-DPO no cambia nada medible al generar. En la puerta (378 clips, lectores y textos
apartados, juez whisper large-v3), los tramos con WER > 0,15 siguen **iguales: 2,65 % → 2,65 %**, cuando
se exigía −25 %. Todo lo demás queda dentro del ruido. No se hace la ronda int4 ni D4.

Plan: Obsidian `01-proyectos/06-vibevoicenix/plan-preferencias-dpo-2026-09-24.md`. Rama `dpo-preferencias`.
Datos, LoRA y audios en el NAS: `vibevoice-historial/2026-09-25_dpo-preferencias/` (http://192.168.2.65:8090),
con una carpeta por fase (`d0-piloto`, `d1-datos`, `d2-entrenamiento`, `d3-puerta`). Copia en el Mac en
`~/Documents/mejora-modelo/dpo/`.

## Montaje

- **Generación como en producción**: CFG 3, freno de guía 0,75, 6 pasos, texto con `normalizar_para_motor`.
  Se guardan los **latentes que salen de la cabeza** (`dpo_generar.py` envuelve `sample_speech_tokens`).
  Decodificados, reproducen el audio con correlación 1,0.
- **Voces**: lectores de LibriTTS-R (en) y CML-TTS (es, fr, de, it, pt). Por lector, una referencia (el
  prefijo) y 3 clips reales para ECAPA.
- **Textos**: lectura (frases reales de otros lectores), difíciles (números, siglas, nombres, frases
  largas, preguntas y habla conversacional) y tramos de dobla que fallaron el QC (solo el texto; solo en
  inglés). Entrenamiento y puerta se separan por hash del texto, y la puerta usa lectores distintos (`--saltar 5`).
- **Recompensa** (`dpo_puntuar.py`, whisper **medium**): −WER + ECAPA + 0,5·UTMOS − penalización de
  duración, normalizada dentro del grupo (misma voz y texto, 4 semillas).
- **Par con margen**: WER ≥ 0,05 mejor, ECAPA ≥ 0,03 mejor, o el perdedor es una catástrofe (WER > 0,5,
  corte con WER > 0,15, repetición o tope de fotogramas).
- **Pérdida** (`dpo_entrenar.py`): Diffusion-DPO en v-predicción sobre `forzado.estados`, con el mismo t y
  el mismo ruido para el modelo y la referencia. La referencia es el LoRA apagado, con su condición
  calculada una vez. Se añade un ancla SFT con λ 0,1. `forzado.disposicion` acepta ahora secuencias
  cortadas por el fin (`cortar=True`), que es lo que genera un corte.
- **LoRA**: rango 16, alfa 32, en los dos LM, lr 5e-5, 600 pasos, acumulación 4. La cabeza, congelada.
- **Puerta** (`dpo_puerta.py`, fijada antes de medir): whisper **large-v3**, jueces distintos de la recompensa.

## D0 · ¿Hay señal? (50 textos × 4 semillas, 10 lectores)

| Medida (whisper medium) | Valor |
|---|---|
| **Grupos con par útil** | **70 %** [56; 82] → **pasa** (≥ 30 %) |
| … por WER / por ECAPA / por catástrofe | 16 % / 60 % / 8 % |
| Grupos con el mismo WER en las 4 semillas | 62 % |
| Rango de WER dentro del grupo | 0,114 [0,031; 0,231] |
| Tramos con WER > 0,15 | 20 % (en 2 %, es 10 %, fr 60 %, it 50 %, pt 30 %, de 25 %) |
| Catástrofes | 3,0 % |

Dos errores del juez corregidos antes de D1:
- **Cifras en pt**: el modelo dijo «dois quilos e meio» y whisper escribió «2,5kg». El normalizador solo
  cubre es/en, así que contaba WER 0,53 sin ser un error del modelo. Ya no hay cifras en fr/de/it/pt.
- **Falso corte**: un clip con WER 0 y hablado rápido (razón 0,6) se marcaba como corte. Ahora un corte
  exige también WER > 0,15.

**Calibración de β** (60 pasos sobre los pares de D0): |Δ| crece de 1,6·10⁻⁴ a 4,6·10⁻³, y con β 2000
llega a β|Δ| ≈ 9. La rejilla del plan, {500, 2000, 5000}, queda en β|Δ| ≈ 0,5-25. 2,4-3 s por paso.

## D1 · Datos (300 textos × 4 semillas, 20 lectores)

1200 muestras, 78 % de grupos con par útil [73; 83] y 649 pares (219 por WER, 418 por ECAPA, 12 por
catástrofe). Con whisper medium: tramos con WER > 0,15 al 8,9 % (fr 18 %, it 16 %, pt 12 %, en 3 %) y
catástrofes al 0,3 %. El 56 % de los grupos da el mismo WER en las 4 semillas.

## D2 · Entrenamiento (673 pares de D0+D1; validación: 71 pares de 2 lectores apartados)

| β | Acierto en entrenamiento (paso 600) | Mejor acierto en validación (base 0,5) | Al final (paso 600) |
|---|---|---|---|
| 500 | 0,97 | 0,563 (paso 500) | 0,493 |
| **2000** | 0,97 | **0,620** (paso 450) | 0,549 |
| 5000 | 0,97 | 0,592 (paso 50) | 0,451 |

**Memoriza los pares y no generaliza**: acierta el 97 % de los pares de entrenamiento, pero en lectores
nuevos oscila en torno al azar (±0,06 de error típico con 71 pares). Se pasa a la puerta el mejor, β 2000
en su paso 450.

Dos entrenamientos a la vez **no caben en la T4**: cada uno llega a ~8 GB por la caché del asignador y las
condiciones de referencia guardadas, y el segundo murió por memoria de CUDA. Uno cada vez, junto con una
generación, sí cabe.

## D3 · Puerta (14 lectores apartados × 9 textos de la puerta × semillas 11, 22, 33 = 378 clips pareados)

| Condición (fijada antes de medir) | Base | LoRA β 2000 | |
|---|---|---|---|
| **Tramos con WER > 0,15** (−25 % relativo) | 2,65 % (10) | 2,65 % (10) | **✗ 0 %** |
| WER medio | 0,0234 | 0,0227 | ✓ −0,0007 [−0,005; +0,003] |
| Catástrofes | 1 | 1 | ✓ |
| Identidad ECAPA (≥ −0,005) | | −0,001 [−0,004; +0,002] | ✓ |
| UTMOS (IC inferior ≥ −0,02) | | +0,012 [−0,010; +0,033] | ✓ |
| Duración (±5 %) | | +1,9 % | ✓ |
| Estilo, Carlos (10 segmentos reales × 2 semillas) | 1,034 | 1,031 | ✓ −0,002 [−0,087; +0,083] |

Por idioma, los tramos con WER > 0,15 se mueven pero dentro del ruido: fr 0 → 7,4 % (4 clips), de 3,7 → 1,9 %,
es 4,9 → 3,7 %, pt 1,9 → 0 %, en 1,2 → 0 %.

## Por qué no funciona (lectura)

1. **Con el juez de la puerta, casi no hay nada que corregir.** La base falla el 2,65 % de los tramos con
   large-v3, frente al 8,9 % que marcaba medium en D1 sobre textos parecidos. Buena parte de la señal que
   DPO veía eran errores del juez de recompensa (sobre todo en fr, it y pt), no del modelo.
2. **La diferencia entre semillas no es una propiedad aprendible de la condición.** La semilla solo cambia
   el ruido de la difusión, que la condición de los LM no conoce, y el LoRA solo toca los LM. Por eso
   memoriza los pares (acierto 0,97) y en lectores nuevos se queda en el azar (0,45-0,62).
3. **Los pares son sobre todo de identidad** (60-65 % por ECAPA ≥ 0,03). La identidad de un clon varía
   ~0,056 entre semillas, pero el plan no busca moverla (la limita la grabación) y la puerta confirma que no se mueve.

## Qué queda

- **El reintento de dobla sigue siendo la palanca**: elige entre salidas en inferencia, justo lo que el
  entrenamiento no consiguió meter en los pesos (F2: −27 % de WER con +21 % de síntesis).
- Si se retoma, primero se mide la señal con el **juez de la puerta** (large-v3) en D0, no con medium; y se
  entrenan solo pares por WER o catástrofe, que en D1 fueron 231 de 649. Con una base al 2,65 % no se esperan
  más de unos pocos clips de mejora posible por cada 378.

## Coste

g4dn.xlarge 5,12 h = **2,74 USD**. Incluye ~50 min parada por un corte de la sesión (≈ 0,45 USD).
Plan de mejora: **12,48 de 20 USD**.

## Código

`scripts/lora/dpo_generar.py`, `dpo_puntuar.py`, `dpo_entrenar.py`, `dpo_puerta.py`, `dpo_textos.json`,
`dpo_generar_lote.sh`, `dpo_d0.sh` y `dpo_d2.sh`. También `forzado.py` (`cortar`) y `juez_lote.py`
(`--whisper`, `--rapido`).
