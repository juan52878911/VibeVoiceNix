# Emociones graduables por niveles, con direcciones de pares reales (2026-09-27)

Ejecución de la vía E2 del [plan de emoción](../plan-emocion-intencion-2026-09-26.md) después de que la campaña de
[la red por dentro](2026-09-27-red-interna-mac.md) cerrase las direcciones «de regresión» con fuerzas pequeñas.
Etiquetas: **(M)** medido, **(E)** estimado, **(S)** supuesto. Código: [`scripts/red/emociones.py`](../../scripts/red/emociones.py)
y [`scripts/red/emociones_gpu.sh`](../../scripts/red/emociones_gpu.sh). Audio fuera del repo, en el prefijo `emo/`
del bucket de dobla.

> **Resultado:** hay emociones graduables por niveles sin tocar pesos ni añadir coste por fotograma. Alegría,
> enojo, tristeza, asco, miedo y calma se mueven de forma monótona con el nivel, con el WER intacto dentro de su
> zona segura. Tienen dos límites. El juez categórico de emoción no separa bien las emociones sobre esta voz.
> Y las emociones que suben la activación (enojo, alegría, miedo) cuestan naturalidad: UTMOS −0,4 a −1,5 en los
> niveles altos.

## Qué cambia frente a los barridos que no movieron nada

| | Barridos de la red (27-09) | Aquí |
|---|---|---|
| Dirección | vector de una regresión (ridge) sobre la prosodia, normalizado | **diferencia de medias**: el mismo actor diciendo la misma frase con una emoción y en neutro (CREMA-D, 91 actores, 996 pares por emoción) (M) |
| Fuerza | λ · rms(c) · d, con λ ≤ 0,2: un empujón de 0,7 unidades | **nivel · Δ** sin normalizar: nivel 1 = el cambio medio de un actor real (9-21 unidades en la condición) (M) |
| Dónde | condición o residual de la capa 16 | condición y residual de la capa 14, en las dos ramas de la guía, cada una con su propio Δ (rama «natural») |

Con λ ≤ 0,2 se empujaba un **5-7 % de una emoción real** (M). Eso explica que no se moviera nada.

## Las direcciones (M)

Pasada forzada (`forzado.py`) sobre los 7.442 clips de CREMA-D, con una referencia neutra del mismo actor como
prefijo: la media por fotograma de la condición, de la negativa y del residual de las 20 capas en las dos ramas.
19 min en una g4dn.

| Emoción | Norma de Δ en la condición | Proyección de IEO (intensidad baja · media · alta del actor) |
|---|---|---|
| Enojo | 21,1 | 0,65 · 0,98 · 1,61 |
| Alegría | 12,1 | 0,45 · 0,69 · 1,71 |
| Miedo | 9,6 | 0,37 · 0,58 · 1,74 |
| Tristeza | 9,2 | 0,96 · 1,05 · 0,96 |
| Asco | 5,9 | (sin gradación clara) |

- **La red codifica la intensidad a lo largo de la dirección.** La frase que CREMA-D graba en tres intensidades
  cae más lejos cuanto más intensa en enojo, alegría y miedo. Es la base de los niveles.
- **Enojo y alegría comparten casi todo** (coseno 0,92): los dos son sobre todo activación.
- **La voz de base ya suena alegre:** el juez categórico (emotion2vec+) la etiqueta «happy» con 0,82. Por eso la
  dirección de tristeza desde un actor neutro apenas se notaba. Se añadieron **contrastes desde la alegría**:
  `<E>_rel` = Δ(E) − Δ(alegría), y calma = −Δ(alegría).

## Pilotos (M)

2 voces de serie × 2 frases en español × semilla 11. Diferencias pareadas contra el nivel 0 de la misma voz,
frase y semilla. Juez de activación y valencia: audEERING (MSP-Podcast, 0-1; licencia no comercial, **solo como
juez**). Tono = mediana de F0 en semitonos.

**Tres formas de sumar, con las direcciones directas:**

| Forma | Lo que mueve | Coste |
|---|---|---|
| Condición, rama natural | enojo +2 a +3,4 st; alegría +1-2 st | UTMOS −0,8 a −1,5 en enojo; con 1,5 el enojo rompe la voz |
| Condición, solo rama condicional | parecido con la mitad de fuerza (la guía lo amplifica) | igual de caro |
| **Residual de la capa 14, rama natural** | **más tono por unidad de calidad**: alegría +3,6/+4,9 st y miedo +2,9/+5,9 st con nivel 0,5/1 | UTMOS −0,4 a −0,7 |

**Contrastes en la capa 14** (la fila de cada emoción, por nivel):

| Emoción | 0,5 | 1 | 1,5 | 2 |
|---|---|---|---|---|
| Calma | −2,1 st, UTMOS +0,32 | −2,8 st, activ. −0,05, UTMOS +0,43 | −4,5 st, activ. −0,13, UTMOS +0,52 | −5,0 st, activ. −0,20 |
| Tristeza | −1,4 st | −2,9 st, activ. −0,09, valen. −0,06, UTMOS +0,30 | **se rompe** (WER +33) | se rompe |
| Asco | −1,1 st | −2,2 st, −0,94 sílabas/s | −3,6 st, −1,31 sílabas/s, identidad 0,74 | se rompe |
| Enojo | +0,7 st | +1,9 st, activ. +0,08, UTMOS −0,60 | +3,5 st, activ. +0,19, UTMOS −1,46 | +2,8 st, UTMOS −1,78 |
| Miedo (contraste) | no se mueve | no se mueve | no se mueve | no se mueve |

- **Calma, tristeza y asco bajan el tono y la activación y mejoran la naturalidad**: son las más limpias.
- **El contraste de miedo no hace nada** (miedo y alegría son casi la misma dirección desde esta base). Para el
  miedo se usa la dirección directa.
- **El juez categórico no separa las emociones** sobre esta voz: casi todo lo llama «happy», porque la base ya
  lo es. Lo que sí se mueve con el nivel, y de forma monótona, es el tono, la activación y el ritmo.

## Los niveles elegidos

Cuatro niveles por emoción dentro de la zona donde el WER no se mueve. El nivel 0 es la voz sin tocar.

| Emoción | Dirección | Nivel 1 | Nivel 2 | Nivel 3 | Nivel 4 |
|---|---|---|---|---|---|
| Alegría | Δ(alegría) | 0,25 | 0,5 | 0,75 | 1,0 |
| Enojo | Δ(enojo) − Δ(alegría) | 0,4 | 0,8 | 1,2 | 1,6 |
| Tristeza | Δ(tristeza) − Δ(alegría) | 0,25 | 0,5 | 0,75 | 1,0 |
| Miedo | Δ(miedo) | 0,25 | 0,5 | 0,75 | 1,0 |
| Asco | Δ(asco) − Δ(alegría) | 0,3 | 0,6 | 0,9 | 1,2 |
| Calma | −Δ(alegría) | 0,5 | 1,0 | 1,5 | 2,0 |

Todas en el residual de la capa 14, en las dos ramas con su propio Δ.

## Resultado final (M)

4 voces (las dos de serie y los clones de Carlos y Liliana) × 4 frases (3 en español y 1 en inglés) × 6 emociones ×
4 niveles, semilla 11, 400 clips. En español son 12 parejas por celda, pareadas contra el nivel 0.

| Emoción | Tono, niveles 1 → 4 (st) | Activación, nivel 4 | UTMOS, nivel 4 | WER, niveles 1-3 | Rotos en el nivel 4 |
|---|---|---|---|---|---|
| Alegría | +1,7 · +2,3 · +3,2 · +3,8 | +0,15 | −0,80 | +0,0 | 0 de 12 |
| Miedo | +1,3 · +2,2 · +2,9 · +4,0 | +0,10 | −0,53 | ≤ +0,7 | 0 de 12 |
| Enojo | +0,6 · +1,5 · +1,5 · +3,5 | +0,22 | −1,71 | +0,0 | 1 de 12 |
| Tristeza | −0,6 · −1,3 · −2,1 · −2,9 | −0,13 | −0,08 | +0,0 | 1 de 12 |
| Asco | −0,9 · −1,8 · −2,1 · −2,5 | −0,15 | −0,21 | ≤ +0,7 | 2 de 12 |
| Calma | −1,6 · −3,0 · −4,1 · −4,4 | −0,21 | −0,26 | ≤ +1,4 | 1 de 12 |

- **Las seis se gradúan de forma monótona** en tono y activación, en las cuatro voces.
- **En español, los niveles 1 a 3 no mueven el WER.** El nivel 4 rompe 1 o 2 clips de 12 en enojo, tristeza, asco
  y calma. En producción, el nivel 4 pide la re-tirada con otra semilla que ya hace el QC de dobla.
- **En inglés el enojo solo aguanta el nivel 1.** Desde el 2 el modelo deja de hablar y genera sonido (whisper
  oye «BELLS CHIMING»). La tristeza y el asco fallan en un clip inglés cada una.
- **La identidad baja con la intensidad** (ECAPA frente al nivel 0: 0,86-0,89 en el nivel 1 y 0,44-0,77 en el 4).
  Parte es esperable, porque el habla emocional real también se aleja de la neutra; conviene escucharlo con los
  clones.
- **El juez categórico sigue sin servir:** etiqueta casi todo como alegre. La medida útil es el tono, la
  activación, la valencia y la escucha.

Página de escucha con las escaleras (neutro → nivel 4) de las 4 voces y las 4 frases: artifact privado del
26-09. Los 400 clips completos están en el prefijo `emo/gen/final/` del bucket. El NAS (ascci) estaba apagado.

## Coste (M)

g4dn.xlarge bajo demanda: 1,96 h, 1,083 USD (extracción de CREMA-D, dos pilotos, la tanda final y sus jueces). Acumulado del plan: 14,87 de 20 USD. CREMA-D se bajó de GitHub: Hugging Face limitó la IP de la instancia
(429) al pedir 7.442 ficheros sin cuenta.

## Qué falta para producción

1. **Llevarlo al motor de OpenVINO.** Sumar en el residual de la capa 14 pide una entrada más en el IR del LM
   (`convertir_lm_estado.py`): un vector de 896 por fotograma, a coste cero. En la condición bastaban dos líneas
   en `sample_speech_tokens`, pero la capa 14 da más emoción por unidad de calidad.
2. **La puerta del plan de emoción** con las 4 voces y 2 semillas, y un juez de emoción que funcione sobre esta voz.
   Con emotion2vec no se puede, porque etiqueta la base como alegre. Hay que medir activación y valencia contra
   audio real de esas emociones, y escuchar.
3. **La naturalidad de las emociones activas** (enojo, alegría, miedo): el LoRA con la fuerza como mando (E4) es
   la vía para no perder UTMOS.
4. **CREMA-D es ODbL:** vale para sacar direcciones y medir. Antes de vender algo que dependa de ellas hay que
   confirmar que el share-alike no alcanza a unas direcciones derivadas.
