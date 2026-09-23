# F2 y F5 · Texto: números, reintentos y sonidos no verbales (2026-09-23)

Fases 2 y 5 del [plan de mejora](../plan-mejora-modelo.md). Síntesis en la VM voz (producción), juicio en
GPU con [`scripts/juez_lote.py`](../../scripts/juez_lote.py).

## F2a · El juez contaba como error lo que no lo es

Con los números, el WER se inflaba porque whisper escribe «3.30», «$40,000» o «two» donde el texto dice
«3:30», «40,000 dollars» o «2». Ahora el juez da también `wer_norm`: referencia y transcripción pasadas por
el mismo [normalizador](../../scripts/normalizar_texto.py). En los clips con números del corpus, los
errores **reales** están en siglas y códigos: «AWS» → «Albol S», «UA 1284» → «UA-208M», «IB6843» → «IBEX».

## F2b · Normalizar el texto antes de sintetizar

El normalizador escribe con letras números, horas, porcentajes, monedas y ordinales, y deletrea siglas y
códigos («a uve doble ese», «i be seis ocho cuatro tres»). Probado con las 8 frases de números del corpus,
5 voces y las semillas 11 y 101, pareado contra el mismo texto sin normalizar:

| | Crudo | Normalizado | Pareado (IC 95 %) |
|---|---|---|---|
| WER real (`wer_norm`), todo | — | — | **−0,041** [−0,065, −0,017] |
| WER real, inglés | 0,133 | **0,060** | |
| WER real, español | 0,112 | 0,103 | |
| UTMOS | — | — | −0,012 [−0,070, +0,050] |

Dos clips empeoran de ≤ 10 % a > 25 %: la voz inglesa Emma leyendo español (no es un caso de producción) y
el clon de Juan con «doce coma cinco por ciento» («se poma de 5 %»). **Aprobado para inglés; en
observación para español.** Todavía no está en el servidor: el experimento le mandó el texto ya
normalizado desde el Mac. En la VM no está `num2words`, así que integrarlo pide añadirlo al entorno de
Nix o escribir la conversión sin dependencias.

## F2c · Reintentar los clips que salen mal

Simulado con los 216 segmentos en español de la F1, que tienen dos semillas cada uno:

| Estrategia | WER | Síntesis extra |
|---|---|---|
| Una semilla | 0,052 | — |
| Mejor de dos, siempre | 0,034 (**−35 %**) | +100 % |
| Reintentar si WER > 0,10 | 0,038 (**−27 %**) | +21 % |
| Reintentar si WER > 0,20 | 0,043 (−19 %) | +11 % |

La puerta del plan (−20 % con ≤ +15 % de tiempo) se cumple con un umbral en torno a 0,15 (interpolado).
En dobla el control de calidad ya transcribe cada segmento, así que el juez no cuesta nada extra: solo se
paga la resíntesis de los que fallan.

## F5 · Risas, dudas y respiraciones escritas en el texto — no pasa

Tres voces (clon de Juan, sp-Spk3, sp-Spk0), dos semillas, un control sin marca por frase. Juez: AudioSet
AST ([`scripts/juez_sonidos.py`](../../scripts/juez_sonidos.py)).

| Marca | Detectado | WER | UTMOS (control ≈ 3,05-3,37) |
|---|---|---|---|
| «Ja, ja, ja,» | risa 1/6 | 0,22 | 2,55 |
| «¡Jajaja!» | risa 0/6 | 0,35 | 2,13 |
| «(risas)» / «[risa]» | 0/6 (las dice: «Risas, eso sí…») | 0,46 / 0,04 | 3,08 / 3,57 |
| «Uf...» | **respiración 5/6** | 0,08 | **2,58** |
| «[inhala]» / «*suspira*» | 2/6 / 2/6 | 0,56 / 0,44 | 2,78 / 2,74 |
| «Eh...», «Este...», «Mmm...» | se pronuncian | 0,11-0,25 | 2,42-2,87 |

Música inventada: 1 clip de 96 («*suspira*»). **El modelo lee las marcas, no las interpreta.** Solo
«Uf...» produce un soplido fiable, y cuesta −0,47 de naturalidad. Los sonidos no verbales tienen que
venir de fuera del modelo: fragmentos reales de la propia persona insertados en las pausas.
