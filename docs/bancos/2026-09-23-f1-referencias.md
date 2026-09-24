# F1 · Qué referencia y cuántos segundos (2026-09-23)

Fase 1 del [plan de mejora](../plan-mejora-modelo.md). Hablantes con consentimiento: Carlos (119 tramos
disponibles fuera de lo evaluado) y Liliana (21). Clonado en el Mac
([`scripts/f1_referencias.py`](../../scripts/f1_referencias.py)); síntesis en la VM voz (producción
OpenVINO, 576 clips); juicio en una g4dn.xlarge ([`scripts/juez_lote.py`](../../scripts/juez_lote.py)).

Cada variante dice los 12 segmentos evaluados del banco de reconstrucción (en español) y las 4 frases
base en inglés del [corpus](../../scripts/corpus_mejora.json), con las semillas 11 y 101. La identidad
(ECAPA) se mide contra el centroide de sus 12 segmentos reales evaluados, que el clon nunca oyó.
Las comparaciones son **pareadas**: mismo segmento y misma semilla. Tabla completa:
`~/Documents/mejora-modelo/f1/informe.txt` ([`scripts/f1_informe.py`](../../scripts/f1_informe.py)).

## Variantes

| Variante | Qué es |
|---|---|
| `dobla30` | La regla de producción: los tramos más típicos (centroide fijo) hasta 30 s |
| `espontanea30` / `neutra30` | Entre los típicos, los de más y de menos pausas por minuto |
| `c5` `c8` `c12` `c20` | La combinación de tramos típicos más cercana a esos segundos |
| `rep2` `rep3` | Los mismos clips de `c5`, repetidos 2 y 3 veces en el prefijo |

## Resultados (pareados, IC 95 %)

| Comparación | Carlos | Liliana | Veredicto |
|---|---|---|---|
| espontánea − dobla: identidad es | **+0,042** [+0,011, +0,070] | −0,001 [−0,028, +0,025] | **no pasa**: no generaliza |
| espontánea − dobla: identidad en | **+0,121** [+0,045, +0,184] | −0,043 [−0,158, +0,099] | |
| espontánea − dobla: UTMOS | +0,102 [−0,017, +0,226] | **−0,154** [−0,232, −0,078] | |
| espontánea − dobla: distancia al ritmo real | −0,18 [−3,6, +3,0] pausas/min | +2,9 [−0,9, +6,9] | el ritmo **no** se transfiere |
| neutra − dobla: UTMOS | **−0,355** [−0,469, −0,243] | **−0,206** [−0,323, −0,099] | una referencia leída empeora el clon |
| **rep2 − c5: identidad es** | **+0,054** [+0,020, +0,088] | **+0,020** [+0,001, +0,040] | **pasa** |
| rep2 − c5: UTMOS | +0,088 [−0,015, +0,193] | **+0,224** [+0,103, +0,345] | |
| rep3 − c5: WER es | **−0,037** [−0,083, −0,005] | +0,007 [−0,022, +0,037] | |

**Curva de segundos** (identidad en español, media):

| | 5 s | 8 s | 12 s | 20 s (Liliana 17) | 30 s | 5 s ×2 |
|---|---|---|---|---|---|---|
| Carlos | 0,503 | 0,525 | 0,564 | 0,545 | 0,574 | 0,557 |
| Liliana | 0,690 | 0,689 | 0,684 | 0,677 | 0,709 | 0,710 |

## Conclusiones

1. **Repetir el clip corto pasa en los dos hablantes.** Cinco segundos repetidos dos veces en el prefijo
   suben la identidad y la naturalidad sin empeorar el WER. Liliana, con 5 s × 2, iguala sus 30 s en
   identidad (0,710 frente a 0,709) y suena mejor (UTMOS 3,04 frente a 2,82). Carlos se queda a
   0,017 de sus 30 s. Es la palanca para clonar con poco audio.
2. **El codo de la curva está en ≤ 12 s.** A Liliana le bastan 5 s para el 97 % de la identidad de 30 s;
   Carlos llega al 88 % con 5 s y se estabiliza en torno a 12 s.
3. **Elegir la referencia por estilo no es una regla general.** La espontánea gana con Carlos y pierde
   naturalidad con Liliana. El prefijo no transfiere el ritmo de forma medible en ninguno de los dos:
   la hipótesis de la personalidad por referencia queda descartada.
4. **Nunca una referencia leída o sin pausas:** la naturalidad cae 0,2-0,36 en los dos.

5. **El autoarranque no aporta** (5 s reales + ~35 s de habla generada por el propio clon de 5 s, vuelto a
   clonar). Pareado: Carlos +0,046 [+0,009, +0,085] sobre los 5 s, igual que repetirlos (−0,008,
   sin efecto); Liliana **−0,045** [−0,062, −0,028] sobre los 5 s y −0,065 frente a repetirlos.
   Repetir el clip es más simple y gana en los dos: el autoarranque se cierra.
