# F8 · LoRA por voz: identidad y personalidad de un clon (2026-09-24)

**Resultado: primera palanca del plan que sube la identidad sin coste de inteligibilidad.** Un LoRA
entrenado con ~10 min de la persona, **aplicado al 50 % de su fuerza**:
- acerca el clon a Carlos en timbre (ECAPA +0,03) y en estilo (distancia −0,16 a −0,25 desviaciones suyas);
- no mueve el WER de forma significativa;
- con **5 min** de audio da lo mismo.

Aplicado entero (100 %), la identidad sube más (+0,053), pero el WER pasa de 2,5 % a 9,5 %: el clon
empieza a hablar como Carlos en su charla espontánea, que whisper transcribe con un WER de 0,289.

## Montaje

- **Persona:** Carlos Segura (identidad con consentimiento de dobla).
- **Entrenamiento:** su banco de 11,6 min, cortado en tramos por pausas con marcas por palabra de whisper
  large-v3 (`scripts/lora/datos_voz.py`): 64 ejemplos, 10,8 min.
- **Evaluación: 11 segmentos reales apartados** del banco de reconstrucción, con texto humano.
  - La correlación con el banco es 0,21-0,54; el segmento 0 (0,54) se quitó.
  - El clon dice ese mismo texto, así que el estilo se compara frase a frase con Carlos diciéndola
    (`scripts/lora/juzgar_estilo.py`: 10 descriptores de `perfil_vocal` en unidades de la variación
    de la propia persona, más el contorno de entonación).
  - También 4 frases en inglés (identidad al cambiar de idioma).
- **Clonado:** sus 4 referencias de siempre (33 s), iguales en base y LoRA.
- **Producción como en voz-stream:** freno de guía 0,75, 6 pasos.
- **LoRA:** en los dos LM (q/k/v/o/gate/up/down), lr 1e-4, acumulación 4, validación por nota.
  300 pasos tardan unos 7 min en una g4dn.

## Resultados (30 clips, semillas 11 y 101, frente a la base; IC 95 % pareado)

| Variante | WER | UTMOS | ECAPA | Distancia de estilo |
|---|---|---|---|---|
| r16 al 100 % | +0,070 ✗ | +0,021 | **+0,053** | −0,198 |
| r16 al 75 % | +0,028 ✗ | +0,069 | +0,035 | −0,182 |
| **r16 al 50 %** | +0,007 | +0,028 | **+0,032** | −0,200 |
| r16 al 25 % | +0,005 | **+0,151** | +0,008 | −0,085 |
| r8, lr 5e-5, 600 pasos | +0,067 ✗ | −0,049 | +0,022 | **−0,291** |
| r16 + cabeza entera | +0,072 ✗ | **−0,247** ✗ | +0,001 | −0,072 |
| r16 parado en el paso 100 | +0,060 ✗ | −0,145 ✗ | +0,019 | −0,214 |
| **Mezcla (Carlos + lectura en/es) al 50 %** | +0,017 | **+0,110** | **+0,031** | **−0,162** |
| **Solo 5 min de Carlos, al 50 %** | +0,013 | +0,032 | **+0,031** | **−0,250** |
| Texto humano de la anotación, al 100 % | **+0,552** ✗ | −0,328 ✗ | −0,006 | −0,075 |

Negrita en las diferencias significativas que importan; ✗ = empeora con IC que no cruza 0.

**Lectura:**
- **La fuerza del LoRA es la palanca:** del 25 % al 100 % cambia identidad por inteligibilidad de
  forma continua. Al 50 % se queda con la mayor parte de la identidad y del estilo sin coste medible de WER.
- **La mezcla con lectura limpia** no evita el WER al 100 %, pero al 50 % suma naturalidad (UTMOS +0,11).
- **Con 5 min** sale lo mismo que con 10,8. El requisito de alta de una voz podría bajar de 10 a 5 min
  (a confirmar con otra persona).
- **El texto de la anotación humana arruina el entrenamiento:** sus tiempos no están alineados con lo
  que se dice. Para entrenar vale la alineación por palabra de whisper, no el texto «mejor».
- **Entrenar la cabeza de difusión empeora la naturalidad:** el LoRA va solo en los LM.

## Confirmación con 5 semillas (75 clips: 55 en español y 20 en inglés)

| Frente a la base | WER | UTMOS | ECAPA | Estilo | Puerta de la 3b |
|---|---|---|---|---|---|
| **Mezcla al 50 %** | −0,006 [−0,030; +0,017] | **+0,108** [+0,038; +0,173] | **+0,029** [+0,016; +0,041]; inglés +0,048 | **−0,122** [−0,211; −0,029] | **✓** |
| **Solo 5 min, al 50 %** | +0,004 [−0,014; +0,019] | +0,060 [0,000; +0,120] | **+0,029** [+0,017; +0,040] | **−0,137** [−0,229; −0,053] | **✓** |
| r16 al 50 % | −0,003 [−0,027; +0,020] | +0,050 [−0,035; +0,129] | **+0,031** [+0,018; +0,043] | **−0,136** [−0,231; −0,042] | ✗ por el IC inferior de UTMOS |

Puerta de la 3b: juez de estilo con IC > 0, ECAPA ≥ −0,005, UTMOS con IC inferior ≥ −0,02 y WER ≤ +0,5 puntos.

## Con la cuantización de producción (simulada en torch: backbone int4 g128, LM de texto int8)

| Frente a la base cuantizada | WER | UTMOS | ECAPA | Estilo |
|---|---|---|---|---|
| **Mezcla al 50 %** | −0,003 [−0,029; +0,023] | **+0,139** [+0,050; +0,224] | **+0,023** [+0,009; +0,037] | **−0,181** [−0,293; −0,067] |
| r16 al 50 % | +0,014 [−0,011; +0,039] | **+0,171** [+0,103; +0,243] | **+0,033** [+0,017; +0,048] | **−0,137** [−0,261; −0,016] |

**La mejora sobrevive a la cuantización.** Esta simulación es más dura que la de nncf: le cuesta a la base
WER +0,037, UTMOS −0,21 y ECAPA −0,035, cuando el int4 real del backbone pasó en su día `banco_ab`. Así
que vale para comparar base y LoRA en igualdad de condiciones, no como medida absoluta del coste de int4.
