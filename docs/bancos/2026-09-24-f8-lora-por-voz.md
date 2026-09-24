# F8 · LoRA por voz: identidad y personalidad de un clon (2026-09-24)

**Resultado: primera mejora de pesos del plan que pasa la puerta.** Un LoRA por voz, entrenado con
~10 min de la persona mezclados con lectura general, **aplicado al 75 % de su fuerza** (receta al final):

| Frente al clon de hoy | Sin cuantizar | Con la cuantización de producción |
|---|---|---|
| Identidad (ECAPA) | **+0,049** [+0,036; +0,062] | **+0,045** [+0,029; +0,061] |
| Estilo (desviaciones de la persona) | **−0,209** [−0,305; −0,119] | **−0,168** [−0,282; −0,055] |
| Naturalidad (UTMOS) | **+0,080** [+0,014; +0,142] | **+0,158** [+0,076; +0,241] |
| WER | −0,001 [−0,024; +0,021] | −0,004 [−0,027; +0,020] |

En inglés, que Carlos nunca grabó, la identidad sube todavía más. Con **5 min** de audio también pasa
(al 50 %); con 2 min no.

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

## Rondas 7-10: más fuerza con la mezcla, rango y pasos (75 clips)

| Frente a la base | WER | UTMOS | ECAPA | Estilo |
|---|---|---|---|---|
| Mezcla r16 al 65 % | −0,007 | +0,073 [−0,000; +0,144] | **+0,048** | **−0,188** |
| **Mezcla r16 al 75 %** | −0,001 [−0,024; +0,021] | **+0,080** [+0,014; +0,142] | **+0,049** [+0,036; +0,062] | **−0,209** |
| Mezcla r16 al 85 % | +0,014 [−0,010; +0,037] (> 0,5 puntos) | +0,048 | +0,056 | −0,231 |
| Mezcla r32 al 75 % | +0,005 [−0,018; +0,027] | **+0,141** | **+0,050** | **−0,260** |
| Mezcla r16 con 1000 pasos, al 75 % | +0,005 | +0,042 | +0,041 | −0,217 |
| Solo 2 min, al 50 % | +0,037 [−0,018; +0,128] | +0,099 | +0,015 | −0,117 |

**Con la cuantización de producción:**
- Mezcla r16 al 75 %: WER −0,004, UTMOS +0,158, ECAPA +0,045, estilo −0,168. **Pasa.**
- Mezcla r32 al 75 %: WER +0,014 [−0,016; +0,045], UTMOS +0,144, ECAPA +0,048, estilo −0,236, contorno +0,056.

**Lectura:**
- La mezcla con lectura limpia es lo que permite subir la fuerza: el LoRA solo, al 75 %, ya subía el WER
  (+0,028). Con la mezcla, al 75 % no lo toca.
- El techo está en el 75 %: al 85 % el WER empieza a subir.
- Rango 32 suena mejor en estilo y naturalidad, pero cuantizado roza el WER. 1000 pasos no aportan.
- **2 min se quedan cortos; 5 min es el mínimo útil.**

## Receta

1. **Datos de la persona** (≥ 5 min, mejor ~10): `datos_voz.py --apartar 0` sobre sus grabaciones.
   Whisper large-v3 con marcas por palabra corta en pausas. **No** usar el texto de una anotación con
   tiempos aproximados.
2. **Repaso:** lectura general en/es con `datos.py --idiomas en,es --hablantes 20` (unos 360 ejemplos por
   idioma). A la persona se le pone `idioma = "voz"`, para que cada paso tenga un tercio de ella.
3. **Entrenamiento:** `entrenar.py --rango 16 --alfa 32 --lr 1e-4 --pasos 600 --acumular 4 --cada 50`.
   Solo los dos LM, sin cabeza. Unos 14 min en una g4dn (~0,12 USD).
4. **Uso:** fundir con `alfa 24` (75 % de la fuerza) y cuantizar como siempre.

## Para producción (no hecho)

- **El LoRA va fundido en los pesos:** cada voz necesita su propio IR del backbone (int4, ~150-300 MB) y
  su LM de texto. Se genera una vez al dar de alta la voz y se guarda con ella (`voces/<id>/` en S3).
- **dobla:** si una voz del vídeo tiene IR propio, el motor tiene que cargar ese backbone para sus
  segmentos. Hoy voz-stream carga uno solo al arrancar; con varias voces con LoRA hace falta un backbone
  por voz en memoria (int4: ~150 MB cada uno) o un servidor por voz.
- **Validar con una segunda persona:** con Liliana no hubo material aparte de su evaluación (0,7 min). Las
  notas de WhatsApp de Juan (≥ 5 min) son la siguiente prueba.
- **Consentimiento:** el LoRA es un modelo de la voz de la persona; solo con su permiso expreso para eso.

## Coste

4,2 h de g4dn.xlarge (~2,25 USD) para 10 rondas. Gasto de GPU del plan: 8,32 de 20 USD. LoRA, medidas y
muestra de escucha en `~/Documents/mejora-modelo/f8/` (fuera del repo: llevan la voz de Carlos).

## Coste y decisión (24-09-2026)

| Concepto (por voz) | Coste | Fuente |
|---|---|---|
| Preparar datos (~10 min de audio) | ~9 min de g4dn, ~0,08 USD | medido |
| Entrenar el LoRA (600 pasos, mezcla) | ~14 min de g4dn, ~0,12 USD | medido |
| Lectura general para la mezcla | una vez, compartida por todas las voces | medido |
| Convertir el backbone de la voz a OpenVINO int4 | ~15 min de CPU, ~0,02 USD en AWS | estimado |
| **Total por voz, una sola vez** | **~0,20-0,25 USD y ~40 min** | |
| Guardarla (~200-300 MB) | ~0,005 USD al mes | estimado |
| Usarla en cada doblaje | segundos (bajar y compilar); **el RTF no cambia** | estimado |

Lo caro no es el cómputo:
- **Ingeniería, una sola vez:** un backbone por voz en voz-stream y dobla, y el entrenamiento al dar de alta la voz.
- **Pedir ≥ 5 min de audio limpio** y el permiso expreso de la persona para modelar su voz.
- **Memoria:** un backbone por voz en los vídeos con varias voces con LoRA.

**Decisión de Juan (24-09): no se hacen versiones por voz por ahora.** Queda documentado para cuando
haya un producto de «voz propia» (personas que narran o doblan a menudo con su voz). Antes de
integrarlo, validarlo con una segunda persona.
