# F4 · Acento a voluntad por conversión de voz (2026-09-23)

Fase 4 del [plan de mejora](../plan-mejora-modelo.md). El problema: al doblar, el clon habla el otro
idioma con el acento de la persona. La idea: que el acento lo ponga una **fuente** que ya habla como
queremos y que la **conversión** le ponga el timbre de la persona. Se usó kNN-VC (MIT, sin
entrenar) con un banco de timbre de audio real de cada persona, que nunca se usa para evaluar
([`scripts/f4_conversion.py`](../../scripts/f4_conversion.py)).

Destinos con consentimiento: Juan (1,6 min de nota de voz de WhatsApp a 17 kbps), Carlos (11,6 min de
charla) y Liliana (1,6 min). Fuentes: voces de serie de VibeVoice hablando inglés nativo (Carter y
Emma) o inglés con acento español (sp-Spk3 y sp-Spk0). Juicio en GPU; identidad contra audio real
apartado.

## Resultados: la persona hablando inglés (4 frases base)

| | Identidad | frente al clon directo | PER (acento; nativo 0,02-0,05) | UTMOS |
|---|---|---|---|---|
| **Carlos**, clon directo | 0,462 | — | 0,141 | 3,64 |
| Carlos, convertido desde inglés nativo | **0,503** | **109 %** | **0,101** | **3,98** |
| **Liliana**, clon directo | 0,540 | — | 0,145 | 3,18 |
| Liliana, convertido desde inglés nativo | **0,521** | **96 %** | **0,048** | **3,43** |
| **Juan**, clon directo | 0,536 | — | 0,142 | 3,23 |
| Juan, convertido desde inglés nativo | 0,386 | **72 %** | 0,117 | 3,69 |

Otros datos:
- **Tamaño del banco** (Carlos, frases base y con números): 1,6 min → 0,450; 11,6 min → 0,492.
- **Ampliar el banco con el clon de la persona hablando inglés** baja la identidad en los tres
  (Carlos 0,503 → 0,408). El clon solo se parece un 50 % a la persona, y eso es lo que entra en el
  banco. **Descartado.**
- **Desde una fuente con acento español** la conversión da el mismo acento que el clon directo, así
  que para conservarlo no hace falta convertir.
- **Velocidad:** RTF 0,028 en una T4 y **0,095 en CPU** (Mac M4, 4 hilos); preparar el banco, 8 s por
  persona y una sola vez. En el i7-8700T de producción, estimado 2-3 veces más lento (sin medir).

## Veredicto frente a la puerta

| Puerta | Carlos | Liliana | Juan |
|---|---|---|---|
| Identidad ≥ 0,9 × clon directo | ✅ | ✅ | ❌ |
| PER ≤ fuente × 1,1 (nativo del todo) | ❌ (0,101 frente a 0,051) | ❌ (0,048 frente a 0,025) | ❌ |
| UTMOS ≥ clon − 0,10 | ✅ | ✅ | ✅ |
| WER igual al de la fuente | ✅ | ✅ | ✅ |

**Pasa en parte.** Con suficiente audio real de la persona (minutos, como en una charla) la conversión
conserva su identidad, suena más natural y acerca mucho el acento al nativo (en Liliana baja dos
tercios), pero no lo iguala. Con una nota de voz corta y comprimida, la identidad no aguanta.

Muestra pequeña: 4 frases base por condición (8 con las de números). Antes de llevarlo a dobla hay
que repetirlo con más frases y más hablantes, y medir el RTF en la VM.

## F4c · ¿Y como vía general de calidad, en el mismo idioma? — no

La F3 mostró que el modelo suena mejor con audio de estudio en el prefijo. Se probó generar en español con
una voz de serie limpia (sp-Spk3 para Carlos, sp-Spk0 para Liliana) y convertir al timbre de la persona,
con los mismos 12 segmentos y semillas que su clon directo de la F1 (pareado):

| | Identidad | WER | UTMOS |
|---|---|---|---|
| Carlos, convertido − clon directo | **+0,036** [+0,008, +0,062] | −0,009 (sin efecto) | **−0,235** [−0,362, −0,101] |
| Liliana, convertido − clon directo | +0,004 (sin efecto) | −0,016 (sin efecto) | **−0,097** [−0,174, −0,023] |

En su propio idioma el clon directo ya es natural, y la conversión le añade artefactos del vocoder. La
subida de naturalidad del inglés venía de que el clon habla inglés peor que una voz nativa. **La
conversión se queda solo para el cambio de idioma.**

