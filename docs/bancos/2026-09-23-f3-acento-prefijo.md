# F3 · Acento cambiando solo el prefijo del clon (2026-09-23)

Fase 3 del [plan de mejora](../plan-mejora-modelo.md): ¿se puede cambiar el acento del clon sin tocar
pesos, solo con lo que entra en el prefijo? Base: la referencia espontánea de F1. Hablantes con
consentimiento: Carlos y Liliana, hablando las 8 frases en inglés del corpus con 2 semillas (16 clips
por variante). Clonado con [`scripts/f3_acento.py`](../../scripts/f3_acento.py). Comparación pareada
contra el prefijo puro:

| Variante | Carlos: identidad / PER / UTMOS | Liliana: identidad / PER / UTMOS |
|---|---|---|
| mix 30/10 (+10 s de voz nativa) | **−0,196** / −0,136 / +0,55 | **−0,171** / −0,060 / +0,55 |
| mix 20/20 | −0,339 / −0,139 / +0,65 | −0,342 / −0,102 / +1,06 |
| mix 10/30 | −0,410 / −0,138 / +0,50 | −0,431 / −0,142 / +1,14 |
| mix 20/20, lo nativo delante | −0,419 / −0,114 / +0,65 | −0,479 / −0,114 / +0,79 |
| transcripción traducida (B3) | −0,043 / **+0,118** / −0,37 | +0,048 / **+0,062** / +0,04 |

Todos los intervalos de identidad de las mezclas excluyen el 0.

**No pasa** (la puerta era identidad ≥ −0,03 y el PER a medio camino del nativo):
- Mezclar audio nativo en el prefijo acerca el acento (con solo 10 s, el PER de Carlos baja de 0,255 a
  0,119), pero **mezcla las voces**: la identidad cae desde el primer segundo nativo. El orden importa:
  lo nativo delante es aún peor.
- La transcripción traducida empeora el acento: la receta del prefijo necesita el texto literal.

**La pista que deja:** en todas las mezclas la naturalidad sube mucho (+0,5 a +1,1 de UTMOS). El modelo
suena mejor cuando el prefijo tiene audio limpio de estudio. Junto con la F4 (convertir desde una voz
de serie limpia también sube la naturalidad sin perder identidad), apunta a probar la conversión como
vía general de calidad, no solo de acento: ver F4c.
