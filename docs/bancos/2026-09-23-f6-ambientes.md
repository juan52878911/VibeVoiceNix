# F6 · Ambientes de fondo sintetizados con código (2026-09-23)

Parte automatizable de la fase 6 del [plan de mejora](../plan-mejora-modelo.md). El plan descartó que el
modelo genere el ambiente (mismo coste por fotograma que el habla, sin control). Se probó la
alternativa barata: fondos generados con código ([`scripts/ambiente.py`](../../scripts/ambiente.py):
teclado, lluvia, oficina con murmullo y cafetería, sin muestras de terceros; el murmullo sale de voces
de serie dadas la vuelta, nunca de personas) mezclados a −22 y −16 dB bajo la voz. 8 clips por
condición; juez de sonidos AudioSet AST.

| Fondo | ¿AST lo reconoce? (mezclado, −22 dB) | WER | UTMOS (limpio 3,20) |
|---|---|---|---|
| teclado | teclado 1/8 | igual | 2,57 |
| lluvia | nada | igual | 2,06 |
| oficina | habla, no «multitud» (0/8) | igual | 2,39 |
| cafetería | habla, no «multitud» (0/8); música 2/8 | igual | 2,47 |

**No pasa como realismo:** el juez no reconoce los fondos como lo que pretenden ser. La voz se entiende
igual (el WER no cambia). UTMOS cae 0,6-1,8, pero eso era lo esperado: UTMOS premia el audio limpio y no
sirve para decidir si un fondo gusta.

**Siguiente paso, con decisión humana:** una librería de ambientes reales de licencia libre (CC0),
escuchados antes de adoptarlos. En dobla el fondo ya es el original del vídeo, copiado tal cual; esto
solo aplica a contenido nuevo.
