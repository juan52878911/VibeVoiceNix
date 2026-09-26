# La red por dentro con los pesos reales, en el Mac (2026-09-27)

Segunda parte del [plan de la red por dentro](../plan-red-interna-2026-09-26.md), después de
[lo que se midió sin pesos](2026-09-26-red-interna-entorno.md). Etiquetas: **(M)** medido, **(E)** estimado,
**(S)** supuesto. Código en [`scripts/red/`](../../scripts/red/); audio, huellas y `.pt` fuera del repo, en
`~/Documents/red-interna/`.

> **Estado:** puertas fijadas y herramientas listas; mediciones en curso. Esta sección de puertas se escribió
> y se subió ANTES de ver los resultados de la campaña.

## Puertas fijadas antes de medir

**I1 · Sondas.** Pasa si el R² de validación es > 0,3 en alguna capa (o en la condición) para los tres
atributos: tono (`f0_st`), energía (`energia_db`) y pausa inminente (`hasta_pausa`). Además tiene que quedar
por encima de la fila `posicion`. La validación que decide es **por frase apartada**: todos los clips de una
frase (todas las voces y semillas) fuera del ajuste. Se informa también por voz apartada. Si ningún atributo
prosódico es lineal en ninguna capa, la vía por direcciones se cierra.

**I2 · Dirección causal, por atributo y por voz**, pareado por (frase, semilla) contra λ = 0:

- el descriptor objetivo se mueve en la dirección pedida con IC 95 % > 0 y de forma monótona en λ;
- ECAPA ≥ −0,005 (media de la diferencia);
- UTMOS con IC inferior ≥ −0,02;
- WER ≤ +0,5 puntos (media de la diferencia);
- ningún clip con WER > 25 % si su base tenía ≤ 10 %.

Pasa la λ máxima que cumple todo. El tono se mide como la mediana de F0 del clip (`perfil_vocal`, `hz`), en
semitonos frente a la base.

**Costura (`--desde 40`).** En la ventana de ±2 s alrededor del fotograma 40 (5,33 s), la misma puerta aplicada
a la ventana: WER ≤ +0,5 puntos, UTMOS con IC inferior ≥ −0,02 y ningún clip roto.

**Sorpresa (§5.5).** Es exploratoria y no tiene puerta de producto. Se considera **útil para ordenar
re-tiradas** si la correlación de Spearman entre la sorpresa media por clip y el WER es ≥ 0,3 con IC 95 % > 0.

## Cambios en las herramientas antes de medir, y por qué

- **Sondas por frase apartada** (`sondas.py --pliegues texto,voz`). Con un clip apartado, la misma frase
  dicha con otra semilla o por otra voz queda en el entrenamiento, con casi el mismo contorno. La sonda puede
  aprenderse la frase en vez de leer la prosodia.
- **α relativa a la escala del sitio.** En 20 clips de prueba, la varianza por dimensión de la condición es unas
  250 veces la de los residuales (mediana 8,5 frente a 0,02-0,35) (M). Con el α = 10 fijo del ensayo en seco, la
  condición daba R² negativo por sobreajuste (−0,6 en tono) y los residuales quedaban muy regularizados. Ahora
  α = k · traza(G)/D, con k ∈ {0,01; 0,1; 1; 10; 100} elegido dentro de cada pliegue externo por 5 trozos
  internos. El ajuste es ridge en forma cerrada con las matrices de Gram acumuladas: coincide con `sklearn`
  Ridge hasta 6·10⁻¹⁵ (M).
- **Duraciones variadas.** Además del corpus congelado, `scripts/red/corpus_duraciones.json` añade 4 frases
  largas (50-54 palabras) y 4 muy cortas (1-3 palabras). `corpus_mejora.json` no se toca.
- **`dirigir.py` reanudable** sin perder los descriptores de lo ya generado. Se añaden la puerta pareada
  (`puerta_dirigir.py`), la costura (`costura.py`) y la sorpresa frente al WER (`sorpresa_wer.py`).
- **Atención con detalle** (`atencion.py --detalle`): lo generado se parte en texto leído, latentes propios
  y la posición actual. Una cabeza con toda la masa en "lo generado" puede estar mirándose a sí misma.

## Montaje

| | |
|---|---|
| Máquina | Mac mini M4, 10 núcleos, 16 GB; torch 2.13 en CPU fp32, transformers 4.57.6 (M) |
| Modelo | `~/.cache/vibevoice-nix/modelo` (el preparado de `voz-stream-mac.sh`, tokenizador Qwen2.5 local) |
| Voces de serie | `sp-Spk1_man`, `sp-Spk0_woman` |
| Clones con consentimiento | Carlos Segura y Liliana Morales. No estaban en la caché del Mac: se fabricaron con `clonar_voz.py --lote` desde el banco de dobla (`voces_banco/<persona>/refs`), con sus semillas de clonado (3 y 4). Se apartó una referencia de cada uno como control de ECAPA (Carlos 7,0 s; Liliana 5,4 s). Techo ECAPA 0,793 y 0,898 (M) |
| Control de ECAPA de `sp-Spk1_man` | no hay audio real de las voces de serie: se usa el centroide de sus propios clips base de otras frases (inglés y números). Vale para diferencias pareadas, no como identidad absoluta |
| Salidas | `~/Documents/mejora-modelo` apunta a un disco externo que no estaba montado: todo va a `~/Documents/red-interna/` |
