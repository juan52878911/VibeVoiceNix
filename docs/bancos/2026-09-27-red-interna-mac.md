# La red por dentro con los pesos reales, en el Mac (2026-09-27)

Segunda parte del [plan de la red por dentro](../plan-red-interna-2026-09-26.md), después de
[lo que se midió sin pesos](2026-09-26-red-interna-entorno.md). Etiquetas: **(M)** medido, **(E)** estimado,
**(S)** supuesto. Código en [`scripts/red/`](../../scripts/red/); audio, huellas y `.pt` fuera del repo, en
`~/Documents/red-interna/`.

> **Estado (27-09): continuado en la g4dn.** La campaña se paró en el Mac a mitad y sigue en AWS con los datos
> subidos a `s3://dobla-879381269751/red-interna/`. Aquí: mecánica con pesos reales, atención (I4), parte de I0 y
> parte del barrido de sexo. Sin sondas, sin puertas juzgadas. La sección de puertas se escribió y se subió
> ANTES de ver ningún resultado de la campaña.

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

## Lo que se midió en el Mac antes de pasar a AWS

### Mecánica con pesos reales (M)

`probar_mecanica.py --modelo ~/.cache/vibevoice-nix/modelo --voz sp-Spk1_man.pt`: **pasan las ocho** sin tocar la
carga (58 s). Paridad muestra a muestra con `generate()` (38.400 muestras). Whisper small transcribe literalmente
los primeros clips del bucle (es y en, dos semillas): es habla normal.

- **El md5 de un clip depende de los hilos:** con `OMP_NUM_THREADS=3` el mismo clip (voz `sp-Spk1_man`,
  semilla 11, frase es 0) difiere como máximo en 3·10⁻⁵ (un paso de PCM16), con la misma longitud. Entre
  máquinas, la paridad se compara por longitud y diferencia máxima, no por md5.
- **Fuga en `bucle.Generador` (arreglada, 821e7a8):** los enganches de residuales se registraban en cada
  generador y no se quitaban. `instrumentar.py` pasó de 15 a 30 s por clip y murió por memoria (SIGKILL) en el
  clip 95. Los `.npz` escritos antes son válidos, porque cada enganche escribía en su propio generador.
  `probar_mecanica` comprueba ahora que no queda ningún enganche vivo.
- **Coste:** con residuales, unos 15 s de CPU para 5,2 s de audio (RTF ≈ 2,9, torch fp32, M4) (M). Las frases
  largas (16,5 s) cuestan unos 115-120 s cada una con dos procesos a la vez. Cada proceso con el modelo ocupa
  unos 5 GB: en este Mac caben dos a la vez.

### I0 · Instrumentar (a medias) (M)

101 clips con `.npz` y `.wav`: los 96 del corpus congelado (4 voces × `es`, `en`, `es_numeros` × 4 frases × 2
semillas) y 5 largas de `sp-Spk1_man`. Faltan 27 largas y las 32 cortas. Sin ellas las duraciones no están
variadas, así que **las sondas no se han corrido con los datos buenos.** Jueces sobre 53 de los 101.

### I4 · Atención (M)

`atencion.py` (modo de 3 regiones), una frase española, semilla 11. Masa media por capa sobre [latentes del
prefijo, texto del prefijo, lo generado]:

| Capa | sp-Spk1_man | sp-Spk0_woman | Carlos | Liliana |
|---|---|---|---|---|
| 0 | 0,11 · 0,49 · 0,40 | 0,10 · 0,49 · 0,40 | 0,10 · 0,50 · 0,40 | 0,10 · 0,50 · 0,39 |
| 4 | 0,11 · 0,48 · 0,41 | 0,10 · 0,50 · 0,40 | 0,09 · 0,47 · 0,44 | 0,10 · 0,48 · 0,42 |
| 7 | 0,51 · 0,39 · 0,10 | 0,48 · 0,39 · 0,12 | 0,45 · 0,40 · 0,16 | 0,45 · 0,41 · 0,14 |
| 9 | 0,51 · 0,23 · 0,27 | 0,51 · 0,20 · 0,29 | 0,47 · 0,23 · 0,30 | 0,47 · 0,23 · 0,30 |
| 12 | 0,16 · 0,73 · 0,11 | 0,16 · 0,72 · 0,12 | 0,17 · 0,67 · 0,16 | 0,14 · 0,70 · 0,16 |
| 15 | 0,15 · 0,29 · 0,56 | 0,14 · 0,30 · 0,56 | 0,13 · 0,32 · 0,55 | 0,12 · 0,32 · 0,56 |
| 16 | 0,66 · 0,11 · 0,23 | 0,63 · 0,11 · 0,26 | 0,58 · 0,14 · 0,28 | 0,58 · 0,12 · 0,30 |
| 19 | 0,74 · 0,02 · 0,24 | 0,70 · 0,02 · 0,27 | 0,66 · 0,05 · 0,30 | 0,66 · 0,04 · 0,29 |

- **Las capas 16-19 miran sobre todo el prefijo de voz** (58-74 % de la masa), en las cuatro voces. Es la
  misma zona donde los prefijos daban más identidad (2026-09-26, §1).
- **Cabezas de identidad comunes a las cuatro voces** (entre las 10 con más masa en los latentes del prefijo):
  18.2, 18.12, 19.7 y 19.8, con 0,78-0,91 de masa. Son candidatas a no tocarse al dirigir.
- **Cabezas con toda la masa en lo generado, iguales en las cuatro voces:** 4.3, 19.1 y 0.12 (≈ 1,00); 19.4,
  0.2, 0.5, 10.1, 17.9 y 11.0 (0,90-0,99). Todavía no se sabe si leen el texto o se miran a sí mismas.
  `atencion.py --detalle` lo separa y queda por correr.
- **Los clones miran un poco menos su prefijo que las voces de serie** en las capas 16-19 (0,58-0,66 frente a
  0,63-0,74) (M, una frase). Lo que eso signifique para su identidad está sin medir (S).

### I2 · Barrido de sexo, a medias (M, sin juzgar)

`dirigir.py --clave condicion/sexo --rama ambas --lambdas 0,0.05,0.1,0.2 --grupos es --semillas 11,101`, con la
dirección de `analizar_prefijos.py` (+1 = mujer), que reproduce el 95,5 % y la U del día 26. 75 clips con
perfil: `sp-Spk1_man` y Carlos completos (32 cada uno) y Liliana 11 de 32. Sin jueces: la puerta no está
evaluada.

## Qué queda, con comandos (en AWS)

1. Completar I0 con `corpus_duraciones.json --grupos largas,cortas` (se reanuda solo) y los jueces del resto.
2. `sondas.py --inst inst --salida sondas` (pliegues por frase y por voz), y su puerta.
3. Completar el barrido de sexo (Liliana) y `puerta_dirigir.py lote|puerta --descriptor hz --signo 1`. Con la
   dirección de tono de las sondas: `condicion/f0_st` en ramas `ambas` y `pos`, la mejor `capaNN/f0_st` y la
   tanda `--desde 40` sobre `largas`, con `costura.py`.
4. `sorpresa_wer.py` sobre `inst` con `inst/juez.json`.
5. `atencion.py --detalle` para separar el alineador de las cabezas que se miran a sí mismas.
