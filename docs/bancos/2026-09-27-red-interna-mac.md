# La red por dentro con los pesos reales, en el Mac (2026-09-27)

Segunda parte del [plan de la red por dentro](../plan-red-interna-2026-09-26.md), después de
[lo que se midió sin pesos](2026-09-26-red-interna-entorno.md). Etiquetas: **(M)** medido, **(E)** estimado,
**(S)** supuesto. Código en [`scripts/red/`](../../scripts/red/); audio, huellas y `.pt` fuera del repo, en
`~/Documents/red-interna/`.

> **Estado (26-09, noche UTC): campaña completa, terminada en una g4dn.** Empezó en el Mac y siguió en AWS con los
> datos en el prefijo `red-interna/` del bucket de dobla. **I1 pasa** (el tono, la energía y la pausa inminente se
> leen con R² 0,32-0,69 por frase apartada). **I2 no pasa** en el conjunto: con λ ≤ 0,2 solo Carlos, en la capa 16,
> mueve el tono y pasa la puerta (+0,16 st, casi inaudible). La costura del mando en vivo pasa con λ ≤ 0,1. La
> sorpresa no predice el WER. La atención separa cabezas de identidad, un alineador de texto y cabezas que se miran
> a sí mismas. La sección de puertas se escribió y se subió ANTES de ver ningún resultado de la campaña.

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

## Lo que se midió en la g4dn (26-09, noche UTC)

### Montaje y reutilización

| | |
|---|---|
| Máquina | g4dn.xlarge bajo demanda (Tesla T4 de 15 GB, 4 vCPU, 16 GB), DLAMI con torch 2.7.0+cu128 y transformers 4.57.6; `scripts/lora/gpu_entorno.sh` más scikit-learn, librosa y phonemizer (M) |
| Síntesis | torch **fp32** en la GPU (`RED_DISPOSITIVO=cuda`), sin TF32 y con algoritmos deterministas. El ruido de la difusión sale del generador de CPU, así que la semilla da el mismo ruido que en el Mac (M) |
| Mecánica en cuda | **8/8**, con 0 enganches vivos al acabar (62 s) (M) |
| Orquestación | `scripts/red/campana_gpu.sh`, en dos carriles (síntesis y medida) con subida a S3 cada 3 min |
| Reutilizado del Mac | I0 (101 clips) y sus jueces (53). `sexo_ambas` estaba a medias (Liliana 11 de 32) y se apartó a `dirigir_mac/` para rehacerlo entero en la GPU: λ = 0 y λ > 0 tienen que salir de la misma máquina (y con los mismos hilos: en CPU, cambiar de 8 a 4 hilos separa el audio a 23,6 dB, [ec2-y-coste.md §10](../ec2-y-coste.md)) |
| Velocidad | RTF 1,1 con registro de residuales y un solo proceso (GPU al 24 %: manda el Python); con dos procesos a la vez la GPU sube al 99 % |

### Un fallo de `sondas.py`, arreglado antes de leer nada (M)

La primera pasada de las sondas dio R² de −10²⁸, incluso en la fila de control `posicion`. La causa era el
promedio de R² por clip. A cada etiqueta continua se le quita la tendencia de reloj con 10 columnas (cúbica más
la fase de 6), y en las frases cortas (menos de ~12 fotogramas) eso deja una varianza residual de 10⁻³⁰. El R² de
ese clip explota y arrastra la media. Arreglo (4b4b1d2): R² agregado sobre todo lo apartado (1 − ΣSSE/ΣSST, la
exactitud también agregada) y fuera de las etiquetas continuas los clips con menos fotogramas que columnas de
reloj más 4. Las sondas se corrieron en el Mac (32 min, 10 núcleos) sobre los mismos 160 clips: en los 4 vCPU de
la g4dn iban a más de 40 min con la GPU parada.

### I0 · Instrumentar, completo (M)

160 clips: los 101 del Mac y 59 nuevos en la GPU (largas y cortas de las 4 voces, semillas 11 y 101), 8219
fotogramas. Jueces de `juez_lote.py` (whisper large-v3, ECAPA, UTMOS) sobre los 160:

| Grupo | WER normalizado (media de las 4 voces) | UTMOS |
|---|---|---|
| es | 0,2 % | 3,39-3,73 |
| en | 0,9 % | 3,70-4,18 |
| es_numeros | **7-13 %** | 3,11-3,80 |
| largas (50-54 palabras) | 0,3 % | **2,79-3,29** |
| cortas (1-3 palabras) | 0 % | 3,44-3,54 |

Los números siguen siendo el punto flojo del WER (lo conocido del normalizador), y las frases largas pierden
naturalidad: UTMOS medio unos 0,4-0,6 por debajo de las normales, con el WER intacto.

### I1 · Sondas, con los 160 clips (M): **pasa**

R² de validación agregado (exactitud en `pausa`). Pliegues por frase apartada, que es lo que decide la puerta,
y entre paréntesis por voz apartada. Selección de sitios; la tabla entera está en `sondas/r2_por_sitio.json`.

| Sitio | Tono (`f0_st`) | Energía | Sonoridad | Pausa (exactitud) | Pausa inminente | Hasta el fin |
|---|---|---|---|---|---|---|
| `posicion` (solo reloj) | 0,00 (0,00) | 0,00 (0,00) | 0,00 (0,00) | 0,913 (0,913) | 0,00 (0,00) | 0,00 (0,00) |
| condición final | 0,46 (0,24) | 0,68 (0,59) | 0,55 (0,44) | 0,969 (0,945) | 0,26 (0,20) | 0,04 (0,12) |
| negativa | 0,43 (0,18) | 0,62 (0,52) | 0,47 (0,41) | 0,962 (0,910) | 0,19 (0,13) | 0,01 (0,04) |
| capa 0 | 0,17 (−0,03) | 0,39 (0,19) | 0,28 (0,23) | 0,943 (0,912) | 0,06 (0,03) | 0,01 (0,03) |
| capa 4 | 0,30 (0,06) | 0,45 (0,31) | 0,31 (0,29) | 0,949 (0,905) | 0,09 (0,03) | 0,01 (0,07) |
| capa 8 | 0,36 (0,22) | 0,53 (0,43) | 0,38 (0,36) | 0,955 (0,903) | 0,13 (0,11) | 0,02 (0,10) |
| capa 11 | 0,41 (0,28) | 0,62 (0,51) | 0,47 (0,45) | 0,960 (0,903) | 0,26 (0,21) | 0,05 (0,47) |
| capa 14 | 0,46 (0,32) | 0,66 (0,58) | 0,53 (0,47) | 0,966 (0,916) | **0,32** (0,25) | 0,06 (0,33) |
| capa 16 | **0,48** (0,37) | **0,69** (0,65) | 0,56 (0,48) | 0,968 (0,965) | 0,29 (0,25) | 0,03 (0,19) |
| capa 19 | 0,46 (0,27) | 0,69 (0,60) | 0,55 (0,42) | 0,969 (0,953) | 0,26 (0,21) | 0,03 (0,12) |

- **Pasa la puerta:** tono, energía y pausa inminente superan 0,3 en alguna capa (tono 0,48 y energía 0,69 en la
  16, pausa inminente 0,32 en la 14) y la fila de reloj da 0. La prosodia es lineal y legible.
- **Dónde:** crece con la profundidad hasta las capas 14-17 y la condición final. Las capas 0-4 casi no la
  tienen. No coincide con la U de la identidad del día 26 (mínima en 9-11): la prosodia vive donde también vive
  la identidad.
- **Por voz apartada cae** (tono 0,48 → 0,37). Parte de lo que lee la sonda es propio de cada voz.
- `pregunta` da 0,910 en todos los sitios, que es la clase mayoritaria: con 3 preguntas en el corpus no se puede
  leer. `ventana` es función pura del reloj y, sin su tendencia, no queda nada que leer (R² negativo, esperado).

### I2 · Dirección causal (M): **no pasa**

Pareado por (voz, frase, semilla) contra λ = 0, 3 voces (`sp-Spk1_man`, Carlos, Liliana) × 4 frases `es` × 2
semillas = 24 parejas por λ. Tono = mediana de F0 en semitonos frente a la base. Fila «todas» con λ = 0,2:

| Barrido | Tono, dif. (IC 95 %) | ECAPA dif. | UTMOS dif. (IC inf.) | WER dif. | Veredicto |
|---|---|---|---|---|---|
| sexo, condición, ambas ramas | +0,19 st [−0,03, +0,45] | +0,004 | −0,027 (−0,080) | −0,23 pts | no: no mueve, UTMOS |
| tono, condición, ambas ramas | +0,03 st [−0,21, +0,32] | −0,005 | −0,002 (−0,060) | +0,00 pts | no: no mueve, no monótono, UTMOS |
| tono, condición, rama pos | +0,18 st [−0,05, +0,44] | −0,004 | −0,044 (−0,106) | +0,00 pts | no: no mueve, UTMOS (Carlos sí se mueve, +0,29 st, pero falla ECAPA −0,0068 y UTMOS) |
| tono, residual de la capa 16 | +0,22 st [−0,09, +0,55] | +0,003 | −0,010 (−0,060) | +0,00 pts | no en el conjunto; **pasa en Carlos con λ = 0,2** |

- **Un solo caso pasa la puerta, en una voz:** Carlos, residual de la capa 16, λ = 0,2. Tono +0,16 st
  [+0,02, +0,36], ECAPA +0,011, UTMOS −0,012 de IC inferior y WER sin cambio. Es un sexto de semitono: pasa la
  puerta, pero no es un mando que se oiga. En la rama positiva Carlos sube +0,29 st [+0,08, +0,55] y falla por
  ECAPA (−0,0068) y UTMOS.
- **En el conjunto no pasa ninguna λ.** Liliana y `sp-Spk1_man` no se mueven con IC > 0 en ningún barrido.
- **El WER no se inmuta** (0,00 puntos en tono, −0,23 en sexo) y ECAPA se queda en ±0,012. Hay margen para
  fuerzas mayores, que es lo siguiente que hay que medir.
- **Se lee pero no se conduce:** la misma dirección que predice el tono con R² 0,46-0,48 casi no lo cambia al
  sumarla con estas fuerzas. O la dirección es correlacional, o λ = 0,2 de la norma se queda corta.
- `sexo_ambas` se rehízo entero en la GPU. La versión del Mac, a medias, está en `dirigir_mac/`.

### Mando en tiempo real, `--desde 40` (M)

Tono en la condición, ambas ramas, desde el fotograma 40 (5,3 s) sobre las 4 frases largas: +0,05 st
[−0,05, +0,15] con λ = 0,2. No se mueve, pero la calidad se conserva (ECAPA −0,001, UTMOS −0,019 de IC inferior,
WER igual).

**Costura** (`costura.py`, ventana de ±2 s alrededor del fotograma 40, 24 parejas):

| λ | WER de la ventana (IC 95 %) | UTMOS de la ventana (IC 95 %) | Rotos | Veredicto |
|---|---|---|---|---|
| 0,05 | −0,38 pts [−1,14, +0,00] | +0,004 [−0,006, +0,015] | 0 | pasa |
| 0,1 | −0,68 pts [−1,73, +0,00] | +0,002 [−0,006, +0,011] | 0 | pasa |
| 0,2 | −0,68 pts [−1,73, +0,00] | −0,014 [−0,048, +0,014] | 0 | no: UTMOS |

Cambiar el mando en mitad de una locución no deja cicatriz con λ ≤ 0,1. El mecanismo del tiempo real está
listo; lo que falta es una dirección que mueva algo.

### Sorpresa frente al WER (M): **no sirve para ordenar re-tiradas**

160 clips con WER. La puerta pedía Spearman ≥ 0,3 con IC > 0:

| Resumen de la sorpresa por clip | Spearman con el WER (IC 95 %) | AUC (WER > 0,10, 12 clips) |
|---|---|---|
| media | +0,01 [−0,12, +0,15] | 0,58 |
| p90 | +0,08 [−0,07, +0,22] | 0,61 |
| racha sobre el p90 global | +0,13 [−0,02, +0,28] | 0,67 |

Por voz hay indicios en los clones (Carlos +0,31, Liliana +0,24, n = 40 cada una) y ninguno en las de serie.
Con tan pocos clips malos (12 de 160, casi todos de números) la prueba tiene poca potencia. Aun así, la media no
separa nada.

### I4 · Atención con detalle (M)

`atencion.py --detalle`, una frase española, semilla 11. Las 5 regiones son: latentes del prefijo, texto del
prefijo, texto leído, latentes propios y posición actual. Lo que se repite en las cuatro voces:

| Papel | Cabezas (capa.cabeza) | Masa |
|---|---|---|
| Identidad (miran el prefijo de voz) | 18.12, 18.2, 19.13, 19.8 | 0,79-0,91 |
| Alineador de texto (miran el texto leído) | 11.0, 11.10, 11.9, 8.1 | 0,63-0,93 |
| Se miran a sí mismas (posición actual) | 19.1, 4.3, 0.12, 17.9 | 0,94-1,00 |
| Latentes propios generados | 0.5, 15.3, 15.13, 0.13 | 0,65-0,85 |

- **Resuelve la duda del Mac:** las cabezas con «toda la masa en lo generado» (19.1, 4.3, 0.12) se miran a sí
  mismas, no leen el texto. El alineador de texto es la capa 11 (y la 8.1), igual en las cuatro voces.
- **Capas 16-19:** 58-74 % de la masa sobre el prefijo de voz. Los clones miran algo menos su prefijo (0,58-0,66)
  que las voces de serie (0,63-0,74), como ya se vio en el Mac.

### Coste (M)

| Máquina | Horas | USD | Para qué |
|---|---|---|---|
| g4dn.xlarge bajo demanda (0,526 $/h) | 2,12 | 1,165 | I0 restante, 5 barridos de dirigir, jueces de I0 y de sexo, sorpresa, atención con detalle |
| 2 × c8a.2xlarge spot | 0,56 | 0,137 | guiones del spot y paridad ([ec2-y-coste.md §10](../ec2-y-coste.md)) |
| Mac | — | 0 | sondas (32 min) y jueces de los barridos de tono (CPU int8) |
| **Total del día** | | **1,30** | acumulado del plan: 13,78 de 20 USD |

Dos contratiempos de coste, ya resueltos: `juez_lote.py` se quedó sin memoria en la T4 (CUDA out of memory) con
dos síntesis a la vez (11,3 de 15 GB), y los jueces de los barridos de tono se pasaron al Mac. Dentro de cada
barrido, un solo juez juzga λ = 0 y λ > 0. El libro de gasto de siempre está en un disco externo sin montar: las
tandas del día se apuntaron en uno provisional con el gasto previo arrastrado.

## Qué queda

1. **Fuerzas mayores:** λ = 0,5 y 1,0 en la capa 16 y en la condición. El WER no se movió con 0,2 y la puerta
   manda subir hasta que algo se rompa.
2. **Dirigir en las capas 11-14**, donde empieza la prosodia legible y está el alineador. Y **parchear**: copiar el
   residual de una locución aguda en otra, para separar «aquí se lee» de «aquí se decide».
3. **La naturalidad de las frases largas** (UTMOS −0,5 frente a las normales, con el WER intacto) es un hallazgo
   de I0 que merece su propio banco.
4. **Fusionar el libro de gasto provisional** con el de siempre al montar el disco.
