# Plan: personalidad de la voz (2026-09-13)

## El problema, medido

Las voces clonadas se oyen «planas»: les falta lo que hace que una voz sea esa persona y no un bot.
Con el banco A/B (238 clips de producción, `scripts/banco_ab.py`) se ve en los números:

| | recorrido tonal | desviación de F0 | UTMOS |
|---|---|---|---|
| clones (andrés, isis, juan, santiago) | 6,7-7,5 st | 2,0-2,3 st | 1,9-3,1 |
| voces de serie | 8,6-9,4 st | 2,5-2,8 st | 3,0-3,6 |

### Y lo que dice el audio real, que corrige la hipótesis

El perfil de las cinco personas autorizadas, con el mismo detector (octavas corregidas), sobre sus
grabaciones reales:

| persona | F0 | recorrido | desviación | movimiento | sílabas/s | rango de energía | inclinación espectral |
|---|---|---|---|---|---|---|---|
| Carlos Segura | 129 Hz | 7,0 st | 2,16 st | 32 st/s | 5,2 | 22,5 dB | −12,6 dB |
| Liliana Morales | 189 Hz | 5,6 st | 1,75 st | 29 st/s | 6,0 | 20,8 dB | −9,4 dB |
| Laura Rodríguez | 233 Hz | 8,0 st | 2,32 st | 35 st/s | 4,9 | 15,4 dB | −7,0 dB |
| Juan Pablo Rojas | 152 Hz | 9,8 st | 2,79 st | 58 st/s | 5,8 | 16,5 dB | −13,7 dB |
| Daniel Felipe Morales | 117 Hz | 8,0 st | 2,22 st | 44 st/s | 4,3 | 18,2 dB | −9,8 dB |

**Las personas reales no se mueven más en tono que los clones** (5,6-9,8 st frente a 6,7-7,5). Si la
fase 1 lo confirma persona a persona, lo «plano» no es el recorrido tonal y la red no tiene que ir a
por él: hay que buscarlo en la microvariación, la dinámica de energía, la calidad de voz o la
articulación, que es justo lo que decide la fase 1 antes de gastar días de CPU.

## Dónde se puede actuar

El decodificador acústico es un **sumidero**: convierte latentes (7,5 por segundo, 64 números) en onda
y no realimenta nada. La entonación, el ritmo y las pausas ya vienen decididos en esos latentes, que
salen del LM y de la difusión. Medido el mismo día: tocar el decodificador (int4) movió el UTMOS −0,043
pero el recorrido tonal solo −0,04 st; tocar la difusión movió las pausas (+15 ms).

- **decodificador** → timbre, textura, naturalidad; no crea melodía
- **difusión** → entonación, energía, expresividad (se ejecuta 6 veces por fotograma, pero es pequeña)
- **LM** → ritmo, pausas, énfasis (lo más caro)

Presupuesto: un fotograma de 133 ms de audio cuesta hoy ~103 ms en la VM; quedan ~30 ms libres, ~43
con las optimizaciones pendientes.

## Los datos: voces con consentimiento

Del banco de `dobla` (S3, `voces/identidades.json`, consentimiento registrado el 09-09-2026), aisladas con
su propia separación (demucs → `voces24k.wav`) y sus anotaciones corregidas en el editor. **Fuera**: la
reunión de más de una hora (una persona sin consentimiento registrado) y los hablantes sin identidad.

| persona | habla limpia | clips ≥ 2 s |
|---|---|---|
| Carlos Segura | 12,8 min | 131 |
| Liliana Morales | 2,6 min | 33 |
| Laura Rodríguez | 1,1 min | 6 |
| Juan Pablo Rojas | 0,3 min | 2 |
| Daniel Felipe Morales | 0,2 min | 1 |

16,9 minutos, el 76 % de una sola persona. **No da para entrenar un modelo de voz desde cero**; sí para
medir, para un codificador pequeño con fragmentos, y para un adaptador. Las voces de los amigos (andrés,
isis, juan, santiago) no aparecen en estas grabaciones — sus clones no pasan de 0,44 de coseno ECAPA
con ningún hablante — y entran cuando haya grabaciones suyas.

## Fases, cada una con su puerta

### 0 · VM taller

`scripts/modo_taller.sh on` + `scripts/crear_taller.sh`: clon de la VM voz con 11 GB y 6 núcleos
(`cpuunits` 512, por debajo de AuraCRM), la voz y app-noticias apagadas mientras dure. Ahí caben el
codificador acústico (1,3 GB en fp32) y `clonar_voz.py`, que en la VM voz no caben.

### 1 · El perfil vocal y la distancia al clon

Para cada persona, sobre su audio real y sobre su clon diciendo los MISMOS textos: F0 (mediana,
recorrido, desviación, movimiento, con octavas corregidas), sílabas por segundo, pausas por minuto,
rango de energía, inclinación espectral, HNR y huella ECAPA.

**Puerta**: la diferencia real − clon en cada descriptor, con intervalo por bootstrap. Si no hay una
diferencia clara en entonación o energía, lo «plano» es timbre y la fase 3 va al decodificador.

#### Resultado (13-09-2026, VM voz en modo taller, ~45 min, `scripts/fase1_personalidad.py`)

Cada persona clonada con ~33 s de sus clips más largos y su transcripción; el clon dice los textos del
RESTO de sus clips (semilla 101, cfg 3,5, 6 pasos; `decir.py` a RTF 2,83 en CPU). Diferencia clon − real
con IC 95 % por bootstrap. Laura (4 clips de prueba), Juan Pablo y Daniel Felipe (0) quedan clonados pero
sin evaluar.

| descriptor | Carlos (n=127): real → clon | clon − real [IC] | Liliana (n=27): real → clon | clon − real [IC] |
|---|---|---|---|---|
| F0 | 129 → 131 Hz | +2,2 [−0,5, +4,7] | 189 → 203 Hz | **+18,9 [+12,6, +25,2]** |
| recorrido tonal | 6,96 → 8,24 st | **+0,99 [+0,68, +1,32]** | 5,66 → 8,79 st | **+3,14 [+2,54, +3,72]** |
| desviación | 2,15 → 2,52 st | **+0,29** | 1,71 → 2,68 st | **+0,91** |
| microvariación | 16,5 → 18,5 cents | **+2,3** | 14,3 → 17,8 cents | **+4,1** |
| sílabas/s | 5,22 → 4,63 | **−0,59 [−0,73, −0,45]** | 6,31 → 6,59 | +0,17 [−0,13, +0,48] |
| pausas/min | 20,9 → 10,2 | **−8,9 [−10,7, −7,0]** | 13,1 → 13,6 | −3,5 [−7,7, +1,3] |
| rango de energía | 22,5 → 21,1 dB | **−1,17 [−1,60, −0,78]** | 20,8 → 21,0 dB | +0,1 [−0,9, +1,0] |
| inclinación espectral | −12,6 → −12,1 dB | **+0,50** | −9,7 → −8,4 dB | **+1,23** |
| armonicidad (HPSS) | 1,23 → 1,00 dB | **−0,34 [−0,67, −0,02]** | 2,38 → 1,06 dB | **−0,95 [−1,64, −0,25]** |

**La hipótesis se cae.** Los clones no son planos de tono: se mueven MÁS que las personas (recorrido,
desviación y microvariación por encima en los dos). Lo que les falta es **ritmo y pausas** (el clon de
Carlos hace la mitad de pausas y va más lento), **dinámica de energía** y **calidad de voz**: menos
armónica y más brillante. Y el de Liliana sale ~1,3 st más agudo que ella.

Salvedad que no se puede ignorar: el audio real es conversación ESPONTÁNEA separada con demucs, y el clon
LEE la transcripción. Parte de las pausas y de la velocidad es estilo de habla espontánea, y la separación
puede tocar la armonicidad medida. La fase 3 tiene que medirse contra eso, no contra una lectura.

**Consecuencia para la fase 3**: el adaptador no va a por el tono. Ritmo y pausas son del **LM** (lo más
caro de condicionar); dinámica, calidad de voz e inclinación son de la **difusión y el decodificador**.
El orden razonable es empezar por lo barato y medible: condicionar la difusión (energía, calidad) y dejar
el ritmo para después.

### 2 · Codificador de estilo

Red pequeña (convolucional sobre mel, ~1-3 M de parámetros) que, de un fragmento de 2 s, predice los
descriptores de la fase 1 y separa a las personas (pérdida contrastiva). Fragmentos con solape: miles
de ejemplos de los 16,9 min. Validación con clips enteros apartados por persona.

**Puerta**: predice los descriptores de clips no vistos mejor que la media (R² > 0) y reconoce a la
persona por encima del azar con validación dejando fuera una grabación. Sirve como **juez de
personalidad** en el banco A/B aunque la fase 3 no llegue.

#### Resultado (13-09-2026, `scripts/fase2_estilo.py`)

1473 fragmentos (765 reales, 708 de clones de la fase 1), validación con 64 clips enteros apartados;
17 épocas (~10 min de entrenamiento en la VM en modo taller, la mejor la 9).

| salida | clips apartados | puerta |
|---|---|---|
| estilo, R² por clip | 8/8 positivos: tono 0,88 · inclinación 0,63 · recorrido 0,32 · desviación 0,30 · energía 0,27 · microvariación 0,16 · armonicidad 0,06 · movimiento 0,05 | pasa |
| persona | 100 % en Carlos, Laura y Liliana (azar 33 %) — Carlos y Liliana salen de la MISMA grabación, así que no es el canal | pasa |
| real / clon | AUC 1,000 | pasaba, **pero no vale** |

**El control de canal lo tumbó** (`scripts/fase2_control_canal.py`): el audio REAL de los clips apartados,
pasado por el códec de VibeVoice (codificador → decodificador), sale marcado como clon — P(real) 0,917 el
real, **0,251 la ida y vuelta**, 0,095 el clon; AUC real frente a ida y vuelta **0,996**. La ida y vuelta
casi no mueve la prosodia (tono −0,6 Hz, recorrido +0,09 st, microvariación +0,7 cents): el juez oía la
TEXTURA DEL CÓDEC, no la personalidad. Queda algo más (ida y vuelta frente a clon, AUC 0,80), tapado por
el códec.

Consecuencia: el codificador de **estilo** y la salida de **persona** sirven; el juez real/clon se
reentrena con el audio real de ida y vuelta etiquetado como real (`scripts/fase2_idavuelta.py`,
`--idavuelta`), con su puerta fijada antes: AUC(ida y vuelta, clon) > 0,7 y AUC(real, ida y vuelta) < 0,75.

#### Resultado del reentrenamiento (13-09-2026, fase 2b)

Los 173 clips reales pasados por el códec (16,9 min de audio en 14,6 min, RTF 0,86) y etiquetados como
reales; misma partición por clip, así que la ida y vuelta de un clip apartado también queda apartada.
32 épocas con parada temprana (28,8 min en la VM en modo taller).

| salida | fase 2 | fase 2b | puerta |
|---|---|---|---|
| **AUC ida y vuelta / clon** | 0,80 | **0,999** | > 0,7 · pasa |
| **AUC real / ida y vuelta** | 0,996 | **0,628** | < 0,75 · pasa |
| AUC real / clon | 1,000 | 1,000 | — |
| persona | 100 % | 100 % | pasa |
| R² tono · inclinación | 0,88 · 0,63 | 0,94 · 0,66 | |
| R² recorrido · desviación | 0,32 · 0,30 | 0,55 · 0,54 | |
| R² movimiento · microvariación | 0,05 · 0,16 | 0,46 · 0,46 | |
| R² rango de energía · armonicidad | 0,27 · 0,06 | 0,37 · 0,34 | |

El juez ya no se apoya en el códec (0,628 queda cerca del azar) y separa casi perfecto el audio real
pasado por el códec del clon: oye lo que pierde la GENERACIÓN. El estilo mejora en los 8 descriptores con
el doble de audio real. **Sirve de juez de personalidad para la fase 3.**

Salvedades: el real es conversación espontánea y el clon lee; parte de lo que separa puede ser leer
frente a conversar, que es justo lo que se quiere acercar. Validación con 33 clips reales de 3 personas
(Juan Pablo y Daniel Felipe tienen muy poca voz para apartar clips). Pesos e informe en
`/var/lib/taller/fase2b` de la VM, fuera del repo.

#### Prueba con frases que la persona nunca dijo (13-09-2026, `scripts/fase2_texto_nuevo.py`)

El juez solo había visto clones diciendo el mismo texto que el clip real. Seis frases nuevas (como mucho
Jaccard 0,21 de palabras con cualquier clip del dataset, casi todo palabras vacías) dichas por el clon de
Carlos, Laura y Liliana a través de voz-stream (motor de producción, sin modo taller), más el clon por
voz-stream diciendo los textos de los clips apartados como control del motor. Puerta fijada antes: AUC(real,
nuevo) > 0,7, AUC(mismo texto, nuevo) < 0,75 y persona por encima del azar.

| P(real) media | real | clon fase 1 (torch) | voz-stream, mismo texto | voz-stream, texto nuevo |
|---|---|---|---|---|
| Carlos | 0,998 (26) | 0,038 (25) | 0,504 (26) | 0,377 (6) |
| Liliana | 0,883 (7) | 0,012 (5) | 0,102 (7) | 0,258 (6) |
| Laura | 0,658 (1) | — | 0,771 (1) | 0,525 (6) |

- **Pasa**: AUC real/nuevo **0,980**; mismo texto/nuevo **0,560** (el texto no mueve el veredicto); persona
  acertada **100 %** en los cuatro grupos, también con las frases nuevas.
- **Hallazgo**: el juez se entrenó con clones de `decir.py` (torch) y los de voz-stream (OpenVINO int8) le
  parecen bastante más reales (AUC clon torch / voz-stream 0,082). Real frente a voz-stream sigue en 0,967,
  así que los separa, pero con menos margen. Parte de lo que aprendió es propio del motor torch. Para la
  fase 3, juzgar siempre con el mismo motor a los dos lados del A/B, o reentrenar el juez con clones de
  producción.
- Laura tiene un solo clip apartado: su fila no dice nada por sí sola.

### 3 · Adaptador de estilo en la difusión

Un adaptador pequeño (FiLM) sobre la condición de la cabeza de difusión, con el vector del codificador
de estilo. Se entrena con la pérdida de difusión sobre los latentes REALES de cada persona (su audio →
codificador acústico) y la condición del LM con su transcripción. El resto del modelo, congelado. Días
de CPU en la taller.

**Puerta** (banco A/B, criterio fijado antes de medir): para cada persona, clon con adaptador frente a
clon sin él — los descriptores de la fase 1 más cerca de los reales, identidad ECAPA contra su audio real
igual o mejor, UTMOS sin bajar más de 0,02 y WER sin subir más de 0,5 puntos. Si pasa: `convertir_difusion.py`
recibe el adaptador y el grafo fusionado lo lleva dentro.

#### Diseño concreto y puertas, fijados antes de medir (13-09-2026)

**Un adaptador por voz, no desde el vector de estilo.** Con cinco identidades, una red que lleve del vector
del codificador de estilo al ajuste no generaliza a una voz nueva: sería una tabla por persona con pasos
de más. Lo que sí se puede comprobar es si ajustar la difusión a UNA voz con su audio real la acerca a esa
voz. Si funciona, el producto es «afinar una voz clonada con unos minutos de su audio». El vector de estilo
sigue sirviendo de juez. `c' = c + rms(c)·(β + γ⊙ĉ + U·Vᵀ·ĉ)`, de rango 4, empieza siendo la identidad
(~9 000 parámetros por voz); en producción son constantes dentro del grafo de difusión.

**Datos** (`scripts/fase3_condiciones.py`): audio real → encoder comunitario → latentes escalados como los
ve la cabeza; `generate()` con el prefijo del clon y la transcripción, forzado a devolver el latente REAL
de cada fotograma y guardando la condición del LM que lo acompaña. Ni difusión ni decodificador: solo el LM.

**3a · pérdida** (`scripts/fase3_adaptador.py`). Comprobación previa: la cabeza base tiene que dar menos
pérdida con la condición de su fotograma que con la de otro al azar; si no, el forzado está mal y no se
entrena. Puerta: en los clips que apartó la fase 2b, para cada persona con al menos 5 (Carlos y Liliana), la
pérdida v con adaptador por debajo de la de la base con el IC 95 % por bootstrap sobre clips entero bajo 0,
con el mismo ruido a los dos lados. El paso de entrenamiento se elige con una parte interna del
entrenamiento, nunca con los apartados.

**3b · oído** (solo si pasa la 3a). Motor torch a los dos lados, cfg 3,0, 6 pasos, semillas 101 y 7; textos
de los clips apartados más las 6 frases nuevas. Por persona evaluable, diferencia emparejada adaptador −
base: juez P(real) con IC inferior > 0; distancia de los 8 descriptores al perfil real (en unidades de su
desviación) con IC superior < 0; ECAPA contra su audio real ≥ −0,005; UTMOS ≥ −0,02; WER sin subir más de
0,5 puntos. Pasa si se cumple todo en Carlos y en Liliana.

#### Resultado de la 3a (14-09-2026): NO PASA

Condiciones forzadas de los 173 clips en 21,5 min (x1,27 del audio). Las comprobaciones previas salieron
bien: con la condición de su fotograma la cabeza base pierde 0,626 y con una cruzada 1,117 (el forzado es
correcto), y la media del codificador (0,626) da menos pérdida que una muestra (0,646), así que la cabeza se
entrenó con la media. Entrenamiento de 1500 pasos en 39 min; la parte interna mejora −1,91 % en el paso 100
y a partir de ahí empeora hasta +1,41 % en el 1500: con tan pocos datos aprende lo que puede enseguida y
luego se sobreajusta. Se usa el paso 100.

| clips apartados | n | base | adaptador | diferencia | IC 95 % | puerta |
|---|---|---|---|---|---|---|
| Carlos | 26 | 0,6077 | 0,5924 | **−2,5 %** | [−0,0171, −0,0135] | pasa |
| Liliana | 7 | 0,5440 | 0,5389 | −0,9 % | [−0,0114, **+0,0023**] | no pasa |
| Laura | 1 | 0,1044 | 0,1203 | +15,2 % | — | no evaluable |

- **Carlos**, con 3 956 fotogramas de entrenamiento (8,8 min), mejora con un IC lejos de 0. **Liliana**, con
  780 (1,7 min), apunta en la misma dirección pero su IC cruza el 0: con 7 clips no se puede afirmar.
- Incluso donde funciona la mejora es pequeña: −2,5 % de pérdida de difusión, moviendo la condición un 29 %.
  No se sabe si eso se oye; es lo que respondería la 3b.
- No se pasa a la 3b: la puerta pedía Carlos y Liliana. Probar la 3b solo con Carlos sería mover la
  portería después de ver los números; si se hace, es otro experimento con su propia puerta.

Lectura: el adaptador aprende algo de la voz real con unos 9 min de audio y no con menos de 2. Para
Liliana (y cualquier voz con poco audio) el cuello es la cantidad de datos, no el método.

#### 3b solo con Carlos: experimento aparte, puerta fijada antes de medir (14-09-2026)

Pedido por el usuario tras ver la 3a. No rescata la fase 3: responde si el −2,5 % de pérdida de Carlos se
nota al generar. Mismo diseño que la 3b (torch a los dos lados, cfg 3,0, 6 pasos, semillas 101 y 7, sus
26 textos apartados más las 6 frases nuevas: 64 parejas) y los mismos umbrales, solo para Carlos: juez
P(real) con IC inferior > 0; distancia de los 8 descriptores al perfil real con IC superior < 0; ECAPA
contra su audio real ≥ −0,005; UTMOS ≥ −0,02; WER sin subir más de 0,5 puntos.

#### Resultado de la 3b con Carlos (14-09-2026): NO PASA, y el juez se deja engañar

64 parejas en 42 min en la VM (torch, RTF ~2,8); WER, UTMOS y ECAPA en el Mac con whisper large-v3.

| adaptador − base (n=64) | base | diferencia [IC 95 %] | puerta |
|---|---|---|---|
| juez P(real) | 0,087 | **+0,216** [+0,159, +0,273] | pasa |
| distancia por clip al perfil real | 0,806 | −0,070 [−0,152, +0,009] | no |
| ECAPA contra su audio real | 0,635 | −0,030 [−0,044, −0,016] | no |
| UTMOS | 3,411 | **−0,335** [−0,415, −0,254] | no |
| WER (puntos) | 4,5 | **+15,7** [+8,3, +23,8] | no |

Descriptores de media (distancia al perfil real de Carlos en desviaciones, base → adaptador): recorrido
0,73 → 0,14, desviación 0,70 → 0,09, movimiento 0,54 → 0,07, inclinación 0,37 → 0,00, microvariación
0,90 → 0,59; pero el tono medio se pasa (133,6 → 121,4 Hz con el real en 130,6: 0,22 → 0,69) y la
armonicidad no cambia. La WER sube igual con frases nuevas (0 → 19,8) que con los textos apartados (5,6 →
20,3), con clips que pasan de 0 a 93 % y uno a 200 %: el clon con adaptador se vuelve poco inteligible.

Lectura:

- **La melodía media sí se acerca a Carlos**, que era lo que faltaba en la fase 1, pero a costa de romper
  el habla. El adaptador mueve la condición un 29 % para bajar la pérdida un 2,5 %: al generar, la condición
  sale de la zona que la cabeza conoce y los latentes se degradan. La pérdida con forzado no avisa de eso,
  porque al entrenar la condición siempre viene del audio real y al generar viene de lo que el propio clon
  va diciendo (hipótesis, no medida).
- **El juez solo no vale como puerta**: sube +0,22 con un audio peor en todo lo demás. Aprendió «suena a
  grabación real» y un audio más sucio se le parece más. Siempre acompañado de WER, UTMOS e identidad.
- La puerta con varias métricas cumplió su función: sin ella, el +0,22 del juez habría parecido un éxito.

Queda para decidir: regularizar el adaptador (menos movimiento de la condición, entrenar con la condición
que sale de generar) o cambiar de palanca: pausas y ritmo, que la fase 1 señaló como la mayor diferencia y
son del LM, no de la difusión.

### 4 · Pausas y ritmo, sin tocar el modelo

Decidido por el usuario tras la 3b: ir a pausas y ritmo, cubriendo todas las frases y sin WER catastróficos.

**Lo que ya se sabía y lo que midió el sondeo** (14-09-2026, clips reales):

| | palabras/min | pausas ≥ 150 ms/min | mediana de pausa | palabras por pausa | palabras por signo |
|---|---|---|---|---|---|
| Carlos (131 clips) | 154 | 21,2 | 0,52 s (p90 1,12) | **7,2** | **7,2** |
| Liliana (33 clips) | 197 | 15,9 | 0,35 s (p90 0,45) | 12,4 | 8,5 |

Carlos pausa, de media, una vez por signo de puntuación; su clon hace la mitad de pausas (10/min) y va
más lento (4,6 frente a 5,2 sílabas/s). El modelo pausa en comas y puntos, pero se salta muchas, y nunca
pausa dentro de una frase (`voz_stream.py`, EL RESPIRO): la puntuación ya está, falta que la cumpla.
Liliana no difería de su clon en la fase 1: es el **control**, lo que se haga no puede estropearla.

**Variantes** (`scripts/fase4_pausas.py`), mismo texto y semilla por voz-stream (motor de producción,
cfg 3,0, 6 pasos, semillas 101 y 7):

- `base`: el texto tal cual.
- `trozos`: el texto partido en los signos donde toca pausar, una petición por trozo, unidos con pausas.
- `saltos`: una petición con `\n` en esos signos (la parada de fin de locución del modelo).
- `trozos_r`, `saltos_r`: lo mismo con la velocidad ajustada a la de la persona (WSOLA, `estirar.py`).

En las cuatro, cada pausa (racha callada ≥ 150 ms) pasa a durar lo que sale de la distribución de pausas
reales de la persona (0,15–1,2 s, sorteada con semilla por clip). Dónde pausar: en un signo si desde la
última pausa van al menos K palabras, con K calibrado en los clips de entrenamiento para igualar sus
palabras por pausa reales; los trozos de menos de 4 palabras se unen al siguiente. La velocidad se
calibra con 12 textos de entrenamiento (factor entre 0,85 y 1,20). Los clips apartados de la fase 2b
no se usan para calibrar nada.

**Puerta** (fijada antes de medir), cada variante contra `base`, en los textos apartados (hay audio real
del mismo texto) y en las 6 frases nuevas:

- **Carlos** (objetivo), clips apartados: la distancia de pausas/min a su clip real baja con el IC 95 %
  superior < 0; la distancia de sílabas/s no sube de media.
- **Liliana** (control), clips apartados: la distancia de pausas/min no sube más de 2 de media, ni la de
  sílabas/s más de 0,3.
- **Cobertura y WER**, en todos los clips de las dos personas: WER medio sin subir más de 0,5 puntos;
  **ningún clip con WER > 25 % si en la base tenía ≤ 10 %**; ningún clip con las palabras transcritas
  fuera de 0,85–1,15 veces las del texto si en la base estaba dentro (ni frases sin decir ni repetidas).
- **Calidad**: UTMOS medio ≥ −0,05 e identidad ECAPA contra su audio real ≥ −0,01.

Pasa la variante que cumpla todo; si pasa más de una, la de menor distancia de pausas en Carlos. Son
cuatro variantes contra la misma base: una que pase por los pelos se confirma con otras semillas antes
de llevarla a producción.

#### Resultado de la fase 4 (14-09-2026): NO PASA NINGUNA

431 peticiones a voz-stream en 30 min con la voz en marcha. K salió 1 para las dos (pausar en todos los
signos; aun así la regla da 17,5 palabras por pausa con Carlos frente a 7,4 reales: los clips tienen
pocos signos por dentro). Velocidad calibrada: Carlos 1,00 (trozos) y 1,05 (saltos); Liliana 1,08-1,09.

| clips apartados, media | pausas/min | sílabas/s |
|---|---|---|
| Carlos real | 22,5 | 5,19 |
| Carlos base → trozos / saltos / trozos_r / saltos_r | 17,7 → 20,8 / 22,0 / 21,0 / 22,2 | 4,49 → 5,18 / 4,94 / 5,18 / 5,16 |
| Liliana real | 17,3 | 5,92 |
| Liliana base → trozos / saltos / trozos_r / saltos_r | 13,9 → 21,9 / 26,6 / 23,5 / 26,1 | 5,84 → 5,17 / 5,02 / 5,55 / 5,47 |

- **Carlos, de media, sí**: pausas y velocidad quedan casi en las suyas, y la distancia de sílabas/s por clip
  baja con IC bajo 0 en trozos y trozos_r. Pero la de pausas por clip no (IC cruza 0: en clips de 5-10 s
  una pausa más o menos son ±6-12 por minuto), y `trozos` mete 2 clips catastróficos.
- **Liliana, no**: todas le sobran pausas (22-27 frente a 17) y su WER sube 7-12 puntos. El control hizo
  su trabajo: pausar en todos los signos no es «su» ritmo.
- **Por qué fallan, leyendo las transcripciones**:
  - `saltos`: el `\n` dispara el fin de locución y el modelo se COME el último tramo («No todos los», «Fui el
    sábado y la verdad me pareció carísimo…», «Íbamos ganando…»). Es exactamente lo que no se puede permitir.
  - `trozos`: un trozo corto con poco contexto hace que el modelo invente — en Liliana repite texto de la
    transcripción de SU PROPIA referencia de voz («…nosotros debemos dar a las entes pues mejore también») —,
    se salte palabras («de horas extras») o convierta la frase en pregunta («¿Para qué es información?»).
  - La base también tiene 8 de 90 clips con WER > 25 % (sobre todo Liliana y frases nuevas): no es todo
    de las variantes, pero la puerta solo cuenta los que la base decía bien.

#### Fase 4b: ritmo sin tocar el contenido, puerta fijada antes de medir (14-09-2026)

Lo que no rompe la cobertura es no cambiar lo que se genera. Variantes, todas desde la misma generación
que la base salvo `frases`:

- `forma`: la base con cada pausa ≥ 150 ms re-durada con la distribución real de la persona.
- `forma_r`: lo mismo con la velocidad de la persona (calibrada en 12 textos de entrenamiento).
- `frases`, `frases_r`: cortes SOLO en fin de frase (. ? !) y trozos de al menos 8 palabras, para no dar
  al modelo trozos cortos; luego pausas re-duradas (y velocidad).

**Semillas nuevas, 23 y 42**: las 101 y 7 ya se han mirado. Mismos textos (apartados y frases nuevas).

**Puerta**: la de la fase 4 (cobertura, catástrofes, WER, UTMOS, ECAPA y el control de Liliana) salvo el
criterio principal de Carlos, que pasa a ser el RITMO: |sílabas/s − real| por clip baja con IC 95 %
superior < 0. Las pausas por minuto se informan pero no deciden (`forma` no puede cambiar cuántas hay).
Pasa la que cumpla todo; si pasa más de una, la de menor distancia de sílabas/s en Carlos.

#### Resultado de la fase 4b (14-09-2026): NO PASA NINGUNA, pero ninguna rompe la cobertura

234 peticiones en 19 min con la voz en marcha. Velocidad calibrada: Carlos 0,93 (base) y 0,89 (frases),
Liliana 1,05 y 1,04.

| variante − base | Carlos: sílabas/s por clip | WER (puntos) | UTMOS | ECAPA | catástrofes / fuera de cobertura | Liliana |
|---|---|---|---|---|---|---|
| `forma` | **−0,15** [−0,28, −0,02] ✅ | **+0,95** [+0,18, +1,84] ❌ | −0,02 ✅ | +0,00 ✅ | 0 / 0 | todo ✅ |
| `forma_r` | −0,16 [−0,28, −0,04] ✅ | +0,78 ❌ | **−0,14** ❌ | +0,01 | 0 / 1 ❌ | UTMOS −0,13, ECAPA −0,04 ❌ |
| `frases` | −0,01 ❌ (se pasa: 5,44 frente a 5,19) | +0,98 ❌ | −0,03 ✅ | **+0,04** ✅ | 0 / 0 | todo ✅ |
| `frases_r` | −0,07 ❌ | +1,69 ❌ | **−0,25** ❌ | −0,01 | 0 / 2 ❌ | ECAPA −0,02 ❌ |

- **Cero catástrofes en las cuatro**, y `forma` y `frases` sin un solo clip fuera de cobertura: cortar solo
  en fin de frase con trozos ≥ 8 palabras quita los fallos de la fase 4.
- **`forma` falla solo por el WER medio de Carlos (+0,95 con tope +0,5)** y no pierde contenido: es la
  misma generación con las pausas re-duradas. 9 clips suben, 2 bajan y 53 quedan igual, y lo que cambia es
  cómo transcribe whisper la misma voz («familias»/«familiares», «roca»/«rock a», signos de pregunta, un «que…»
  final colgado). Comprobado: la cola tras la última pausa es idéntica muestra a muestra (0 LSB) en los
  clips donde whisper deja de oír ese «que». Aun así la puerta dice NO y no se reinterpreta.
- **La velocidad con WSOLA cuesta naturalidad** (UTMOS −0,14 a −0,25): descartada como palanca de ritmo.
- **El motor de producción ya está más cerca de Carlos de lo que decía la fase 1**: esa fase medía clones de
  torch con cfg 3,5 (10 pausas/min, 4,6 sílabas/s); por voz-stream con cfg 3,0 la base hace 16,9-17,7
  pausas/min y 4,5-4,95 sílabas/s según la semilla, frente a 22,5 y 5,19 reales.

Queda para decidir: confirmar `forma` con más semillas (el WER medio se estabiliza con más clips y la
puerta sería la misma) o dejar el ritmo donde está.

#### Fase 4c: confirmación de `forma`, puerta fijada antes de medir (14-09-2026)

Pedida por el usuario. Solo `base` y `forma`, **cuatro semillas nuevas (3, 11, 55 y 89)**, nunca usadas
en las fases 4 y 4b; mismos textos (26 apartados de Carlos y 7 de Liliana, más las 6 frases nuevas): 128
parejas de Carlos y 52 de Liliana. La calibración de pausas es la misma (clips de entrenamiento).

**Puerta: la de la 4b sin tocar** — Carlos, |sílabas/s − real| por clip baja con IC 95 % superior < 0;
Liliana, pausas/min no más de +2 y sílabas/s no más de +0,3; en todos los clips WER medio ≤ +0,5 puntos,
ningún clip con WER > 25 % si la base tenía ≤ 10 %, cobertura 0,85-1,15, UTMOS ≥ −0,05 y ECAPA ≥ −0,01.
Si pasa, `forma` se integra en voz-stream como ajuste de pausas por voz; si no, el ritmo se queda como está.
Los resultados de las semillas 23 y 42 no se suman: la confirmación se decide solo con las nuevas.

#### Resultado de la fase 4c (14-09-2026): NO PASA por un solo clip de cobertura

204 peticiones en 18 min con la voz en marcha; 128 parejas de Carlos y 52 de Liliana.

| `forma` − `base` | Carlos (104 apartados / 128 clips) | Liliana (28 / 52) |
|---|---|---|
| sílabas/s por clip, distancia al real | **−0,215 [−0,320, −0,111]** ✅ (4,70 → 5,14; real 5,19) | +0,03 ✅ |
| mediana de pausa, distancia al real | −0,118 [−0,214, −0,031] | +0,01 |
| pausas/min, distancia al real | −0,06 [−0,64, +0,59] (17,5 → 19,3; real 22,5) | −0,12 ✅ |
| WER (puntos) | **−0,07 [−0,82, +0,65]** ✅ | −0,11 ✅ |
| UTMOS | −0,004 ✅ | −0,009 ✅ |
| ECAPA | +0,007 ✅ | 0,000 ✅ |
| catástrofes | **0** ✅ | 0 ✅ |
| fuera de cobertura | **1** ❌ | 0 ✅ |

- Con cuatro semillas nuevas el WER de Carlos, que tumbó la 4b (+0,95), sale en −0,07: aquello era ruido de whisper.
- El único fallo es `carlos-segura__videoplayback-068__s11`. La base tenía una pausa de 2,48 s tras «De esta forma.»; `forma` la acorta y whisper transcribe desde «Y acá…» (cobertura 0,81 con mínimo 0,85). **Comprobado byte a byte**: los 6 tramos con voz de la base están idénticos y en el mismo orden dentro de `forma` (5,56 s de voz). Las palabras están; es whisper sin el contexto de la pausa larga. La base de Carlos tiene 9 de 128 clips fuera de cobertura: la medida de cobertura con whisper es inestable en fragmentos cortos.
- **La puerta dice NO y no se reinterpreta.** Llevar `forma` a producción es una decisión del usuario sabiendo esto: la puerta con whisper no puede certificar la cobertura de una transformación que, por construcción, no cambia la voz; la garantía de contenido de `forma` es estructural (tramos con voz idénticos), y esa sería la prueba que tendría que llevar en el servicio.

## Riesgos

- **Sobreajuste a Carlos**: 76 % de los datos. Muestreo equilibrado por persona y validación por grabación.
- **Poca voz para tres personas** (< 1,2 min): sirven para validar, no para aprender su estilo.
- **Coste en tiempo real**: el adaptador tiene que caber en los ~30 ms libres por fotograma; se mide con
  `/crono` antes de desplegar nada.
