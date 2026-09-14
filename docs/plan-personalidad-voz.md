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

## Riesgos

- **Sobreajuste a Carlos**: 76 % de los datos. Muestreo equilibrado por persona y validación por grabación.
- **Poca voz para tres personas** (< 1,2 min): sirven para validar, no para aprender su estilo.
- **Coste en tiempo real**: el adaptador tiene que caber en los ~30 ms libres por fotograma; se mide con
  `/crono` antes de desplegar nada.
