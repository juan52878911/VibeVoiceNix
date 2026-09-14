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

### 3 · Adaptador de estilo en la difusión

Un adaptador pequeño (FiLM) sobre la condición de la cabeza de difusión, con el vector del codificador
de estilo. Se entrena con la pérdida de difusión sobre los latentes REALES de cada persona (su audio →
codificador acústico) y la condición del LM con su transcripción. El resto del modelo, congelado. Días
de CPU en la taller.

**Puerta** (banco A/B, criterio fijado antes de medir): para cada persona, clon con adaptador frente a
clon sin él — los descriptores de la fase 1 más cerca de los reales, identidad ECAPA contra su audio real
igual o mejor, UTMOS sin bajar más de 0,02 y WER sin subir más de 0,5 puntos. Si pasa: `convertir_difusion.py`
recibe el adaptador y el grafo fusionado lo lleva dentro.

## Riesgos

- **Sobreajuste a Carlos**: 76 % de los datos. Muestreo equilibrado por persona y validación por grabación.
- **Poca voz para tres personas** (< 1,2 min): sirven para validar, no para aprender su estilo.
- **Coste en tiempo real**: el adaptador tiene que caber en los ~30 ms libres por fotograma; se mide con
  `/crono` antes de desplegar nada.
