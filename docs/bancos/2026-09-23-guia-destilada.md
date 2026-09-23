# Guía destilada: menos RTF quitando la rama negativa del CFG (2026-09-23, EN CURSO)

**Estado: pausado a mitad.**
- La primera corrida (guia1) **no pasa** por identidad.
- La segunda (guia2, con memoria acústica) está entrenada, pero su evaluación generando no llegó a terminar: se paró para viajar.

## Idea

Cada fotograma pasa dos veces por el backbone TTS: la rama con texto y la negativa (`<|image_pad|>`) del CFG. La cabeza de difusión calcula las dos ramas en cada paso del solver.

Se entrena una copia de la cabeza (el alumno) para que dé, con la condición POSITIVA sola, lo que hoy da la guía completa de producción: cfg 3,0 + freno 0,75. Así la rama negativa sobra.
- Los 6 primeros fotogramas siguen con el maestro, porque la rampa de arranque (cfg 4,5) arregla la primera palabra.
- El backbone positivo no cambia, así que no hay que refabricar voces.

## Qué se construyó

- `scripts/lora/condiciones.py`: condiciones positiva y negativa por fotograma, forzadas sobre CML-TTS/LibriTTS-R (en, es, de, fr).
  - 2.437 ejemplos, 183.000 fotogramas.
- `scripts/lora/destilar_guia.py`: entrenamiento del alumno.
  - Entrena en trayectorias del solver (6 pasos) del maestro y del propio alumno, más latentes reales con ruido.
  - Con `--memoria` recibe los últimos latentes generados.
- `scripts/lora/evaluar.py --cabeza ... --freno 0.75 --desde 6`: evaluación como en producción (maestro en la rampa y alumno después).
- `scripts/lora/escala_guia.py`: una vara para leer el error.
- Producción, apagado por defecto (`VIBEVOICE_GUIA_DESTILADA`):
  - `convertir_difusion.py` con `VIBEVOICE_CABEZA_GUIA` genera `difusion_guia_p6_int8.xml`.
  - `motor.DifusionGuiaOV` y `TtsLmOV.saltar_negativa`.
  - `voz_stream.py` cambia al alumno pasada la rampa y salta la pasada negativa.

## Resultados

**Destilación, en validación** (error relativo del final del solver frente a producción, mismo ruido):

| | Error |
|---|---|
| Sin guía (cfg 1) | 0,165-0,175 |
| guia1 (solo la condición positiva, 3000 pasos, 33 min) | **0,085** |
| guia2 (con memoria de 6 latentes + media, desde el fotograma 6, 4000 pasos) | **0,090** (otra validación: solo fotogramas ≥ 6) |

**Vara** (`escala_guia.py`, 4.096 fotogramas; mide cuánto se aleja del resultado de producción):

| Cambio | Error |
|---|---|
| cfg 2,5 | 0,010 |
| cfg 3,5 | 0,007 |
| cfg 4,5 | 0,036 |
| Freno apagado | 0,084 |
| guia1 | 0,075 |

El alumno se desvía tanto como apagar el freno. La meta sería ≲ 0,03, el orden de la rampa.

**guia1 generando** (192 pares frente al maestro, en/es/de/fr + Juan, Carlos y Liliana):

| | Diferencia | Puerta |
|---|---|---|
| WER | +4 % (0,033 → 0,035), IC [−0,011; +0,014] | ✓ |
| UTMOS | −0,012, IC [−0,060; +0,033] | ✗ (IC inferior ≥ −0,02) |
| **ECAPA** | **−0,045**, IC [−0,055; −0,037]; clones −0,03 a −0,15 | ✗ |

Lectura: la rama negativa solo ve el habla ya generada y es lo que sostiene la voz. Sin ella, la identidad cae. De ahí sale guia2.

**Velocidad en la VM** (i7-8700T, 6 hilos): el grafo de difusión pasa de 11,5 ms a 10,3 ms por fotograma (solo el grafo).
- El ahorro grande es la pasada negativa del backbone (~20 ms de ~118). Sin medir de punta a punta: pide un segundo voz-stream y en la RAM de la VM solo cabe parando el de producción.
- Estimado: RTF 0,885 → ~0,72.

## Para retomar

1. **Evaluar guia2 generando.**
   - Tiene `~/Documents/mejora-modelo/guia/gpu/guia2/cabeza_mejor.pt`, y del maestro ya existen `eval/maestro/medidas.json` y `lote.json`.
   - En una g4dn, ~30 min: `evaluar.py --datos datos --voces voces.json --salida eval/alumno2 --cabeza guia2/cabeza_mejor.pt --cfg 3.0 --freno 0.75 --desde 6`, después `juez_lote.py` y `comparar.py` frente a `eval/maestro`.
   - Los datos y las condiciones están bajados: no hay que regenerarlos.
2. **Si no pasa, las siguientes palancas:**
   - Una memoria mayor (un pequeño transformer causal sobre toda la historia).
   - Entrenar más con lr 5e-5.
   - Guía parcial: rama negativa cada 2 fotogramas con el maestro.
3. **Si pasa:**
   - Convertir en la VM: `VIBEVOICE_CABEZA_GUIA=... python convertir_difusion.py`.
   - Medir el RTF de punta a punta con el servicio parado unos minutos.
   - Pasar `banco_ab.py` y `ws_fidelidad`.
   - Añadir la opción al módulo Nix.

## Otras medidas del mismo día

**Acento nativo en el servidor** (kNN-VC en la VM, i7-8700T; Python y torch del servicio, torchaudio sustituido para el import):

| Hilos | Tramos de 3 s | Tramo de 24 s | Banco de 1 min | Banco de 3 min | Banco de 11,6 min | Pico de RAM |
|---|---|---|---|---|---|---|
| 1 | RTF 0,79 | RTF 1,01 | 18 s | 48 s | 192 s | 3,0 GB |
| 6 | **RTF 0,77** | **RTF 0,70** | 27 s | 36 s | 131 s | 3,0 GB |
| 12 | RTF 1,13 | RTF 0,71 | 16 s | 39 s | 150 s | 3,0 GB |

Las medidas con 1 y 12 hilos coincidieron con una construcción de Nix que ocupaba 2 núcleos.
- El acento suma ~0,7 de RTF a la síntesis: en el M4 era 0,095, así que el i7 es ~8 veces más lento.
- Apenas escala con hilos: conviene convertir segmentos en paralelo en procesos de 1-2 hilos.

**También queda en esta rama:**
- Normalizador de texto en voz-stream: `normalizar`, por defecto «auto».
- Referencia corta ×2 por defecto en `clonar_voz.py`.
- `num2words` en el entorno Nix: construido en la VM, no desplegado.
