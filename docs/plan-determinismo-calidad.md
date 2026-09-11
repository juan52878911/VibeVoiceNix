# Plan: determinismo y calidad de voz-stream, medido en la VM

Fijado el 10 de septiembre de 2026, en la rama `claude/audio-optimization-quality-d6d895`. Este
documento se escribió **antes** de tocar código y se completa al final con la sección
[Resultados](#resultados). Lo que aquí es hipótesis se marca como tal; lo medido lleva la cifra al
lado, como en [optimizacion.md](optimizacion.md).

---

## Por qué

Las palancas grandes de rendimiento ya están medidas y agotadas (ver
[optimizacion.md](optimizacion.md): la CPU va limitada por cómputo, sin VNNI, y lo que pagaba
—int8, pocos pasos, grafos compilados, depthwise a mano, solapar el decodificador— está hecho). Al
releer el servicio entero con esa lupa, lo que queda son huecos de **determinismo** y de
**medición de calidad**:

| # | Hueco | Dónde |
|---|---|---|
| 1 | **Bug de concurrencia.** Al reanudar una sesión tras una pausa solo se reponen el RNG y el candado. Otra generate() que se cuele en la pausa (otra sesión, o `/tts/stream`) deja pisados los pasos de difusión, `neg_cada` (que además las sesiones nunca fijan), el contador de la guía de arranque y `_REMATE`. Rompe «misma semilla = mismo audio» y puede meter la rampa de cfg 4,5 en mitad de una frase | `voz_stream.py`: `SesionViva._pausar/_reanudar`, `_ajustar_pasos`, `_ajustar_neg_cada`, `reforzar_guia_arranque` |
| 2 | El banco `scripts/fidelidad.py` no manda semilla ni pasos: no hay A/B emparejado, cada comparación se hace contra otro ruido | `scripts/fidelidad.py` |
| 3 | No hay métrica de naturalidad: pasos, freno y cfg se deciden por WER o a oído, y whisper entiende perfectamente una voz horrible | bancos |
| 4 | Sin medir si 8 o 10 pasos suenan mejor que 6. Con 6, la última evaluación de la red es en t=166 (σ≈0,28) y el solver salta a cero; con 8 es t=125, con 10 t=100 | `VIBEVOICE_PASOS` |
| 5 | La API sortea el ruido si el cliente no manda semilla; no hay semilla por defecto del servicio (`scripts/perfiles.py` sí fija 11) | `PeticionTTS`, `PeticionSesion`, `AbrirSesionWS` |
| 6 | El CLI `vibevoice` y dos documentos siguen con cfg 1,5 cuando lo medido y la API dicen 3,0 | `nix/modules/vibevoice.nix`, `opciones.md`, `arquitectura.md` |

### Descartado antes de empezar, para que nadie lo repita

| Idea | Por qué parecía buena | Qué pasó de verdad |
|---|---|---|
| Cabeza de difusión en fp32 (hoy int8) | quitar el residuo de deriva que se midió en OpenVINO | son 42 M parámetros × 6 pasadas × lote 2 por fotograma, limitada por memoria: 168 MB frente a 42 MB por pasada, ~+45 ms por fotograma de 133. Descartado por cuenta |
| Sigmas de Karras en el solver | mejor reparto de pasos a pocos pasos | con el schedule coseno del modelo los timesteps salen degenerados: `999, 999, 998, 993, 837, 0`. `trailing` da lo mismo que `linspace`. Medido en el planificador |
| Acelerar el WSOLA de `estirar.py` | bucle en Python puro | cuesta 0,01 s por segundo de audio. No es cuello |
| Redondear en vez de truncar al pasar a PCM16 | 0,5 LSB de error frente a 0,25 | sin sesgo DC y −96 dBFS: inaudible. No merece el cambio |

---

## Dónde se ejecuta

- **VM `voz`** (VMID 210 en Proxmox 192.168.2.100; 192.168.2.54): motor OpenVINO, el numérico de
  producción. Todas las medidas se hacen aquí, nunca en el Mac (MPS fp16 no es el mismo camino).
- **CT `docker-sandbox`** (192.168.2.62, 6 GB): la imagen `voz-stream:local` al final, para
  comprobar que arranca y que el arreglo también vale en el motor torch. Necesita 8 GB mientras
  dure; la VM `voz` y el CT no coinciden encendidos.
- **Mac**: solo los clientes de banco y la puntuación de naturalidad (segundos).
- AuraCRM (VM 200) no se toca. `app-noticias` (CT 100) se puede apagar mientras dure la imagen.

---

## Fases

### 0 · Fijar el plan

Este documento, la nota gemela en la bóveda de Obsidian, y la rama publicada e indexada en
UltraMemory (`um branch`).

### A · La sesión se lleva puesto TODO su estado de generate

- El contador de la guía de arranque sale de la closure a un dict de módulo (`_ARRANQUE`), como ya
  está `_REMATE`.
- `foto_generacion()` / `reponer_generacion(foto, pasos, neg_cada)`: la lista única de estado
  global que arrastra una generate() (RNG, contador de arranque, remate) más lo que la sesión
  conoce (pasos, `neg_cada`). La foto se hace **antes** de soltar el candado del modelo y se repone
  **después** de recuperarlo, igual que ya se hacía con el RNG.
- `SesionViva` acepta `neg_cada` (por HTTP y websocket) y lo fija al arrancar cada generate().
- **Prueba nueva `pausa` en `scripts/ws_fidelidad.py`**, que tiene que **fallar antes y pasar
  después**: la sesión A abre con una frase corta y se queda parada antes de su primer fotograma
  (espera la ventana de adelanto); la intrusa B entra con `pasos + 4` y otra semilla y genera al
  menos 6 fotogramas; A sigue. `md5(A)` tiene que ser el de A a solas y `md5(B)` el de B a solas.
  Variante `pausa-stream` con `/tts/stream` de intrusa (`pasos + 4`, `neg_cada 2`).

### B · Semilla por defecto del servicio

`VIBEVOICE_SEMILLA` (vacía = sorteo, como hoy) como defecto de `semilla` en las tres peticiones;
`/health` la anuncia como `semilla_defecto`; opción `services.voz-stream.semilla` (null por
defecto) y variable en `docker/compose.yaml`. Un `"semilla": null` explícito sigue sorteando.
Activarla en producción es una decisión que sale del banco, no de este cambio.

### C · cfg 3,0 en el CLI y los docs

`services.vibevoice.cfgScale` 1,5 → 3,0 con la tabla de fidelidad en la descripción;
`opciones.md`, `arquitectura.md` y `sesiones_fidelidad.py --cfg` alineados.

### D · Banco emparejado y métrica de naturalidad

- `fidelidad.py --semillas 11,7,3,23,42,101 --pasos N --csv --crono`: un WAV por (frase, semilla),
  fila CSV por clip (semilla, pasos, cfg, duración, tiempo, RTF, WER, exacto, oído), y el reparto
  de `/crono` guardado.
- `scripts/naturalidad.py puntuar --dir D`: UTMOS22 (strong) por clip a 16 kHz, columna `utmos`
  unida al CSV. `comparar --base D6 --contra D8 D10`: tabla emparejada por (frase, semilla) con
  ΔUTMOS medio y «mejora en k/36». UTMOS solo se compara por diferencia dentro del banco, nunca
  por valor absoluto.

### E · Despliegue y verificación en la VM

Arrancar la VM; **evidencia del fallo** con el código desplegado (`ws_fidelidad.py --pruebas
pausa` → FALLO); commit; `nixos-rebuild switch --flake .#voz --target-host root@192.168.2.54
--build-host root@192.168.2.54`; huella del código en `journalctl -u voz-stream | grep codigo`
igual al sha256 local; suite completa de `ws_fidelidad.py` en verde.

### F · Banco 6/8/10 pasos

- Fijado antes de medir: las 6 frases de `fidelidad.py`, semillas `11,7,3,23,42,101`, cfg 3,0,
  voz `sp-Spk1_man`, 36 clips por valor de pasos, nada más corriendo en la VM.
- Métricas: WER medio/peor, exactos/36, UTMOS medio/mín/p10, RTF medio, ms por fotograma de
  `/crono` (generate, cabeza).
- **Umbral de decisión, escrito antes de ver un solo número:** se adopta 8 si ΔUTMOS medio ≥
  +0,10 con mejora en ≥ 24/36 clips, el WER medio no empeora más de 1 punto y RTF₈ ≤ 1,10 × RTF₆.
  10 solo se evalúa si 8 cumple, con el mismo umbral respecto a 8. Si no se cumple, se queda 6 y la
  medida va a «Lo que NO funcionó».

### G · Imagen Docker en docker-sandbox

VM `voz` apagada mientras dure; CT a 8 GB; `docker --context crm-remote compose --profile pesado
build voz-stream` desde la raíz del repo; misma huella de código en los logs; `/health` con
`motor: torch-int8`; `ws_fidelidad.py --pruebas fidelidad,concurrencia,pausa` en verde;
`fidelidad.py --semillas 11` con WER en línea con el histórico. Motores distintos dan md5
distinto: la VM y el contenedor **no** se comparan entre sí. Al terminar, todo como estaba y la
VM `voz` encendida.

### H · Documentar

Resultados a [optimizacion.md](optimizacion.md) («Lo que funcionó» con Hipótesis → Escepticismo →
Resultado; «Lo que NO funcionó» con lo descartado), [opciones.md](opciones.md) y
[api.md](api.md) para la semilla y `neg_cada`; y la sección de abajo.

---

## Verificación de punta a punta

1. `pausa` contra la VM: FALLO antes del despliegue, OK después; suite completa OK.
2. Huella del código igual al sha256 local, en la VM y en el contenedor.
3. `fidelidad.py --semillas 11 --pasos 6` dos veces: mismos md5 clip a clip.
4. Tabla 6/8/10 con las tres métricas y la decisión aplicada según el umbral de arriba.
5. `/health` de la VM en verde al cerrar; `app-noticias` y la VM encendidas.

## Reversión

`nixos-rebuild switch --rollback` en la VM, o `git revert` y redesplegar. Docker: `compose down` y
la imagen anterior.

---

## Resultados

Todo medido en la VM `voz` (openvino, 6 hilos) el 10 de septiembre de 2026, salvo la puntuación
UTMOS, que corre en el Mac sobre los WAV bajados de la VM.

### A · El bug, antes y después

Prueba `pausa` de `ws_fidelidad.py` (`sp-Spk3_man`, cfg 4,5):

| | audio | md5 |
|---|---|---|
| A a solas (6 pasos, semilla 11) | 6,93 s | `e3721afb…` |
| A con B (10 pasos, semilla 12) en su pausa, código `898a33c7` (**antes**) | 7,20 s | `d37167ad…` — distinta desde el byte 560, el primer fotograma |
| A con `/tts/stream` (10 pasos, `neg_cada` 2) en su pausa (**antes**) | 7,20 s | `871bdfae…` |
| las dos, código `ec7239c1` (**después**) | 6,93 s | `e3721afb…` = a solas |

La suite entera de `ws_fidelidad.py` (fidelidad, eventos, concurrencia, pausa, pausa-stream,
respiro, corte, errores, auth) sale «todo correcto» con el código nuevo. Huella en la VM
`ec7239c1c29a` (4024 líneas), igual que el fichero local.

### B, C · Semilla por defecto y cfg 3,0

`/health` anuncia `semilla_defecto: null` (sorteo, como siempre). El CLI y los docs ya dicen 3,0.

### D · El banco es determinista

`fidelidad.py --semillas 11 --pasos 6` repetido: 6/6 md5 iguales. Y el banco entero de 6 pasos
repetido de principio a fin: **36/36 md5 iguales**. Las diferencias que siguen son del cambio, no
del sorteo.

### F · 6, 8 y 10 pasos: se queda 6

Las 6 frases × semillas 11, 7, 3, 23, 42, 101; cfg 3,0; `sp-Spk1_man`. Umbral fijado antes de
medir: 8 si ΔUTMOS ≥ +0,10 con mejora en ≥ 24/36, WER medio sin empeorar > 1 punto, RTF₈ ≤ 1,10 ×
RTF₆.

| pasos | WER medio | WER peor | exactos | UTMOS medio | UTMOS p10 | ΔUTMOS | mejora en | RTF | ms/fot |
|---|---|---|---|---|---|---|---|---|---|
| **6** | 11,8 % | 83,3 % | 23/36 | 3,547 | 3,126 | — | — | **1,057** | 125,1 |
| 8 | 11,2 % | 211,1 % | 27/36 | 3,592 | 3,205 | +0,046 | 22/36 | 1,081 | 128,8 |
| 10 | 10,7 % | 122,2 % | 23/36 | 3,523 | 3,116 | −0,024 | 19/36 | 1,137 | 134,2 |

8 no llega al umbral (+0,046 y 22/36) y además produce la peor alucinación del banco (211 %); 10
baja el UTMOS. El coste por paso es el previsto: 2,5-2,6 ms de cabeza por fotograma. **Decisión: 6
pasos.** La primera tanda de 6 pasos dio RTF 1,267 por correr recién reiniciado el servicio (páginas
del modelo en swap); repetida dio 1,057 con los mismos 36 md5. El RTF de la primera tanda tras un
despliegue no vale.

### G · La imagen Docker, construida y probada en el CT docker-sandbox

La primera construcción (`docker compose --profile pesado build voz-stream`, contexto remoto, 6
núcleos, CT ampliado a 52 GB de disco y 8 GB de RAM mientras duró) **arrancó y murió en la primera
línea**: `ModuleNotFoundError: No module named 'estirar'`, en bucle de reinicio. El Dockerfile
copiaba `voz_stream.py` y `prueba.html` pero no `estirar.py`, que `voz_stream.py` importa desde que
existe la velocidad sin mover el tono; la derivación de Nix sí lo copiaba y por eso en la VM nunca
se vio. **La imagen llevaba rota desde ese commit.** Arreglado en `docker/Dockerfile.voz-stream`.

Con el arreglo: huella `ec7239c1c29a` en los logs (la misma que la VM y el fichero local),
`/health` con `motor: torch-int8`, `semilla_defecto: null`, 1847 MB residentes tras cargar, 2,2 GB
en `docker stats`. La suite de `ws_fidelidad.py` (fidelidad, concurrencia, pausa, pausa-stream,
corte, errores; voz `sp-Spk1_man`, la imagen solo trae las oficiales) sale «todo correcto»: el
arreglo del estado por sesión vale también en el motor torch. Seis clips con semilla 11 y 6 pasos,
transcritos en el Mac con faster-whisper small: WER medio 7,4 %, 4/6 exactos, frente a 6,1 % y 4/6
de la misma semilla en la VM (motores distintos, así que el md5 no se compara). RTF en el CT: 1,7 a
2,0, en línea con los 2,19 históricos del camino torch.

### La semilla pesa más que los pasos

WER medio / frases exactas por semilla (6 frases cada celda), en los tres bancos:

| pasos | s3 | s7 | s11 | s23 | s42 | s101 |
|---|---|---|---|---|---|---|
| 6 | 13 % / 4 | 2 % / 5 | 6 % / 4 | 8 % / 3 | **41 % / 1** | **0 % / 6** |
| 8 | 11 % / 3 | 0 % / 6 | 0 % / 6 | 43 % / 4 | 13 % / 2 | **0 % / 6** |
| 10 | 13 % / 3 | 2 % / 5 | 4 % / 4 | 9 % / 4 | **36 % / 1** | **0 % / 6** |

La 101 acierta las 6 frases con cualquier número de pasos y es de las que mejor suenan; la 42
falla en todas las tandas. El WER medio del banco de pasos (11,8 %) frente al 3,6 % histórico se
explica por eso: aquí las semillas están fijas y dos de las seis son malas.

**18 semillas a 6 pasos** (las 6 de arriba más 12: 1, 2, 5, 13, 17, 19, 29, 31, 37, 57, 77, 99;
mismas 6 frases, `sp-Spk1_man`, cfg 3,0, VM; 108 clips, WER medio global 8,2 %), ordenadas por WER
y, a igualdad, por UTMOS:

| semilla | WER medio | WER peor | exactos | UTMOS medio | UTMOS mín |
|---|---|---|---|---|---|
| **101** | **0,0 %** | 0 % | **6/6** | **3,689** | 3,384 |
| **17** | **0,0 %** | 0 % | **6/6** | 3,628 | 3,195 |
| 2 | 2,4 % | 14 % | 5/6 | 3,715 | 3,226 |
| 19 | 2,4 % | 14 % | 5/6 | 3,615 | 3,212 |
| 7 | 2,4 % | 14 % | 5/6 | 3,465 | 2,651 |
| 57 | 2,4 % | 14 % | 5/6 | 3,177 | 2,794 |
| 31 | 4,2 % | 14 % | 4/6 | 3,357 | 2,520 |
| 29 | 4,8 % | 29 % | 5/6 | 3,473 | 3,230 |
| 1 | 5,2 % | 17 % | 4/6 | 3,516 | 2,840 |
| 11 | 6,1 % | 22 % | 4/6 | 3,706 | 3,249 |
| 13 | 6,1 % | 22 % | 4/6 | 3,664 | 3,172 |
| 77 | 6,1 % | 22 % | 4/6 | 3,536 | 3,251 |
| 5 | 6,5 % | 17 % | 3/6 | 3,455 | 3,153 |
| 99 | 6,6 % | 29 % | 4/6 | 3,397 | 3,151 |
| 23 | 8,3 % | 22 % | 3/6 | 3,566 | 3,109 |
| 3 | 13,0 % | 67 % | 4/6 | 3,398 | 2,830 |
| 37 | 29,6 % | 78 % | 3/6 | 3,528 | 3,163 |
| 42 | 40,7 % | 83 % | 1/6 | 3,457 | 3,126 |

Entre la mejor y la peor semilla hay 40 puntos de WER **con el mismo modelo, la misma voz y los
mismos pasos**; entre 6 y 10 pasos, uno. Con el sorteo por petición (lo de hoy), una de cada seis
peticiones cae en una semilla como la 3, la 37 o la 42. Fijar `services.voz-stream.semilla = 101`
para `sp-Spk1_man` quitaría esa lotería a cambio de que el mismo texto suene siempre igual; el 11
que usan `perfiles.py` y los bancos históricos está a mitad de tabla (6,1 %). Seis frases son pocas
para elegir entre 101 y 17, y una semilla buena para una voz no tiene por qué serlo para otra: la
decisión de fijarla, y con qué frases ampliar el banco antes, es de Juan. **No se ha cambiado el
defecto** (`null`, sorteo), como decía el plan.

### Qué queda

- Decidir la semilla por defecto (arriba). Si se fija, repetir el banco de 18 semillas con las
  frases de la voz que se use en producción y con la voz clonada, si la hay.
- El WER de la frase «El uso de memoria bajó un veinticuatro por ciento» (17 % de media) y de «No
  hay incidencias que reportar en las últimas horas» (29 %) es del modelo en esas frases, no del
  ruido: son las dos que fallan con casi cualquier semilla.
