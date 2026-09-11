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

*Pendiente: se rellena al terminar cada fase, con las cifras.*
