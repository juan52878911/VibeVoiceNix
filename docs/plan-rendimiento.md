# Plan de RTF, memoria y disco: puertas fijadas antes de medir

Plan completo y estado de partida: nota de Obsidian
`01-proyectos/06-vibevoicenix/plan-rtf-memoria-almacenamiento-2026-09-14.md`. Este fichero guarda
**solo las puertas**, escritas el 14-09-2026 **antes de ver un número**, y después los resultados.
Una puerta que no pasa no se reinterpreta: se anota y se cierra la palanca.

La calidad es la restricción dura. **(M)** = medido; **(E)** = estimado.

---

## Fase 0: medir (sin tocar producción)

Nada de esto cambia el servicio. Las cifras deciden si se abren las fases 2 y 3.

| # | Qué | Cómo | Umbral que decide |
|---|---|---|---|
| 0.1 | RSS y VmHWM antes y después de cada paso de la carga, incluido cada `compile_model` | `scripts/lab_fase0.py carga --modo viejo` en la VM, con voz-stream parado | informativo (dimensiona A1) |
| 0.2 | Pasada del LM TTS aislada a 6, 3 y 2 hilos, con la caché en 500 y 1500 posiciones; y el par condicional + negativa en paralelo (2 streams × 3 hilos) frente a dos pasadas seguidas a 6 hilos | `scripts/lab_fase0.py lm` | **B2 se abre solo si**, en las dos longitudes: pasada a 3 hilos ≤ 1,6 × la de 6 hilos **y** el par en paralelo ≤ 0,80 × dos pasadas seguidas a 6 hilos (es la misma cifra, 1,6/2, medida sobre el mecanismo real). Si falla en cualquiera de las dos longitudes, B2 se cierra |
| 0.2b | `forward_lm` (LM de texto, 4 capas torch int8) por ventana | `scripts/lab_fase0.py lm_texto` | informativo (C5) |
| 0.3 | iGPU UHD 630 en un LXC con `/dev/dri` (sin passthrough, sin tocar la VM 200): decodificador int8 y fp16 y LM int4 con el plugin GPU de OpenVINO; ms por fotograma, RAPL, temperatura | LXC de laboratorio en pve | **C1 y C2 se cierran si no se cumplen las dos:** decodificador ≤ 90 ms por fotograma en GPU (mediana de 200 llamadas tras calentar, en int8 o fp16) **y** con la GPU al 100 % la frecuencia media de los núcleos del host baja ≤ 15 % frente a la GPU en reposo, con la misma carga de CPU en la VM |
| 0.4 | Temperatura, frecuencia y potencia del paquete durante un banco normal | `scripts/vigilar_host.sh` en pve mientras corre `scripts/banco_md5.py` | informativo: dice si la varianza de base (1,13-1,27) es térmica |

### 0.2 repetida: regla de agregación (fijada el 14-09-2026 tras una primera tanda no concluyente)

La primera tanda (2 rondas) no se usa para decidir: la pasada a 6 hilos con 1500 posiciones dio 20,1 ms
en una ronda y 41,5 ms en la otra, y los dos pares en paralelo se invirtieron de una ronda a otra,
mientras arrancaba AuraCRM en el host. La puerta de B2 no decía cómo combinar rondas, así que se
fija ahora, **antes** de repetir y sin mirar qué resultado da cada regla:

- 6 rondas completas y alternas, cada una con todas las configuraciones (6, 3 y 2 hilos, par con 2
  streams y par con 2 modelos) en las dos longitudes, y voz-stream parado.
- Por configuración y longitud se toma la **mediana de las 6 rondas**. Los cocientes se calculan
  con esas medianas.
- Se usa el mejor de los dos montajes del par. B2 se abre si en **las dos** longitudes t3/t6 ≤ 1,6 **y**
  par/(2·t6) ≤ 0,80.
- **La medida no vale**, y B2 queda sin decidir, si el recorrido intercuartílico de t6 supera el 25 %
  de su mediana; en ese caso se anota y se repite con el host en reposo.

## Fase 1: bit a bit (A1 + A2 + A4 en código, A5 y A3 por Nix y a mano)

**Qué entra.**
- **A1**: el modelo se construye en `meta` y solo se leen del safetensors los tensores que siguen vivos
  tras soltar el LM TTS, el decodificador, el codificador y la cabeza torch.
- **A2**: la tabla de embeddings del LM de texto se sirve por mmap del safetensors en bf16 y se pasa a
  fp32 al consultarla (bf16 → fp32 es exacto, y consultar y luego convertir da lo mismo que convertir y
  luego consultar).
- **A4**: `CabezaOV` se compila en su primera llamada; con la difusión en un grafo no se llama nunca.
- **A5**: whisper a 4 hilos, con `Nice` y `CPUWeight` bajos.
- **A3**: poda de IR y cachés que no usa nadie (guion `scripts/podar_disco.sh`, en seco por defecto).

**Puerta A1 + A2 + A4 (todas):**
1. **Huella de tensores idéntica**: para cada tensor vivo del modelo torch tras la carga (nombre, forma,
   dtype, sha256 de los bytes; los `Linear` cuantizados por su `int_repr`, escala y punto cero),
   carga vieja = carga nueva. Es la prueba previa: si falla, no se sigue.
2. **md5 idéntico** del audio de las 8 frases de `scripts/banco_md5.py` con semilla 101, en todas las
   rondas, entre la base (código de producción) y la variante.
3. **`scripts/ws_fidelidad.py` completo** en verde contra la variante.
4. **RTF**: tandas alternas base → variante → base → variante, cada una un proceso nuevo con voz-stream
   parado; la primera ronda de cada proceso no cuenta. Pasa si la mediana del RTF de la variante
   ≤ 1,02 × la de la base. Una mejora de más del 2 % se anota pero no se atribuye al cambio sin repetir.
   Cada tanda, base y variante por igual, devuelve el swap a RAM en cuanto el servidor responde, igual
   que hace `voz-stream-sin-swap` en producción (añadido antes de la primera cifra: el primer intento
   se paró al ver que la base arrancaba con 1,8 GB de pico de swap y la variante no lo tendría).
5. **Memoria** (objetivo, no puerta de calidad): VmHWM y RSS tras el calentamiento, base y variante. Si
   el pico no baja al menos 1 GB, A1 no aporta lo que prometía y se documenta así.

**Puerta A5:** las transcripciones de whisper-server con 4 hilos son **idénticas** a las de 6 hilos sobre
los 8 audios del banco md5 (texto exacto). El RTF del STT se anota.

**Puerta A3:** tras podar, `/health` lista los mismos IR que antes y el md5 de las 8 frases no cambia.
Se conservan `decoder_mm_int8` (producción), `decoder_mm_int4` (control del banco A/B),
`decoder_mm_fp16` (C1 y referencia de SNR), `tts_lm_estado_int4` (producción), `tts_lm_estado_int8` (C3),
`tts_lm_estado_fp16` y `cabeza_fp16` (fuentes de `comprimir()`), `cabeza_int8` y `difusion_p6_int8`.

## Fases 2-4

Se fijan aquí antes de medir cada una, según lo que dé la fase 0.

### Fase 2: B1, topología real de la VM voz (puerta fijada el 14-09-2026, antes de medir)

B2 queda cerrada por la 0.2. Sigue B1: decirle a la VM 210 que sus 12 vCPU son 6 núcleos × 2 hilos
(hoy el guest ve 12 núcleos físicos).
- **Hipótesis:** OpenVINO repartiría sus 6 hilos en núcleos distintos en vez de en hermanos SMT, con
  menos varianza y quizá mejor RTF.
- **Protocolo:** 4 tandas alternas, base → B1 → base → B1. Cada una arranca la VM desde cero con la
  topología de esa tanda, espera a que `voz-stream-sin-swap` termine y corre `banco_md5.py` a
  3 rondas. La ronda 0 no cuenta.
- **Condiciones de host:** CT 100/101/102 parados y AuraCRM (VM 200 + CT 203) en marcha, igual en
  todas las tandas.
- **Puerta, todas a la vez:**
  1. md5 idéntico a la base en todas las rondas;
  2. `ws_fidelidad.py` completo en verde con B1;
  3. **RTF**: mediana de las rondas válidas de B1 ≤ **0,97 ×** la de la base (un 3 % mínimo, por
     encima del ruido medido entre tandas de la fase 1, del 0,6 %), **o** el IQR relativo del RTF de
     B1 ≤ la mitad del de la base (la otra promesa de B1: menos varianza).
- **Si no pasa:** se vuelve a la configuración de antes y B1 se cierra.

**Resultado (20:08-20:19, `scripts/fase2_b1.sh`): B1 NO PASA y se cierra.** Dentro de la VM, las tandas B1
vieron de verdad 6 núcleos × 2 hilos (`Thread(s) per core: 2`, hermanos 0-1), y QEMU recibió el
segundo `-smp`.

| | base | B1 |
|---|---|---|
| md5 frente a la base | — | **idéntico en todo** |
| `ws_fidelidad.py` | — | todo correcto |
| RTF de las rondas válidas | 0,9517 · 0,8886 · 0,8978 · 0,8973 | 0,9044 · 0,9013 · 0,9079 · 0,8823 |
| mediana / IQR | **0,8976** / 0,1 % | **0,9028** / 0,3 % |
| puerta | — | B1/base = **1,0059** (> 0,97) y IQR sin bajar a la mitad |

La VM 210 quedó sin `args`, como antes. Decirle al guest que tiene hermanos SMT no cambia cómo reparte
OpenVINO sus 6 hilos, o no lo cambia para bien.

### Fase 3: C1, el decodificador en la iGPU (puerta fijada el 15-09-2026, antes de medir)

Juan da la fase 3. La 0.3 cumplió sus dos umbrales (46 ms en GPU, frecuencia −12 %), pero la misma carga de
CPU iba +51 % más lenta con la GPU al 100 %, y eso solo lo decide el banco.

**Montaje.**
1. **La iGPU a la VM:** se desliga de `i915` en el host **en caliente** (`unbind` + `vfio-pci`), sin
   reiniciar pve. AuraCRM no se para. Si hiciera falta reiniciar el host, se para y se pregunta a Juan.
   Después, `hostpci0: 0000:00:02.0` en la VM 210.
2. **En NixOS:** runtime OpenCL de Gen9 (`intel-compute-runtime-legacy1` o equivalente).
3. **En `motor.AcusticoOV`:** dispositivo configurable por entorno, siempre con
   `INFERENCE_PRECISION_HINT=f32` (el f16 da 16 dB y está descartado). El decodificador va en su propio
   hilo con cola FIFO (el solapado ya existente), para que la CPU siga con LM y difusión mientras la GPU
   decodifica.
4. **El IR:** `decoder_mm_int8` (en GPU con f32 da 53-58 dB frente a CPU). El fp16 queda fuera: 106 ms
   por fotograma en f32.

**Puerta, todas a la vez.** Variante = decodificador en GPU asíncrono; base = producción.
1. **Numérica del sumidero** (`banco_md5.py`, 8 frases con semilla 101): mismas duraciones exactas que la
   base en todos los clips. El decodificador no realimenta, así que un largo distinto sería un fallo. Y
   SNR del audio frente a la base ≥ **25 dB** en todos los clips.
2. **`ws_fidelidad.py` completo** en verde. Se exige lo que no depende del md5 (eventos, concurrencia,
   limpieza, errores, autenticación), y en las pruebas de md5 que la variante sea coherente consigo
   misma (sesión = stream = websocket).
3. **Banco exhaustivo** (`banco_ab.py`, 238 parejas, control int4 incluido) con la puerta estándar:
   - UTMOS con el IC inferior ≥ −0,02;
   - WER con el IC superior ≤ +0,5 puntos;
   - identidad ±0,005 global y ±0,0023 por clon;
   - tono medio y recorrido ±0,03 st;
   - final del habla ±20 ms.
4. **RTF**, que es el motivo de la fase: tandas alternas base → GPU → base → GPU (`banco_md5.py`, 3
   rondas, la 0 no cuenta). Pasa si la mediana GPU ≤ **0,93 ×** la base (un 7 % mínimo, por encima del
   ruido medido y a la altura de lo que la 0.3 deja esperar), con RAPL y temperatura registrados.

**Si no pasa:** la iGPU vuelve al host (`vfio-pci` → `i915`), la VM 210 se queda sin `hostpci0` y C1 se
cierra.

**Diario del montaje (15-09-2026).**
- **13:25, despliegue** de `f855136` (driver `intel-compute-runtime-legacy1`): md5 de las 8 frases
  idéntico y `ws_fidelidad` correcto.
- **13:31, iGPU pasada en caliente** con `fase3_igpu_host.sh pasar`: `vfio-pci` y `hostpci0` en la VM 210.
  pve respondió en todos los pasos y AuraCRM no se tocó. En la VM, `i915` inicializa la UHD 630 y
  OpenVINO lista `GPU`.
- **13:33, primera ejecución de `fase3_ab.sh`: `gpu-1` NO ARRANCÓ.** El proceso murió al compilar el
  decodificador en GPU con `free(): invalid size` y `*** longjmp causes uninitialized stack frame ***`,
  sin nada en el dmesg del guest.
- **No se reproduce**, en procesos separados dentro de la VM:
  - el decodificador en GPU con OpenVINO solo (47-50 ms), tras `import torch` y tras `import numba,
    llvmlite`;
  - LM en CPU y decodificador en GPU en el mismo proceso, con y sin `MALLOC_ARENA_MAX=2` y
    `OMP_NUM_THREADS=6`;
  - `motor.cargar` completo con `VIBEVOICE_ACUSTICO_DISPOSITIVO=GPU`;
  - y dos arranques del servidor real con la receta exacta de `gpu-1` (modelo listo en 10,4 y 10,5 s).
- **13:40, segunda ejecución de `fase3_ab.sh`: `gpu-1` vuelve a morir en el mismo punto.** No era
  intermitente.
- **Causa CONFIRMADA (13:45):** `systemd-run` no pone `HOME`, y el runtime OpenCL de Intel (NEO/IGC) lo
  necesita para su caché de kernels. El servidor de laboratorio arrancado a mano con `env -u HOME` muere
  exactamente igual (`longjmp causes uninitialized stack frame`); con `HOME=/root` arranca siempre. Mis
  pruebas buenas por ssh tenían `HOME`. Arreglo: `fase3_ab.sh` exporta `HOME=/root`.
- **Condición añadida al despliegue, si la fase 3 pasara:** la unidad de voz-stream (`DynamicUser`)
  tiene que llevar un `HOME` o una caché explícita para NEO, y hay que comprobar 10 arranques seguidos
  con la GPU sin fallos.
- **Resultado (13:45-13:54, tercera ejecución, con `HOME`): C1 NO PASA y se cierra.**

  | puerta | exigido | medido | |
  |---|---|---|---|
  | duraciones | idénticas | **idénticas en los 24 clips** | ✅ |
  | SNR del audio frente a la base | ≥ 25 dB | **mín. 64,9 dB**, mediana 66,6 dB | ✅ |
  | `ws_fidelidad.py` | en verde | todo correcto | ✅ |
  | RTF | GPU ≤ 0,93 × base | base 0,9838 · 0,9311 · 0,9107 · 0,9268 (mediana **0,9289**); GPU 0,8692 · 0,8687 · 0,8749 · 0,8756 (mediana **0,8720**): **0,9387×** | ❌ |

  **Reparto por fotograma** (`/crono`, ronda 2):

  | | LM TTS | cabeza | decodificador |
  |---|---|---|---|
  | base | ~47 ms | ~16 ms | ~37 ms (CPU) |
  | GPU | **~72 ms** | ~18 ms | ~61 ms (GPU, solapado) |

  El decodificador sale de la CPU, pero el LM y la cabeza se frenan por lo mismo que medía la 0.3
  (+51 % con la GPU al 100 %). Queda una ganancia neta del 6,1 %, bajo el 7 % exigido. El banco
  exhaustivo no se corre, porque la puerta 4 ya no pasa. El pico de memoria sí baja (VmHWM 1950 →
  1600 MB), pero no era la puerta.
- **Qué queda:** la iGPU vuelve al host (`fase3_igpu_host.sh devolver`) y la VM 210 queda sin
  `hostpci0`. `hardware.graphics` con `intel-compute-runtime-legacy1` sigue en `nix/configuration.nix`
  sin efecto, hasta decidir si se quita.
- **Ruido de host en esa primera tanda:** la base dio RTF 1,16-1,18 con el `kvm` de AuraCRM a ~236 %
  en un pico. Las tandas alternas lo reparten entre base y GPU, pero la cifra absoluta no se compara con
  la de otros días.

### Fase 4: C3, calidad del LM int8 frente a int4 (puerta fijada el 14-09-2026, antes de medir)

`tts_lm_estado_int8.xml` nunca ha pasado por el banco exhaustivo. La pregunta es si el int4 de producción
pierde calidad frente al int8. Con 6 hilos, el int8 costaba ~0,13 de RTF.
- **Corpus:** `scripts/banco_ab.py` completo, igual que en §7 y §8 de optimizacion.md: 7 voces (4
  clones + 3 de serie) × 17 frases × 2 semillas, cfg 3,5, generado por un proceso de voz-stream con
  `VIBEVOICE_IR_LM` apuntando al int8, frente a producción (int4). Control incluido: el decodificador
  int4, un cambio de timbre conocido que el banco tiene que separar.
- **Puntuación:** whisper large-v3, UTMOS, ECAPA y F0, en el nodo `ascci` si su CPU lo permite en un
  tiempo razonable; si no, en el Mac.
- **Se considera que el int8 SUENA MEJOR si**, frente al int4, se cumple a la vez:
  1. UTMOS con la diferencia media ≥ +0,02 y el IC 95 % inferior > 0;
  2. WER sin empeorar: IC superior ≤ +0,5 puntos;
  3. identidad, global y por clon, sin bajar más de 0,0023.
- **Si el int8 suena mejor:** se prueba int4 con AWQ y estimación de escala (`nncf`, con datos de
  latentes reales), con el mismo banco y la misma puerta contra el int4 de hoy.
- **Si no suena mejor** (UTMOS con el IC conteniendo el 0, o por debajo): el int4 se queda, C3 se
  cierra y `tts_lm_estado_int8` pasa a la lista de poda.
- **Válido solo si** el control int4 del decodificador sale separado de la base, como en §7 (UTMOS con
  el IC superior < 0).
- **Entorno de puntuación**, anotado antes de puntuar: CT 103 `banco-lotes` en `ascci` (i3-3220, sin
  AVX2), Python 3.11, con las versiones del venv del Mac del 13-09 (torch 2.13.0 CPU, torchaudio 2.11.0,
  faster-whisper 1.2.1, ctranslate2 4.8.2, speechbrain 1.1.1, librosa 0.11.0, numpy 2.3.5) salvo
  **scipy 1.16.3** (la 1.18.0 exige Python ≥ 3.12). Las cifras de la fase 4 no se comparan con las
  del 13-09: base, int8 y control se puntúan los tres aquí, en el mismo entorno.
- **Cambio de máquina, antes de puntuar ninguna comparación (20:53):** en el i3 los primeros 20 clips de
  la base tardaron **29 min** (~87 s por clip, sin AVX2), así que las tres variantes serían ~17 h. No es
  un tiempo razonable. El Mac era la alternativa escrita, pero queda para tareas cortas y la otra sesión
  lo está usando. La puntuación pasa al **LXC 204 de pve (i7-8700T, AVX2)** con el **mismo** entorno,
  copiado tal cual del CT 103: venv `/opt/banco`, versiones y cachés de modelos. Las tres variantes se
  puntúan allí. La base a medias de `ascci` no entra en ninguna comparación.
- **Base:** si 17 clips de producción de hoy (andres, semilla 101) coinciden en PCM con el corpus
  `difusion` del 13-09, ese corpus es la base; si no, se regenera entera. **Medido (20:21): 17/17
  idénticos.** Producción no ha cambiado un bit del audio desde el 13-09 (difusión en un grafo, fase 1 y
  `forma` apagado incluidos), así que el corpus `difusion` es la base.
- **Si pasa:** B1 se queda en la configuración de la VM y `nucleos_fisicos()` deja de contar 12. La puerta estándar del banco
(`scripts/banco_ab.py`) es la del plan: UTMOS con IC inferior ≥ −0,02; WER con IC superior ≤ +0,5
puntos; identidad ±0,005 global y ±0,0023 por clon; tono medio y recorrido ±0,03 st; final del habla
±20 ms; control int4 del decodificador incluido.

---

## Resultados

### Fase 0 (14-09-2026, VM voz recién reiniciada, voz-stream parado)

**0.1 Memoria de la carga (M)**, `lab_fase0.py carga`, MB:

| paso | carga vieja RSS | pico | carga nueva RSS | pico |
|---|---|---|---|---|
| torch + openvino importados | 238 | 238 | 237 | 237 |
| from_pretrained fp32 entero / pesos vivos leídos | 3016 | **4321** | 728 | 844 |
| soltar LM TTS y decodificador | 1370 | 4321 | — | — |
| LM de texto en int8 (+ malloc_trim en la vieja) | 962 | 4321 | 830 | 844 |
| compile LM TTS int4 | 1046 | 4321 | 868 | 868 |
| compile cabeza int8 | 1049 | 4321 | (perezosa) | — |
| compile decodificador int8 | 1742 | 4321 | 1342 | 1342 |
| compile difusión p6 int8 | 1753 | 4321 | 1363 | 1363 |
| malloc_trim final | **1610** (anon 1196) | 4321 | **1355** (anon 852) | **1361** |

- **Repacks de OpenVINO (d):** el decodificador int8 añade ~700 MB, 360 anónimos y 340 de fichero; el LM
  TTS ~85 MB; la difusión ~12 MB; la cabeza ~3 MB.
- **Segunda medida, con la huella:** pico 4477 → 1366 MB.

**Huella de A1+A2 (M): IDÉNTICA**, 65 de 65 tensores (forma, dtype, sha256; Linear int8 por
`int_repr` y escalas; embeddings comparados como fp32). Pasa la prueba previa.

**0.2 LM TTS aislado: primera tanda NO CONCLUYENTE** (2 rondas, con AuraCRM arrancando en el host). ms por
pasada de 1 token:

| ronda | posiciones | 6 h | 3 h | 2 h | par 2 streams | par 2 modelos |
|---|---|---|---|---|---|---|
| 0 | 500 | 14,4 | 18,0 | 21,0 | 62,8 | 35,8 |
| 0 | 1500 | 20,1 | 24,5 | 27,9 | 85,9 | 46,9 |
| 1 | 500 | 15,7 | 18,4 | 24,9 | 33,3 | 68,3 |
| 1 | 1500 | **41,5** | 38,0 | 36,2 | 43,6 | 50,7 |

t6 a 1500 posiciones se dobla de una ronda a otra y los pares se invierten: se repite con la regla de
agregación fijada arriba. Lo que sí se repite en las cuatro filas es t3/t6 = 0,92-1,26 (≤ 1,6).

**0.2 repetida (19:55-20:00, `scripts/fase0_lm.sh`, voz-stream parado y CT 100/101/102 parados):
VÁLIDA y B2 SE CIERRA.** Medianas de 6 rondas (M), en ms por pasada:

| posiciones | 6 h | 3 h | 2 h | par 2 streams | par 2 modelos | IQR t6 | t3/t6 | mejor par/(2·t6) |
|---|---|---|---|---|---|---|---|---|
| 500 | 16,73 | 19,07 | 22,14 | **30,95** | 37,05 | 12 % ✓ | 1,140 ✓ | **0,925 ✗** |
| 1500 | 21,03 | 23,94 | 27,89 | **43,01** | 50,24 | 4 % ✓ | 1,138 ✓ | **1,022 ✗** |

- **Por qué no gana:** una pasada a 3 hilos solo cuesta un 14 % más que a 6, pero dos a la vez no
  caben en los 6 núcleos sin estorbarse. El par con 2 streams ahorra un 7,5 % con 500 posiciones y
  pierde con 1500; con dos modelos compilados es peor en las dos.
- **Qué pasa con B2:** necesitaba ≤ 0,80 en las dos longitudes y **se cierra**. Es la misma pared
  que tumbó el solapado del decodificador: en esta CPU dos etapas de cómputo a la vez se estorban.
- **La caché KV en sesiones largas (C4):** 500 → 1500 posiciones cuestan +4,3 ms por pasada (+26 %).

**0.3 iGPU, primera pasada PRELIMINAR** (LXC 204 con `/dev/dri`, OpenVINO 2025.4.1, `intel-opencl-icd`
22.43 de Debian 12; host sin reposo, con la VM voz desplegando; no decide la puerta):

| IR en GPU | resultado |
|---|---|
| `decoder_mm_int8` | **41,9 ms** mediana (p90 44,6), compila en 6,2 s, cálculo en f16 |
| `decoder_mm_fp16` | 52,1 ms (p90 55,8) |
| `difusion_p6_int8` | no compila: el plugin GPU no acepta sus formas dinámicas (`to_shape was called on a dynamic shape`) |
| `tts_lm_estado_int4` | no compila: falta un kernel OpenCL en este runtime para Gen9 (`intel_sub_group_block_read`) |

- **SNR del decodificador, GPU frente a CPU** con el mismo IR y los mismos latentes aleatorios:
  **~16 dB** con amplitud realista (int8 15,9 y fp16 15,9, RMS 0,022). Con latentes pequeños la
  salida es casi muda y la cifra no significa nada (4,7 dB con un RMS de 1e-6).
- **Lectura:** 16 dB es del orden del error del int8 frente al fp16 (16,9 dB) y queda lejos de los
  25 dB que pide C1. La sospecha era el cálculo en f16 que la GPU usa por defecto, y se confirma:
- **Forzando `INFERENCE_PRECISION_HINT=f32` en la GPU:**

  | IR | ms mediana (p90) | SNR GPU frente a CPU |
  |---|---|---|
  | `decoder_mm_int8` | **57,4** (60,0), compila en 10,5 s | **52,9 dB** (escala 2) · **57,8 dB** (escala 5, RMS 0,022) |
  | `decoder_mm_fp16` | 105,8 (112,6): **fuera del umbral de 90 ms** | sin medir: cargado a la vez en CPU y GPU no cabe en los 3 GB del LXC (la GPU usa la RAM) y se paró al ver 728 % de CPU del host |

  **El int8 en f32 es numéricamente el mismo decodificador** (dif. máx. 2e-3 sobre picos de ~0,2) y queda
  bajo los 90 ms. En f16 se descarta: 16 dB. Esto cumple la primera mitad del umbral, **preliminar**
  mientras no se repita con el host en reposo.
- **Falta** la segunda mitad del umbral, la frecuencia de la CPU con la GPU al 100 %.
- **Prueba controlada de frecuencia, primera pasada (19:00, `scripts/fase0_igpu_frecuencia.sh`): NO
  VÁLIDA.** Salió una caída de frecuencia del 38 % (2582 → 1600 MHz) y la carga de CPU pasó de 83,7 a
  135,3 ms por llamada (+62 %). Pero la condición de la prueba, la misma carga de CPU, no se dio:
  - **carga inestable sin la GPU:** esa llamada va de 35 a 92 ms segundo a segundo;
  - **paquete lejos del PL1:** 9-25 W;
  - **frecuencia a saltos:** entre 800 y 3860 MHz;
  - **host compartido:** tras el reinicio corrían CT 101 minecraft, CT 102 gym y CT 203 docker-sandbox,
    parados antes del reinicio, con carga media de 6,5.

  La VM no recibía tiempo de CPU. **C1 y C2 quedan sin decidir; no se cierran ni se abren con esto.**
- **Regla de validez para repetirla**, fijada ahora y antes de repetir: la prueba vale solo si el
  recorrido intercuartílico de los ms por llamada de la carga de CPU en la ventana sin GPU es ≤ 25 %
  de su mediana (la misma regla que la 0.2). Si no vale, se anota y se repite con el host en reposo.
  Umbral sin cambios: caída de frecuencia ≤ 15 %.
- **Segunda pasada (20:00, con CT 100/101/102 parados y AuraCRM, es decir VM 200 y CT 203, en marcha): VÁLIDA y
  CUMPLE.**
  - **Validez:** IQR de la carga de CPU sin GPU = **19 %** de su mediana (≤ 25 %).
  - **Umbral:** la frecuencia media cae un **12,1 %** (2194 → 1928 MHz; con medianas, 2591 → 2336), ≤ 15 %.
  - **Lo que el umbral no mide:** la misma carga de CPU (decodificador int8 a 6 hilos en la VM) pasa de
    **79,8 a 121,2 ms** de mediana por llamada con la GPU encendida, un **+51 %**, y su IQR sube del
    19 % al 66 %.
  - **La iGPU bajo esa carga:** 99,7 ms de mediana (cuartiles 64,7 / 114,2).
  - **Potencia:** el paquete solo marca 13,5-14,5 W, así que la frenada no la pone el PL1. Parece que
    los hilos de la VM y el trabajo de la GPU se esperan entre sí.

- **Primer umbral con el host en reposo (20:05, dos pasadas):** decodificador int8 en f32 en GPU
  **46,8 y 46,3 ms** de mediana (p90 49,5 / 48,8), ≤ 90 ms.
- **0.3 COMPLETA: cumple los dos umbrales, así que C1 queda abierta.** Nada se decide sin el banco de la
  fase 3, que necesita pasar la iGPU a la VM (ventana del host, a decidir por Juan). La difusión y el
  LM no compilan en GPU con el runtime 22.43, así que **C2 se cierra** con ese runtime.

  **Lectura (E):** si el decodificador (~41 ms de 110) sale de la CPU pero el resto se frena la mitad,
  el fotograma queda en ~(110 − 41) × 1,5 ≈ 104 ms. La ganancia neta sería de un ~5 % o nula. Solo el
  banco de la fase 3 lo diría.
- **Host durante esa pasada (M, sin carga controlada, 900 s):** la iGPU tira de **9,9 W** de uncore de
  media cuando trabaja (pico 12,5 W). El paquete sube a 34,7 W, justo el PL1 de 35 W, frente a 22,6 W
  sin GPU. Máximo 86 °C, sin estrangulamiento térmico. La frecuencia de esos segundos no vale para el
  umbral, porque la carga de CPU no era la misma con y sin GPU.

**0.4 Temperatura y potencia durante un banco (M)**, `vigilar_host.sh` en pve mientras corría el A/B de la
fase 1 (18:00:42-18:11:10, 608 s):

| minuto | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MHz medios | 3733 | 3192 | 3351 | 3165 | 3215 | 3262 | 3558 | 3212 | 3456 | 3212 | 3445 |
| W paquete | 27,8 | 36,7 | 30,8 | 35,8 | 35,0 | 33,5 | 31,1 | 35,3 | 31,5 | 35,2 | 28,3 |
| °C | 72,8 | 77,7 | 75,4 | 77,4 | 77,4 | 77,3 | 75,6 | 77,7 | 76,8 | 77,5 | 76,0 |

- **Temperatura:** media 76,5 °C, máxima 90 °C, **sin estrangulamiento térmico** (el contador del
  paquete no se movió).
- **Potencia:** paquete a 33,0 W de media, con **438 de 608 s a ≥ 34 W**, es decir, en el PL1 de 35 W.
- **Frecuencia:** baja justo en los minutos de más vatios.
- **Conclusión:** la varianza de base la pone **el límite de potencia, no la temperatura**. Cualquier
  consumidor nuevo (iGPU, más hilos) sale de esos 35 W.

### Fase 4: C3, LM int8 frente a int4 — el int8 NO suena mejor, C3 se cierra (15-09-2026, 00:10)

- **Corpus:** generado en la VM (`scripts/fase4_c3_vm.sh`). La base es el corpus `difusion` del 13-09,
  17/17 idéntico en PCM a producción de hoy.
- **Puntuación:** `banco_ab.py` en el LXC 204 de pve, 3 h 5 min para las tres variantes. El informe
  completo está en [bancos/2026-09-15-lm-int8.md](bancos/2026-09-15-lm-int8.md).

**Validez del banco:** el control (decodificador int4) da UTMOS **−0,041 [−0,048, −0,033]**, con el IC
superior < 0. **El banco vale.** Reproduce el control del 13-09 (−0,043) con otro entorno y otra
máquina.

| puerta del int8 frente al int4 | exigido | medido | |
|---|---|---|---|
| UTMOS | media ≥ +0,02 **e** IC inferior > 0 | +0,031 [**−0,006**, +0,070] | ✗ |
| WER | IC superior ≤ +0,5 puntos | +0,058 [−1,514, **+2,007**] (3,21 → 3,27 %) | ✗ |
| identidad global y por clon | sin bajar más de 0,0023 | global +0,007; andres +0,019 · isis +0,011 · juan −0,0008 · santiago +0,012 | ✓ |

- **Veredicto:** el int8 no suena mejor que el int4, así que **el int4 de producción se queda, no se
  prueba AWQ y `tts_lm_estado_int8` pasa a la lista de poda**.
- **Coste que se ahorra:** con el int8, el RTF de la mediana del banco pasaría de 0,885 a 0,964 (+9 %).
- **Por qué no valen las medidas de forma de onda:** SNR −2,6 dB, MCD 104 dB y 189 de 238 clips
  desplazados no dicen nada aquí. El LM decide el camino de la locución, así que con otro redondeo
  el clip es otra lectura igual de válida (206/238 transcripciones idénticas). Mandan WER, UTMOS e
  identidad, como con la difusión en un grafo.
- **Lo único con IC fuera del 0:** el tono medio sube **+0,74 st [+0,51, +0,97]**, en todas las voces.
  No es mejor ni peor; es otro timbre de lectura, y no compensa ni el RTF ni el WER.

### Fase 1: A1 + A2 + A4 — PASA (14-09-2026, 18:00-18:11)

`scripts/fase1_ab.sh` en la VM voz: cuatro procesos nuevos alternos en 127.0.0.1:8092, con el entorno,
el python y el `voz-stream.py` de producción en marcha (commit `8c34be8`, `sha256:b0b0eaabf4e7`). La
variante es el mismo código OpenVINO con **solo `motor.py` cambiado**, y el sha de origen se comprobó
antes de empezar. `banco_md5.py`: 8 frases con semilla 101, 3 rondas por tanda; cada tanda devuelve el
swap a RAM al arrancar.

| puerta | exigido | medido | |
|---|---|---|---|
| huella de tensores | idéntica | 65/65 idénticos | ✅ |
| md5 | idéntico en todo | **8/8 frases × 12 rondas × 4 tandas idénticos** | ✅ |
| `ws_fidelidad.py` completo contra la variante | todo correcto | **25 OK, «todo correcto»** (md5 de referencia `2a978a26…`, el de producción) | ✅ |
| RTF, mediana de las rondas válidas | variante ≤ 1,02 × base | base **0,9490** (0,9488 · 0,9493 · 0,9525 · 0,9387) · variante **0,9435** (0,9265 · 0,9388 · 0,9481 · 0,9529) | ✅ |
| memoria (objetivo) | pico −1 GB | VmHWM **4326 / 4503 → 1952 / 1945 MB** (−2,4 GB) · RSS tras el banco 2168 / 2166 → 1841 / 1831 MB (−330 MB) | ✅ |

Además (M): la variante arranca en 12,8-12,9 s frente a 22,4-25,5 s de la base, y no toca el swap.
La base empuja 860 MB al swap en cada arranque, y en producción eso lo arregla `voz-stream-sin-swap`.
Reparto por fotograma en las rondas válidas: igual en las dos, con LM TTS ~48 ms, cabeza ~16 ms,
decodificador ~37 ms y resto ~6,9 ms. La mejora del 0,6 % de RTF queda dentro del ruido y no se
atribuye al cambio.

### Fase 1 desplegada (14-09-2026, ~18:45)

`scripts/desplegar_vm_voz.sh` con el commit `b67ee49` (`origin/main` `de927bd` + A1+A2+A4 + A5):
`git archive` a la VM y `nixos-rebuild switch --flake path:…#voz`. Verificación en producción (M):

| | medido |
|---|---|
| `motor.py` desplegado | = HEAD (`2a7b754b…`) |
| VmHWM de voz-stream tras arrancar | **1867 MB**; tras el banco 1942 MB (antes 4326-4503) |
| RSS / swap | 1818-1828 MB / **0** |
| md5 de las 8 frases, 2 rondas, frente a la base del A/B | **idéntico en todo** |
| `ws_fidelidad.py` completo | **todo correcto** |
| RTF de la ronda válida | 0,9422 (base 0,9490) |

### Fase 1: A5, whisper a 4 hilos — PASA (14-09-2026, 18:30-18:38)

`scripts/fase1_whisper.sh`:
- **Montaje:** las 8 frases de `banco_md5.py`, generadas por producción con semilla 101, se
  transcribieron 3 veces con whisper-server de producción (6 hilos, :8081) y 3 veces con una segunda
  instancia del mismo binario y modelo a 4 hilos (:8091).
- **Resultado:** **8/8 transcripciones idénticas** byte a byte entre 6 y 4 hilos, y ninguna variación
  entre repeticiones de un mismo lado. Whisper mantiene sus propios errores en los dos casos («¡H4!» por
  «Hecho.», «a una máxima» por «ahora mismo»).
- **Tiempo medio por transcripción:** 9,89 s a 6 hilos y 4,25 s a 4. **No se atribuye a los hilos**,
  porque son procesos distintos y el orden fue fijo (primero producción, en marcha desde hace horas,
  y después la instancia nueva).
- **Cambio en `nix/configuration.nix`:** `hilos = 4` y `prioridadBaja = true`, es decir, `Nice 10` y
  `CPUWeight 20`.

### Fase 1: A3, poda de disco (ejecutada por Juan, 14-09-2026 ~18:25)

`scripts/podar_disco.sh --borrar` en la VM voz (M):
- **Disco:** `/` pasa de 19 a **17 GB** usados de 39.
- **IR que quedan en `/var/lib/voz/ov`:** `cabeza_fp16`, `cabeza_int8`, `decoder_mm_fp16`,
  `decoder_mm_int4`, `decoder_mm_int8`, `difusion_p6_int8` y `tts_lm_estado_{fp16,int4,int8}`, justo lo
  que la puerta mandaba conservar.
- **`/health`:** lista los mismos cuatro IR de producción que antes.
- **md5 tras la poda:** las 8 frases generadas por producción a las 18:30, ya podada la VM, dan
  **8/8 md5 idénticos** a la base del A/B de la fase 1. **A3 PASA.**

**0.2b LM de texto (M)**, 4 capas torch int8, ventana de 5 tokens: 8,5 / 8,1 / 9,2 ms con 50 / 200 / 500
tokens de contexto.
