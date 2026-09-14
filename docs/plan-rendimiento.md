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

Se fijan aquí antes de medir cada una, según lo que dé la fase 0. La puerta estándar del banco
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

**0.2b LM de texto (M)**, 4 capas torch int8, ventana de 5 tokens: 8,5 / 8,1 / 9,2 ms con 50 / 200 / 500
tokens de contexto.
