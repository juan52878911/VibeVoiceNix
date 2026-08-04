# Plan de optimización de VibeVoice en CPU

**Objetivo:** bajar el RTF de VibeVoice-Realtime-0.5B todo lo posible en un
i7-8700T, con vistas a conversación en tiempo real.

**Estado de partida:** RTF 4,04 (6 hilos anclados, 20 pasos de difusión).
Para conversar hace falta RTF < 1; para que se sienta fluido, < 0,5.

---

## Hechos ya medidos — no volver a probarlos

Cada uno costó una medición real. Repetirlos es tiempo perdido.

| Hecho | Evidencia |
|---|---|
| **Limitado por ancho de banda de memoria, no por cómputo** | La CPU sola alcanza 17,2 GB/s de los 21,3 teóricos = **80,7%**, el techo práctico de la DDR4 |
| **Más hilos empeoran** | 2 hilos RTF 4,19 · 8 hilos 4,31 · 12 hilos **5,18** (24% peor). Óptimo: **6 hilos anclados** (`OMP_PLACES=cores`) → 4,04 |
| **La iGPU es un callejón sin salida** | Vulkan 2,5x más lento que CPU; PyTorch XPU/IPEX no soportan Gen9.5 (`torch.xpu.is_available()` → False); y comparte el mismo bus, así que no suma ancho de banda |
| **`cfg_scale` no afecta a la velocidad** | 1.5/1.3/1.0 → RTF 3,92/4,02/4,20. Parchear el código para saltarse la pasada incondicional tampoco sirvió (3,90). Con batch 2 los pesos se leen una vez: la segunda mitad sale casi gratis |
| **La difusión domina** | Por token: 20 pasos × 4 capas × batch 2 = **160** evaluaciones de capa, frente a **24** del LLM |
| **Bajar pasos rinde menos de lo esperado** | 20→8 pasos: RTF 5,39 → 3,90 = **1,38x**, no el ~2x teórico. El LLM no encoge y marca el techo de Amdahl |

**Corolario importante:** como el cuello es leer *pesos* desde RAM, lo que
paga es **reducir bytes de peso** (cuantización), no reducir operaciones.

---

## Candidatos a evaluar

Ordenados por retorno esperado. Los marcados con ⚠ cambian la salida del
modelo y exigen juzgar la calidad de oído, no solo mirar el RTF.

### 1. Cuantización int8 dinámica ⚠
Ataca la causa raíz: pesos a 1/4 del tamaño = 1/4 del tráfico.

**Escepticismo honesto:** el i7-8700T (Coffee Lake) **no tiene AVX512-VNNI**.
La multiplicación int8 se emula y puede comerse el ahorro de ancho de banda.
Puede salir peor — por eso se mide.

Si int8 puro falla, probar **solo el LLM en int8** dejando la cabeza de
difusión en fp32 (o al revés): puede que una de las dos partes sí gane.

### 2. Pasos de difusión ⚠
Ya medido hasta 8 pasos (1,38x). Falta 6 y 4, y sobre todo **decidir el
mínimo aceptable de oído**. El modelo usa `DPMSolverMultistepScheduler`, que
está diseñado para pocos pasos, así que 6 puede sonar bien.

### 3. `torch.compile` con inductor
Fusiona operaciones y reduce viajes a memoria intermedios. En matrices
pequeñas (dim 896) la fusión puede importar más que en grandes. Coste: el
tiempo de compilación en el primer uso.

### 4. Ajuste de oneDNN / MKLDNN
`torch.backends.mkldnn`, formatos de memoria (`channels_last` no aplica aquí),
y `torch.inference_mode()` en lugar de `no_grad()`.

### 5. Descartado sin medir: bf16
Coffee Lake no tiene AVX512-BF16. Sería emulado y casi seguro más lento.
Solo probar si sobra tiempo.

### 6. Streaming — la palanca real para conversar
**No baja el RTF, pero es lo que de verdad convierte esto en conversación.**
Emitir los primeros milisegundos mientras se genera el resto. Microsoft
declara ~200 ms al primer sonido, y el repo trae `vibevoice_realtime_demo.py`
con websocket.

Con RTF 0,8 sin streaming esperas 8 s antes de oír nada. Con streaming oyes
en cientos de milisegundos aunque el RTF siga por encima de 1.

---

## Cómo medir bien

- **Un modelo por proceso.** Cargar fp32 y su copia int8 a la vez pasa de
  4,7 GB y el OOM mata el proceso (ya pasó). La VM tiene 5 GB.
- **Salida sin buffer** (`python -u`) o las filas se pierden al morir.
- **Siempre 6 hilos anclados** para que las comparaciones sean válidas:
  `OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close`
- **Guardar el .wav de cada variante.** El RTF sin calidad no significa nada:
  una variante 3x más rápida que suena mal no sirve.
- **`HF_HUB_OFFLINE=1`** — el modelo está en el store.
- El prefijo de voz **se muta** en `generate()`: recargar y `deepcopy` en cada
  medición o la segunda salida no se parece a la primera.

## Criterio de éxito

1. RTF medido para cada variante, con el mismo texto y la misma voz
   (`sp-Spk1_man`, la que el usuario prefirió).
2. Un `.wav` por variante para juzgar calidad.
3. Una recomendación clara: qué combinación usar y qué se sacrifica.
