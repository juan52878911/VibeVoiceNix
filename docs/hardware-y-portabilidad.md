# GPU por passthrough, RAM, y llevar esto al Mac

Tres preguntas de hardware, respondidas con lo medido en este proyecto y con
las especificaciones reales de la máquina.

---

## 1. GPU dedicada por IOMMU passthrough

### Lo que sí está a favor

**El IOMMU ya funciona.** El host tiene **9 grupos IOMMU** activos sin tocar
nada: los kernels recientes activan Intel VT-d por defecto, así que no hace
falta `intel_iommu=on`. El i7-8700T soporta VT-d, y Proxmox 9 hace passthrough
sin problemas. *Técnicamente, mañana mismo.*

### Lo que lo bloquea

El equipo es un **ThinkCentre M920q Tiny**: chasis de **1 litro**. No es un
caso donde falte una ranura libre — es que no cabe una GPU.

Sí tiene un **PCIe x8 propietario** (necesita el riser Lenovo `01AJ929` o
`01AJ940`), abierto por el extremo, así que eléctricamente admite tarjetas x16
a x8. Pero:

| Límite | Detalle |
|---|---|
| **Espacio** | Solo entran tarjetas de perfil bajo y una ranura |
| **Alimentación** | Ladrón externo de 65-90 W, **sin conector PCIe** |
| **Refrigeración** | Un chasis de 1 L no evacúa el calor de una GPU |

Las tarjetas que caben físicamente son de gama muy baja, y con 65 W de
presupuesto total no hay margen ni para una GTX 1650.

### La vía que sí existe: eGPU externa

Hay builds documentados de M720q/M920q con **riser ADT-Link R43SG** sacando el
PCIe fuera del chasis, GPU en una caja aparte y **fuente propia**. Uno conocido
monta una GTX 1080 Ti a x4.

Coste realista: riser 40-60 €, fuente 50-80 €, GPU usada 150-400 €, más una
caja improvisada. **Total 250-550 €** y el resultado deja de ser un mini-PC de
1 litro para ser un cacharro abierto sobre la mesa.

### Mi valoración

**No lo recomiendo para este proyecto**, y no por el dinero:

1. **Ya no lo necesitas.** El objetivo era tiempo real, y OpenVINO llegó a
   **RTF 1,09** en CPU. Con streaming, la espera es de 271 ms. La GPU
   resolvería un problema que ya está resuelto.
2. **El cuello es el ancho de banda**, y lo hemos medido tres veces. Una GPU
   con VRAM propia sí lo rompe — pero también lo rompe, en parte, el segundo
   módulo de RAM, que cuesta 20 € en vez de 400.
3. **Rompe la propiedad que buscabas.** Todo esto es replicable en cualquier
   Proxmox. Un montaje de eGPU con riser es específico de tu mesa.

**Cuándo sí tendría sentido:** si quieres correr un LLM local grande (7B+) para
OpenClaw, no solo voz. Ahí la VRAM es el requisito real y no hay CPU que valga.
Pero entonces la conversación es "un equipo con ranura PCIe de verdad", no
"adaptar el Tiny".

---

## 2. RAM: lo que de verdad hace falta

Es la mejora con mejor retorno del proyecto, y aparece por dos motivos
independientes.

### Motivo A: el techo de memoria bloquea trabajo real

Hoy ha aparecido **cuatro veces**:

| Dónde | Qué pasó |
|---|---|
| Banco de cuantización | OOM: fp32 + copia int8 pasaban de 4,7 GB |
| Servidor de streaming | Murió **3 veces** al coincidir con las conversiones |
| Grafos de OpenVINO | No caben en un sandbox de Nix (piden 4,6 GB) |
| Host de construcción | Sin margen mientras la VM tiene 5 GB reservados |

Ese tercer punto es el más caro: obliga a que los IR se generen fuera del
store, y es el **único artefacto derivado** de todo el proyecto que no es una
derivación de Nix.

### Motivo B: duplica el recurso que limita

Medido en el host: **17,2 GB/s de un máximo teórico de 21,3 = 80,7 %**. Ese es
el techo práctico de la DDR4, o sea que la CPU sola ya exprime el bus.

**Está en single channel** — un módulo de 8 GB y el segundo zócalo **vacío**
(máximo 32 GB, confirmado por `dmidecode`).

Un segundo módulo idéntico da **dual channel**: ~34 GB/s reales, el doble de
lo que hoy limita cada inferencia.

### Qué comprar

| Opción | Coste | Resultado |
|---|---|---|
| **+1× 8 GB DDR4-2667 SODIMM** | **~20 €** | 16 GB, dual channel |
| 2× 16 GB | ~60 € | 32 GB (el máximo), dual channel |

Con 16 GB desaparecen los cuatro bloqueos y los IR pueden pasar al store.

**Importante: el segundo módulo debe ser idéntico** (misma frecuencia y
preferiblemente mismo fabricante), o el dual channel puede no activarse y te
quedas solo con la capacidad.

---

## 3. Llevar esto al Mac para OpenClaw

La pregunta era si una VM en el Mac serviría para un asistente local ligero.
**Sí, pero la VM sobra — y además estorba.**

### Por qué una VM Linux es mala idea aquí

**Apple Silicon no hace passthrough de GPU a máquinas virtuales.** Ni UTM ni
Multipass ni VMware. Una VM Linux en tu Mac es **CPU pura**, y encima:

- Pierdes acceso a **Metal**, que es donde está toda la potencia del equipo
- `whisper.cpp` tiene soporte Metal excelente, y en una VM no lo usas
- Multipass en Apple Silicon **solo arranca imágenes ARM64**; para x86 habría
  que emular con UTM, entre 5 y 20 veces más lento

### Lo que sí funciona: nativo en macOS

Los tres motores corren nativos en Apple Silicon:

| Motor | En el Mac |
|---|---|
| **Piper** | ONNX Runtime en ARM — ya va a RTF 0,042 en un i7, aquí volaría |
| **whisper.cpp** | **Metal nativo**, mucho más rápido que los 0,853 del i7 |
| **VibeVoice** | CPU o MPS |

Y hay un detalle que juega a tu favor: **la memoria unificada de Apple Silicon
tiene entre 6 y 23 veces más ancho de banda** que tu DDR4 single channel. Como
demostramos que el cuello **es** el ancho de banda, VibeVoice iría bastante más
rápido en el Mac incluso sin tocar la GPU.

### La advertencia sobre MPS

No esperes milagros de Metal en PyTorch: para inferencia de modelos de lenguaje,
MPS ronda **7-9 tokens/s** frente a los **~230 de MLX**. Su compilador es
inmaduro y muchas operaciones caen a CPU. Para VibeVoice, que es difusión y no
un LLM autoregresivo puro, el resultado está sin medir.

### Mi recomendación

Para **un asistente ligero con OpenClaw en el Mac**, no necesitas VibeVoice:

```
Piper      →  voz de respuesta, RTF ~0,02 en Apple Silicon
whisper.cpp →  entender audios, con Metal
```

Eso es un asistente de voz completo en **menos de 500 MB de RAM**, respondiendo
en décimas de segundo. VibeVoice es el laboratorio: úsalo en el homelab, donde
ya está a RTF 1,09 con streaming.

### Qué haría falta en el repo

El flake ya construye `voz-api` para `aarch64-darwin`. Lo que falta:

1. **`whisper-cpp` con Metal** — está en nixpkgs, hay que activar la variante
2. **Un módulo `nix-darwin`** para levantarlo como servicio en macOS
3. **Re-generar el `uv.lock` de vibevoice** si quisieras el laboratorio también
   allí: hoy fija ruedas `torch+cpu` de linux-x86_64

Los puntos 1 y 2 son un rato de trabajo. El 3 es opcional y probablemente no
merezca la pena.
