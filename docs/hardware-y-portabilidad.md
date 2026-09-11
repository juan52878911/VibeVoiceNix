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

## 2. RAM: hecho, y qué cambió

**Era la mejora con mejor retorno del proyecto y ya está puesta**: el host
lleva 16 GB en dual channel, dos módulos de 8 GB, uno por canal
(`dmidecode` en el M920q):

```
Locator: ChannelA-DIMM0   Size: 8 GB   Speed: 2667 MT/s   Configured: 2400 MT/s
Locator: ChannelB-DIMM0   Size: 8 GB   Speed: 2400 MT/s   Configured: 2400 MT/s
```

Esta sección decía que **el segundo módulo tenía que ser idéntico** o el dual
channel podía no activarse. **No fue así**: los dos tienen frecuencia nominal
distinta, el dual channel se activó igual y ambos corren a 2400, que es lo que
impone el más lento.

### Motivo A: el techo de memoria bloqueaba trabajo real

Apareció **cuatro veces**, y con 16 GB han desaparecido las cuatro:

| Dónde | Qué pasaba |
|---|---|
| Banco de cuantización | OOM: fp32 + copia int8 pasaban de 4,7 GB |
| Servidor de streaming | Murió **3 veces** al coincidir con las conversiones |
| Grafos de OpenVINO | No caben en un sandbox de Nix (piden 4,6 GB) |
| Host de construcción | Sin margen mientras la VM tiene 5 GB reservados |

El tercero era el más caro y es el único con cola: los IR **siguen**
generándose fuera del store porque el contenedor constructor tenía 2560 MB.
Con la memoria de hoy puede que ya quepan, y entonces dejarían de ser el único
artefacto derivado del proyecto que no es una derivación de Nix. Está sin
comprobar.

### Motivo B: duplicaba el recurso que limitaba... y por eso dejó de limitar

Con un solo módulo se midió **17,2 GB/s de un máximo teórico de 21,3 = 80,7 %**:
la CPU sola exprimía el bus, y de ahí el corolario que ordenaba todo el
proyecto (lo que paga es mover menos bytes, no hacer menos operaciones).

**Con los dos módulos ese corolario ya no se cumple.** Tres pruebas en la VM lo
descartan: una pasada de backbone de dos tokens cuesta 1,77× la de uno cuando
debería costar ~1,0×; agrupar seis latentes en una llamada del decodificador no
gana nada; y bajar el decodificador de int8 a int4 da un 7 % en vez del 38 % que
darían los bytes. Ahora manda el **cómputo**: seis núcleos a 2,4 GHz con AVX2 y
sin VNNI. El detalle está en
[optimizacion.md](optimizacion.md#qué-queda-sobre-la-mesa).

No se ha vuelto a medir el ancho de banda en sí; la cifra de 17,2 GB/s que
aparece arriba es la de canal único.

### Si algún día hacen falta 32 GB

| Opción | Coste | Resultado |
|---|---|---|
| 2× 16 GB | ~60 € | 32 GB (el máximo del equipo), dual channel |

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

Y hay un detalle que jugaba a tu favor: **la memoria unificada de Apple Silicon
tenía entre 6 y 23 veces más ancho de banda** que la DDR4 del Tiny en canal
único. Ese argumento **se ha quedado a medias**: por un lado el host ya va en
dual channel, y por otro dejó de ser cierto que el cuello sea la memoria (ver
el apartado 2). Lo que sí se midió después, y es lo que importa: VibeVoice en
el Mac por MPS en fp16 da **RTF 1,39** (`docker/README.md`), frente al **1,09**
del OpenVINO int8 de la VM. La ventaja del Mac está en el ancho de banda, pero
la de la VM está en los grafos compilados y el int8, y gana la VM.

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
