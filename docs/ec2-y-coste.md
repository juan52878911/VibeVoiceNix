# Coste por carácter y migración a EC2

**La restricción dura sigue siendo la calidad.** Todo lo de aquí se optimiza **debajo** de la puerta del
banco A/B; ninguna cifra de coste justifica saltársela.

**(M)** = medido, con su fuente. **(E)** = estimado, con la cuenta a la vista.

Propuesta larga, con el detalle de instancias y fases: nota de Obsidian
`01-proyectos/06-vibevoicenix/propuesta-ec2-2026-09-15.md`. Este fichero guarda lo que hay que saber
para decidir y para no repetir análisis.

---

## 1. La métrica correcta es el coste por millón de caracteres

No es el precio al mes de una instancia: eso mide sobre todo **cuánto pagas por tenerla parada**. Con el
candado de una locución a la vez (`voz_stream.py`), el coste del trabajo es:

```
$/M caracteres = precio_hora × RTF / 0,0576
```

El 0,0576 son los **57 600 caracteres por hora de audio** que da el banco a ~16 caracteres por segundo
de habla (M, banco del 01-09-2026).

| Vía | $/M caracteres | Estado |
|---|---|---|
| Polly Generative | 30 | precio de lista |
| Polly Neural | 16 | precio de lista |
| Polly Standard | 4 | voz concatenativa antigua: no compite con esto |
| c7i.2xlarge bajo demanda | ≈ 4,7 (E) | **5,9 (M)** con el stack anterior |
| c7i.2xlarge en spot | ≈ 2,6 (E) | **2,7 (M)** |
| c8g.xlarge (Graviton4) en spot | ≈ 1,05 (E) | **sin medir**, y con riesgo de calidad (§4) |
| c7a.xlarge (Zen 4) en spot | ≈ 0,85 (E) | sin medir; depende de que vCPU = núcleo físico |
| **La VM de casa** (i7-8700T a 35 W) | **≈ 0,08-0,12 (E)** | solo electricidad; funcionando hoy |

**Conclusión que hay que tener presente antes de mover nada:** frente a Polly Neural este motor ya es
**6× más barato** y frente a Generative **11×**, con lo medido. Contra Standard pierde bajo demanda y
gana en spot.

**Y el suelo real es el hardware propio, por dos órdenes de magnitud.** Una hora de reloj del i7 a
RTF 0,9 produce ~64 000 caracteres y consume 35 Wh: a $0,15/kWh son medio centavo, o sea ~$0,08 por
millón (E). EC2 en spot cuesta **20-30× más**, porque lo que se paga es el alquiler de la máquina, no el
cómputo. **La nube no es la palanca de coste: es la palanca de disponibilidad y de ráfaga.**

---

## 2. Lo que dispara la factura es el tiempo parado, no el tamaño

Un vídeo de 10 minutos (≈8 min de habla) son 360 s de TTS a RTF 0,75 (E) más transcripción y ffmpeg:
≈ 10 minutos de instancia, **≈ $0,03 en spot** (E). Cien vídeos al mes: $3-5. Lo que cuesta $261 es el
97 % del mes sin trabajo.

| Modo | Fijo al mes | Por hora de audio | Para qué |
|---|---|---|---|
| Homelab, como hoy | ~0 marginal | ~0 | **lo más barato, y ya pasa todas las puertas** |
| Spot bajo demanda desde una AMI, apagando al acabar | ≈ $0,60 (snapshot de ~7 GB + S3) | ≈ $0,15 | lotes (dobla): el camino recomendado |
| Instancia parada y arrancada a mano | $2,40 (EBS de 30 GB reservado) | ≈ $0,15 | lo mismo sin automatizar |
| 24/7 en spot, c7i.xlarge | ≈ $50 | — | asistente en vivo, aceptando cortes |
| 24/7 bajo demanda, c7i.2xlarge | $261 | — | solo con un SLA que lo exija |

Un Savings Plan recorta el 24/7 del orden de un 30-45 % (E, sin verificar), pero ata un año: con el
volumen sin conocer, es el peor momento para firmarlo.

---

## 3. Qué instancia, si hay que alquilar

Elegir por **núcleos físicos**, no por vCPU: en Intel y AMD cada vCPU de EC2 es un hyperthread; en
Graviton es un núcleo. Más hilos que núcleos empeora (M: 12 hilos +24 % en el i7; 8 hilos 1,154 frente a
0,988 con 6). Y **un proceso por instancia**: dos workers × 2 hilos dieron 0,64× de caudal agregado (M).

| Instancia | Núcleos / RAM | OD $/h | Spot $/h | Nota |
|---|---|---|---|---|
| c7i.xlarge | 2 / 8 GiB | 0,179 | 0,070 | sin dato: 3 interrupciones de spot seguidas el 16-09 |
| c7i.2xlarge | 4 / 16 GiB | 0,357 | 0,194 | **RTF 0,987 bf16 / 1,259 f32 (M, 16-09)** |
| c8i.xlarge (Granite Rapids) | 2 / 8 GiB | 0,187 | 0,087 | RTF 1,230 bf16 (M) |
| c7a.xlarge (Zen 4) | **4** / 8 GiB (1 hilo por núcleo, M) | 0,205 | 0,088 | RTF 0,766 bf16 (M) |
| **c8a.xlarge (Zen 5)** | **4** / 8 GiB | 0,216 | 0,087 | **RTF 0,498 bf16 / 0,638 f32 (M)**: la óptima por $ |
| c8a.2xlarge (Zen 5) | 8 / 16 GiB | 0,432 | 0,184 | RTF 0,404 bf16 (M, 7 hilos) |
| c8g.xlarge (Graviton4) | 4 / 8 GiB | 0,160 | 0,081 | sin medir: necesita imagen arm64 |
| g4dn / g6 (GPU) | — | 0,526 / 0,805 | — | **no**: los IR de OpenVINO no corren en NVIDIA; el camino de GPU es torch fp16 sin cuantizar, o sea otro audio. Con una locución a la vez, una T4 tendría que dar RTF < 0,25 para empatar. Cuota de la cuenta en 0 |

Precios de us-east-1 del 15-09-2026, de terceros (Vantage); el spot cambia cada hora.
Spot y RTF de la tabla: banco del 16-09-2026 en AWS Batch con la imagen de dobla (mediana spot 24 h;
10 frases, 3 rondas, cuenta la 3ª). Informe completo y datos crudos en el repo `dobla`,
`docs/benchmark-instancias-ec2.md`. El motor pica **2,2 GB** (VmHWM) en todas, pero el job completo de dobla muere por OOM en 8 GiB al clonar con el motor ya residente: dobla usa **m8a.xlarge** (misma CPU que la c8a, 16 GiB).
Las tres AMD dan audio idéntico bit a bit entre sí en bf16; Intel da otro distinto por generación.

**Si el trabajo es por lotes** (dobla: nadie espera mirando), el RTF deja de ser restricción de producto
y pasa a ser solo coste. Entonces se elige por **$/cómputo en spot**, no por latencia, y la puerta P3 de
RTF ≤ 0,90 solo aplica al asistente en vivo.

---

## 4. La trampa número uno: bf16 silencioso

> **Confirmado el 16-09-2026 (M):** OpenVINO 2025.4.1 da `INFERENCE_PRECISION_HINT = bfloat16` por
> defecto y compila los 9 IR en bf16 tanto en c7i.2xlarge (AMX) como en c8a.xlarge (AVX512_BF16 sin
> AMX). dobla en AWS ha corrido siempre en bf16. Forzando f32 (envolviendo `Core.compile_model`), el
> RTF sube +28 % en las dos, y las mismas 10 frases cambian de duración (c7i 70,7 s bf16 → 59,7 s f32;
> c8a 69,5 → 64,4 s).
>
> **Decidido el 17-09-2026 (M):** `banco_ab.py` en m8a.xlarge, 48 parejas (carlos, avril × 12 frases × 2
> semillas), base f32, control int4 válido (UTMOS −0,020 [−0,032, −0,009]). **bf16 no pasa**: UTMOS
> +0,013 [−0,033, +0,060] (IC inf < −0,02), WER +0,66 [−1,24, +2,68] pts (IC sup > +0,5), identidad
> +0,014 [+0,002, +0,026], final del habla ±100 ms de IC. No se hunde, pero no demuestra equivalencia.
> dobla corre en f32 desde entonces con `VIBEVOICE_OV_PRECISION=f32` (commit 478bd54 lo hace elegible;
> sin la variable, lo que decida OpenVINO). f32 cuesta +24 % de RTF: 0,707 frente a 0,571 en ese banco.
> Informe en el repo `dobla`, `docs/banco-precision-2026-09-17/`.

En Sapphire Rapids y en Zen 4/5, **el plugin de CPU de OpenVINO pasa a bf16 por su cuenta** cuando la
máquina tiene AVX512_BF16 o AMX, salvo que se le ponga `INFERENCE_PRECISION_HINT=f32`. `motor.py` solo
pone ese hint en el camino de GPU, porque en AVX2 no hacía falta.

Consecuencia inmediata: **el banco de AWS del 01-09-2026 casi con seguridad corrió en bf16 y nadie midió
su calidad**; su RTF 0,99 puede ser optimista frente al f32. El precedente propio asusta: el mismo
decodificador en f16 en la iGPU quedó a **16 dB** de la CPU, y el bf16 tiene aún menos mantisa (7 bits
frente a 10).

**Antes de medir nada en EC2** hay que hacer configurable por entorno, con f32 por defecto en CPU:
`INFERENCE_PRECISION_HINT`, `DYNAMIC_QUANTIZATION_GROUP_SIZE` (a 0) y `KV_CACHE_PRECISION` (a f32). Las
dos últimas se cerraron en su día porque «no se aplican en esta CPU»: esa razón **desaparece** en
AVX-512/AMX. Y el cambio se valida en la VM de casa, donde en AVX2 tiene que dar md5 idéntico (las 14
combinaciones ya salieron bit a bit iguales, M).

En ARM el riesgo es el mismo o peor: el plugin usa fp16 por defecto **y ejecuta los modelos cuantizados
en modo simulación**, así que el decodificador int8 podría ir en fp32 emulado, o sea lento.

---

## 5. El md5 no es la puerta entre CPUs distintas

Con otra ISA los kernels son otros (`brgemm_avx512`/AMX en vez de `brgemm_avx2`), cambia el orden de
acumulación y por tanto el redondeo. Como el latente de la difusión vuelve al LM, algunos clips
cambiarán de duración y de lectura — igual que pasó al fusionar la difusión (27 de 238 clips) y con el
LM int8 (189 de 238 desplazados).

| Puerta | ¿Vale entre CPUs? |
|---|---|
| Huella de los 65 tensores torch | **sí**: es independiente de la ISA; si falla, el artefacto está mal |
| md5 frente a la VM de casa | **no**, y no se reinterpreta como «se rompió» |
| md5 de la máquina nueva **consigo misma** (3 procesos × 3 rondas) | **sí, obligatoria**: es lo que promete `semilla` |
| `ws_fidelidad.py`, con el md5 de referencia de la máquina nueva | sí |
| `banco_ab.py`, 238 parejas con control int4 | **sí: es LA puerta**. UTMOS IC inf ≥ −0,02 · WER IC sup ≤ +0,5 · identidad ±0,005 global y ±0,0023 por clon · tono y recorrido ±0,03 st · final ±20 ms · control int4 con UTMOS IC sup < 0 |
| Banco de semillas (18 × 6 frases) | sí, informativo: la 101 se eligió con la numérica del i7 |

**Sin la puerta del banco no hay despliegue, por muy bajo que salga el RTF.**

---

## 6. Los IR son un artefacto, no algo que se genere en la nube

Convertirlos pica **4,6 GB de RAM y ~15 minutos** (M). En EC2 no se convierte nada: los cuatro IR de
producción (562 MB) se bajan de S3 con su md5 en el arranque, igual que ya se guardan en el NAS
(`/tank/nfs/vibevoice/modelos/ov-2026-09-15`). El artefacto canónico es el de casa.

Efecto lateral en el homelab: **esto desbloquea bajar la VM de 5120 a 4096 MB**, que estaba bloqueada
justo por ese pico de conversión.

El token y las voces propias **nunca** van en la AMI: un `.pt` es la voz clonable de una persona.

---

## 7. Cómo llevarlo

**NixOS desde la AMI oficial y el mismo flake** (`nixosConfigurations.voz-ec2`, derivada de `voz`, con
`virtualisation.amazon-image` en vez de `disko.nix`/`host.nix`), y después congelar una AMI privada. Es
la única vía que mantiene una sola fuente de verdad, y todo lo medido en la VM se traslada.

**Las imágenes Docker actuales no sirven tal cual**: ponen `OMP_PLACES=cores` (−118 % con OpenVINO, M),
no llevan los IR ni `VIBEVOICE_MOTOR=openvino`, pesan 9,8 GB y arrancan el camino torch (RTF 2,2).

Construir la AMI en casa y subirla tampoco: son 4-7 GB por la NIC de `pve`, que ya se colgó una vez con
tráfico sostenido.

---

## 8. Lo que NO hay que hacer

- No comparar el RTF de EC2 con el de la VM «medido otro día»: solo deciden las tandas alternas dentro
  de la misma máquina.
- No dar por buena una instancia por su RTF sin el banco de calidad.
- No usar las imágenes Docker actuales en EC2, ni poner `OMP_PLACES`/`OMP_PROC_BIND` con OpenVINO.
- No subir `NUM_STREAMS` ni meter dos workers en una instancia (0,64× medido).
- No convertir los IR en cada arranque, ni ejecutar el conversor y voz-stream a la vez en 8 GiB.
- No reproponer B1, B2, C1, C2, C3, decodificador int4, cabeza int4/fp32, 8-10 pasos, `neg_cada > 1`,
  `torch.compile` ni «optimizar Python»: están medidos y cerrados en
  [optimizacion.md](optimizacion.md). Lo único que la CPU nueva **reabre** es bf16, la cuantización
  dinámica de activaciones y la caché KV en u8, y las tres pasan por el banco.
- No gastar créditos en GPU ni en instancias de un solo núcleo.
- No firmar un Savings Plan antes de conocer el volumen.

---

## 9. Lo siguiente, si se decide medir

Lo más rentable por dólar gastado es **Graviton**: 4 horas de c8g.xlarge en spot son **$0,32** y cierran
con números la pregunta de si se puede llegar a ~$1 por millón de caracteres. Antes hace falta el cambio
de precisión del §4, relockear `torch+cpu` para aarch64 (el `uv.lock` fija ruedas de `linux-x86_64`) y
aceptar que el backend de cuantización de torch pasa a `qnnpack`.

El plan completo de fases (E0 a E4) y su presupuesto —≈ $6 de los 50 USD de créditos— está en la nota de
Obsidian.

---

## 10. La campaña de la red en spot (26-09-2026)

Tres guiones en [`scripts/red/`](../scripts/red/) para correr las herramientas de la red por dentro (torch en
CPU, sin OpenVINO) en una máquina spot, reanudables ante una interrupción:

| Guion | Qué hace |
|---|---|
| `spot_ec2.sh lanzar [tipo] [horas]` · `ip` · `subir` · `credenciales` · `gasto` · `terminar` | calcado de `gpu_ec2.sh`: solicitud spot de una sola vez (terminate al interrumpir), Ubuntu 22.04 de Canonical, zona más barata del momento, apagado a las N horas, mismo tope y libro de gasto con la tarifa spot leída al lanzar. `credenciales` deja en la máquina credenciales temporales de STS (caducan solas), nunca las del Mac |
| `spot_entorno.sh` (dentro) | el `uv.lock` de `pkgs/vibevoice` (torch 2.13.0+cpu, transformers 4.57.6, VibeVoice al commit fijado) más los jueces de CPU; modelo, codificador comunitario y prefijos oficiales; voces y patrón desde S3 con la ruta en variables de entorno; f32 estricto en cada proceso (`sitecustomize`) comprobado con un matmul contra f64; y `probar_mecanica.py --modelo` con las ocho pruebas |
| `campana.sh` (dentro) | instrumentar → sondas → dirigir → jueces (`juez_lote.py` y `puerta_dirigir.py`, con costura en los barridos `--desde`), con los parámetros y puertas de [la campaña del Mac](bancos/2026-09-27-red-interna-mac.md). Al arrancar baja el trabajo de S3 y borra lo truncado; sube cada 2 min, al acabar cada paso y con el SIGTERM del aviso de spot |

**Tanda corta (M):** solo paridad, porque la campaña la midió la otra sesión en el Mac. Fueron dos máquinas: la
primera se terminó a propósito después de instrumentar, y la segunda reanudó desde S3.

| | Máquina 1 | Máquina 2 |
|---|---|---|
| Tipo | c8a.2xlarge, AMD EPYC 9R45, 8 núcleos (1 hilo por núcleo), AVX-512 y AVX512_BF16 sin AMX | igual |
| Zona y tarifa spot | us-east-1a, 0,2068 $/h (la subred por defecto cayó en la zona cara; desde entonces se elige la más barata) | us-east-1f, 0,1406 $/h |
| Minutos y USD | 15,9 min, 0,075 USD | 17,9 min, 0,062 USD |
| Entorno montado y probado | 238 s, de ellos 145 s de `probar_mecanica` | 248 s (y 141 s al repetirlo con el arreglo de 821e7a8) |
| Mecánica con los pesos reales | 8/8 | 8/8, con 0 enganches vivos al acabar |
| Interrupciones | ninguna de AWS; terminada a mano tras subir el clip | ninguna; reanudó sin rehacer el clip y corrió la paridad |

Total: **0,137 USD** para las dos máquinas, cada una con 0,02 USD fijos de disco y S3. La reanudación también
se probó con un `.npz` truncado en S3: se detecta al arrancar y el clip se rehace.

**RTF de torch fp32 en la c8a.2xlarge (M), 8 hilos:**

| Clip | RTF |
|---|---|
| Con el registro de residuales de `instrumentar.py`, en caliente | **1,8-2,0** (10 s de reloj para 5,1-5,3 s de audio; 14 s para 7,9 s) |
| El mismo clip, primero del proceso, en frío | 3,5 |
| Sin residuales, en caliente | 1,7 (9 s para 5,2 s) |

Es la misma escala que los 235 ms por fotograma de la máquina de 4 vCPU del
[banco sin modelo](bancos/2026-09-26-red-interna-entorno.md). Con estos números, la campaña completa (4 voces
× 20 frases × 2 semillas en instrumentar, cinco barridos de dirigir) son unas 2-3 h de c8a.2xlarge en spot:
**≈ 0,3-0,6 USD (E)**, jueces aparte.

**La paridad con el clip del Mac no pasa, y no es bf16 (M).** El clip de `sp-Spk1_man`, `es0`, semilla 11 sale
con la misma longitud que el del Mac (124 800 muestras, 39 fotogramas), pero a 20,2 dB de SNR (diferencia
máxima 0,20). Para separar la causa se regeneró en la máquina 2:

| Variante en la máquina 2 | Frente a la máquina 1 | Frente al Mac |
|---|---|---|
| f32, 8 hilos | **md5 idéntico** | 20,2 dB |
| `ONEDNN_DEFAULT_FPMATH_MODE=BF16` y sin forzar f32 | **md5 idéntico** | 20,2 dB |
| f32, 4 hilos | 23,6 dB | 16,5 dB |

- **torch en CPU no pasa a bf16 por su cuenta** en Zen 5, ni forzando el modo bf16 de oneDNN. La trampa del §4
  es de OpenVINO. El matmul f32 da un error relativo de 3,6·10⁻⁷ frente a f64.
- **La diferencia es el orden de las sumas**, amplificado porque el latente vuelve al LM (§5). Cambiar solo el
  número de hilos en la misma máquina separa el audio tanto como pasar del Mac a x86.
- **La puerta entre máquinas es la del §5:** que la máquina sea consistente consigo misma y que
  `probar_mecanica.py` pase allí. Las dos cosas se cumplen: las dos c8a dan el mismo md5 y la paridad de
  `probar_mecanica` es muestra a muestra. «md5 o SNR > 60 dB frente al Mac» no se puede cumplir entre ISAs.
- **Consecuencia para cualquier campaña:** λ = 0 y λ > 0 tienen que salir de la misma máquina **y con el mismo
  número de hilos**. Si no, la diferencia pareada mezcla el mando con la numérica.

**El libro de gasto** vive en `~/Documents/mejora-modelo`, un enlace a un disco externo que el 26-09 no estaba
montado. `spot_ec2.sh` se niega a arrancar sin libro, porque uno vacío haría creer que no se ha gastado nada.
Con `LIBRO=<ruta>` se usa otro, que tiene que llevar el gasto previo como una tanda de arrastre. Las dos tandas
de hoy están en un libro provisional con el arrastre de 12,48 USD, y hay que fusionarlas con el de siempre
cuando se monte el disco.

---

## Documentos relacionados

| Documento | Qué añade |
|---|---|
| [optimizacion.md](optimizacion.md) | qué se optimizó, qué no funcionó y por qué |
| [plan-rendimiento.md](plan-rendimiento.md) | las puertas del plan de RTF, memoria y disco, y sus resultados |
| [hardware-y-portabilidad.md](hardware-y-portabilidad.md) | GPU por passthrough, RAM y llevar el stack al Mac |
| [despliegue.md](despliegue.md) | cómo se despliega hoy en la VM |
