# Comparativa de motores: VibeVoice-0.5B frente a Qwen3-TTS-0.6B

Qué pasó cuando se puso un segundo modelo al lado del que sirve la VM, con la
misma voz, las mismas frases y el mismo juez. El objetivo era saber tres cosas
antes de tocar producción: si clona mejor, si habla más idiomas de verdad, y si
cabe en la CPU del M920q. Las tres se responden con números, y el orden en que
se responden importa: la tercera decide dónde corre el modelo, no si se usa.

## 1. Por qué Qwen3-TTS y no otro

Se evaluaron siete estrategias con la misma rúbrica (calidad, errores, semanas,
euros, riesgo) antes de escribir una línea. Resumen de lo que descartó a los
demás candidatos, septiembre de 2026:

| Modelo | Por qué no |
|---|---|
| Voxtral TTS (4B), Fish S2 Pro, Breeze TTS 2, XTTS-v2 | licencia no comercial |
| OmniVoice (600 idiomas) | solo GPU |
| CosyVoice3-0.5B | orientado a GPU; lento en CPU |
| Chatterbox Multilingual v3 | 500M + difusión, sin afinado oficial, sin cifras de CPU |
| NeuTTS-Nano-Spanish | licencia propia con registro |
| Pocket TTS (100M) | plan B ligero: 6 idiomas y 6× tiempo real, pero techo de clonado bajo |
| DGX Spark | RTF 0,60 medido para el 1.7B, lo mismo que una L4, por 4.800 EUR |

Qwen3-TTS-12Hz-0.6B-Base (Apache 2.0, enero de 2026): 10 idiomas, clonado
desde 3 s por x-vector o por audio + texto en contexto, streaming, afinado
oficial monolocutor, y en su informe técnico la mejor similitud de locutor en
los 10 idiomas (español: WER 1,49 %, SIM 0,812). Su defecto conocido: acelera
el ritmo pasados ~100-150 caracteres (issue #239, cerrado sin arreglo).

## 2. Cómo se midió

- **Voz**: el hablante 0 del vídeo `PXL_20260828_235918691` de `dobla`, 28 s de
  segmentos anotados a mano sobre la pista de voz separada por demucs (46 % de
  silencio: 15 s de voz real). Es una referencia de las que el doblaje se
  encuentra, no una de estudio. Su **techo** (una mitad contra la otra, ECAPA)
  es **0,764**; f0 138 Hz.
- **Frases**: las dos de `banco_clonado.py` (neutro y expresivo, sin `¿ ¡`) más
  una larga de 250 caracteres, en español; un neutro y un largo en inglés,
  portugués, francés, italiano y alemán. Tres semillas (11, 42, 101).
- **Juez**: `scripts/banco_motores.py`. ECAPA contra la referencia
  (speechbrain, el mismo de `oido.py`), WER con faster-whisper `small` en el
  idioma del texto, `sesgo_st` = 12·log2(f0 clon / f0 referencia), `car/s` sobre
  el tiempo con voz (umbral relativo p95 − 25 dB).
- **Motores**: VibeVoice-0.5B desnudo (torch, cfg 3, 6 pasos) con el prefijo
  `.pt` de `clonar_voz.py` sobre esa referencia; Qwen3-0.6B con el motor C
  (`--int8`, 4 hilos) de dos formas: referencia cruda (`--ref-audio`, injerto
  ICL) y x-vector `.bin` de `clonar_voz_qwen.py` (`--xvector-only`).
- **Dónde**: el Mac M4, con los tres motores corriendo a la vez. Por eso el RTF
  de esta tabla **no vale**: solo mide contención. El RTF que decide es el de
  la VM (§5).

## 3. Resultados

| motor | idioma | n | fallos | ECAPA media | ECAPA min | WER media | WER peor | sesgo st | car/s | RTF |
|---|---|---|---|---|---|---|---|---|---|---|
| VibeVoice **producción** (voz_stream, parches, MPS) | en | 6 | 0 | 0.528 | 0.433 | 12.5% | 28.6% | -0.1 | 22.1 | 1.08 |
| VibeVoice **producción** (voz_stream, parches, MPS) | es | 9 | 0 | 0.544 | 0.363 | 13.0% | 55.6% | +0.8 | 20.2 | 1.31 |
| Qwen3-0.6B motor C injerto ICL | de | 6 | 0 | 0.149 | 0.022 | 8.3% | 17.5% | +1.7 | 21.2 | 0.78 |
| Qwen3-0.6B motor C injerto ICL | en | 6 | 0 | 0.407 | 0.187 | 1.4% | 4.3% | +0.6 | 19.4 | 1.47 |
| Qwen3-0.6B motor C injerto ICL | es | 9 | 0 | 0.533 | 0.446 | 1.3% | 4.8% | +1.0 | 19.6 | 2.09 |
| Qwen3-0.6B motor C injerto ICL | fr | 6 | 0 | 0.360 | 0.252 | 3.5% | 8.3% | +1.2 | 25.0 | 0.75 |
| Qwen3-0.6B motor C injerto ICL | it | 6 | 0 | 0.359 | 0.174 | 0.0% | 0.0% | +1.6 | 17.8 | 0.74 |
| Qwen3-0.6B motor C injerto ICL | pt | 6 | 0 | 0.415 | 0.296 | 6.4% | 15.4% | +1.3 | 17.6 | 0.65 |
| Qwen3-0.6B motor C **x-vector .bin** | de | 6 | 0 | 0.225 | 0.170 | 8.8% | 20.0% | -0.1 | 21.2 | 0.78 |
| Qwen3-0.6B motor C **x-vector .bin** | en | 6 | 0 | 0.410 | 0.314 | 1.1% | 2.2% | +0.6 | 19.3 | 0.70 |
| Qwen3-0.6B motor C **x-vector .bin** | es | 9 | 0 | 0.536 | 0.395 | 1.6% | 4.8% | -0.2 | 19.6 | 0.72 |
| Qwen3-0.6B motor C **x-vector .bin** | fr | 6 | 0 | 0.436 | 0.388 | 3.1% | 6.2% | +0.2 | 24.8 | 0.71 |
| Qwen3-0.6B motor C **x-vector .bin** | it | 6 | 0 | 0.302 | 0.066 | 0.4% | 2.5% | -0.6 | 16.9 | 0.73 |
| Qwen3-0.6B motor C **x-vector .bin** | pt | 6 | 0 | 0.472 | 0.385 | 7.2% | 17.9% | +0.7 | 17.9 | 0.70 |
| VibeVoice desnudo (torch, cfg 3) | de | 6 | 0 | 0.436 | 0.358 | 22.4% | 40.0% | +0.9 | 23.0 | 3.45 |
| VibeVoice desnudo (torch, cfg 3) | en | 6 | 0 | 0.523 | 0.439 | 5.9% | 28.6% | +0.1 | 23.0 | 5.17 |
| VibeVoice desnudo (torch, cfg 3) | es | 9 | 0 | 0.595 | 0.545 | 11.1% | 35.7% | +0.8 | 21.5 | 16.39 |
| VibeVoice desnudo (torch, cfg 3) | fr | 6 | 0 | 0.598 | 0.564 | 43.1% | 75.0% | +0.5 | 22.0 | 6.69 |
| VibeVoice desnudo (torch, cfg 3) | it | 6 | 0 | 0.612 | 0.552 | 13.0% | 27.5% | +0.7 | 20.2 | 5.99 |
| VibeVoice desnudo (torch, cfg 3) | pt | 6 | 0 | 0.628 | 0.540 | 14.5% | 23.1% | +0.8 | 20.4 | 5.18 |

Cómo leerla (referencia con techo 0,764; el RTF es del Mac con contención, no
cuenta):

- **Identidad en español: empate.** Producción 0,544, Qwen3 x-vector 0,536. El
  VibeVoice desnudo da 0,595 y el peor caso más alto (0,545 frente a 0,395):
  conserva mejor el timbre, pero es el modelo sin los parches del servicio.
- **Inteligibilidad: Qwen3 gana en todos los idiomas, y de largo.** WER medio
  en español 1,6 % frente a 13,0 % de producción (peor caso 4,8 % frente a
  55,6 %). En inglés 1,1 % frente a 12,5 %. Con esta referencia —una charla
  real separada por demucs, no una voz de estudio— VibeVoice se atasca en una
  de cada tres frases; Qwen3 no se atasca en ninguna de las 78.
- **Idiomas de verdad.** VibeVoice "habla" francés con un 43 % de WER y alemán
  con un 22 %: el timbre se conserva porque el prefijo manda, pero no se
  entiende. Qwen3: italiano 0,4 %, francés 3,1 %, portugués 7,2 %, alemán 8,8 %.
- **Clon cruzado (referencia en español → inglés): VibeVoice conserva más
  identidad (0,53 frente a 0,41)**, a costa de la inteligibilidad. Qwen3
  pierde 0,13 de identidad al cambiar de idioma, y en alemán se queda en
  0,22: la voz sigue siendo "un hombre grave", no esta persona.
- **Sesgo de tono: resuelto con el x-vector.** −0,2 st en español frente al
  +0,8 de producción (y al +2,5 documentado en voces graves con VibeVoice).
  Con el injerto ICL Qwen3 también sube +1,0.
- **Ritmo estable.** 19-20 car/s en español y 19 en inglés en los tres textos,
  incluido el de 250 caracteres: con trozos de 160 el issue #239 no aparece.
  VibeVoice desnudo habla más rápido (21-23) y en producción 20.


## 4. Lo que se aprendió por el camino

- **El x-vector y el injerto dan la misma identidad media en español (0,536
  frente a 0,533), pero el x-vector no tiene sesgo de tono (−0,2 st frente a
  +1,0) y aguanta mejor el peor caso al cambiar de idioma** (inglés mínimo 0,314
  frente a 0,187; francés 0,388 frente a 0,252). Es coherente con lo que
  documenta el motor C: el `.bin` lleva la identidad sin la sala de la
  grabación. El shim sirve el `.bin` cuando existe, y `clonar_voz_qwen.py` lo
  fabrica siempre. (En una sola frase y semilla el x-vector llegó a 0,642
  frente a 0,569: una semilla no es una medida.)
- **Quitar los silencios de la referencia no ayuda a Qwen3**: con la misma
  grabación compactada a 27 s de voz (de 28 s con 46 % de silencio), ECAPA en
  español 0,529 frente a 0,536, y en inglés 0,344 frente a 0,410. El
  codificador de locutor no sufre por los huecos; lo que limita es la voz y la
  sala. `clonar_voz_qwen.py` no recorta silencios a propósito.
- **RTF del motor C en el M4 sin contención, 6 hilos, x-vector**: 0,57 en
  español y 0,56 en inglés (frases de 5-14 s, carga del modelo incluida).
- **El PyTorch de referencia (`qwen-tts`) se desboca en el Mac**: en fp32, en
  MPS y en CPU, una frase de 6 s produce 655 s de audio (no emite el fin).
  Con los mismos pesos el motor C va bien. Queda pendiente de diagnóstico
  (probablemente bf16 obligatorio); mientras, el motor C es el camino y el
  PyTorch solo hace falta para el afinado.
- **Motor C en el M4 con voz clonada**: RTF 1,14 en int8 sin contención; de
  los 7 s de una frase de 6 s, 4,9 son del decodificador de voz. En x86 el
  autor mide 2,02 en un Ryzen AVX2, y el i7-8700T es de esa familia.
- **Las rutas del store sin `-o` se recolectan**: `nix build -o banco/...`.

## 5. La puerta: RTF en la VM

`scripts/banco_rtf_vm.sh` corre el motor C en la VM con int8/int4 y 4/6 hilos
sobre las tres frases en español. Umbral del plan: **RTF ≤ 1,0 y ≤ 3,5 GB**.

Pendiente: la VM `taller` no existe todavía (hay que ejecutar
`scripts/crear_taller.sh` desde el Mac; el clasificador de permisos no deja
hacerlo desde la sesión). En cuanto exista, la tabla va aquí.

## 6. Decisión

Con lo medido:

1. **Qwen3-0.6B es el motor del doblaje y de los idiomas.** Mismo timbre en
   español, diez veces menos errores, seis idiomas inteligibles. Va al lote de
   `dobla` (`--motor qwen3`, shim en `taller`) sin tocar la VM `voz`.
2. **VibeVoice se queda en tiempo real hasta que la puerta diga otra cosa.**
   Si el motor C baja de RTF 1,0 en el i7 (`scripts/banco_rtf_vm.sh`), la
   configuración `voz-qwen` del flake lo sustituye con un cambio de nombre.
3. **La forma de clonar es el x-vector `.bin`**, no el injerto: misma identidad,
   sin sesgo de tono, 8 KB por voz. `clonar_voz_qwen.py` lo fabrica y el shim
   lo prefiere.
4. **Queda abierto el clon cruzado.** Para doblar al inglés con la voz de una
   persona concreta, Qwen3 conserva menos identidad que VibeVoice (0,41 frente
   a 0,53) aunque se entienda todo. Es la siguiente medida que decide: referencia
   más limpia y larga, o el 1.7B (mejor SIM cruzado en el informe técnico),
   contra el mismo banco.

## 7. El clon cruzado, cerrado (8 de septiembre de 2026)

El punto 4 de arriba quedaba abierto: para doblar al inglés con la voz de una
persona, ¿compensa el WER de Qwen3 lo que pierde de identidad? La respuesta es
**no**, y esta vez con las voces del trabajo real, no con una de banco.

### Cómo se midió, y en qué se diferencia de §2

Dos voces del vídeo `PXL_20260829_002857474` de `dobla`, cortadas de la
**anotación humana** (no de la diarización automática), y elegidas por su
**techo** — la referencia partida en dos mitades, una contra otra:

| voz | material | techo |
|---|---|---|
| Laura | 30,0 s | **0,946** |
| Juan Pablo | 19,3 s | **0,598** |

Elegirlas por el techo es lo que aporta esta medida. La de §2 usó una voz de
techo 0,764, o sea el caso cómodo. Juan Pablo es el caso que rompe los
doblajes: su propio audio, comparado consigo mismo, **no llega al umbral de
"misma persona" (0,626)**. Ninguna de las dos mitades de un mismo hombre se
reconoce como él.

Mismas frases de §2, mismas semillas, mismo juez ECAPA, mismo día. Qwen3 con el
motor C (`--int8`, injerto ICL); VibeVoice con el prefijo `.pt` de
`clonar_voz.py` sobre **esa misma referencia**.

### Resultado

Identidad ECAPA media contra la referencia:

| voz | motor | es | en | media | % de su techo |
|---|---|---|---|---|---|
| Laura (techo 0,946) | VibeVoice | 0,65 | **0,45** | **0,572** | 61 % |
| Laura (techo 0,946) | Qwen3-0.6B | 0,52 | **0,30** | 0,433 | 46 % |
| Juan Pablo (techo 0,598) | VibeVoice | 0,52 (n=4) | — | — | 87 % (solo es) |
| Juan Pablo (techo 0,598) | Qwen3-0.6B | 0,36 | 0,20 | 0,295 | 49 % |

La celda de VibeVoice en inglés sobre Juan Pablo quedó sin medir: el
contenedor x86 emulado del Mac tardaba ~14 min por clip largo y la decisión ya
estaba tomada con el resto. Está anotado como pendiente, no como cero.

### Lo que dice

1. **VibeVoice gana en las dos lenguas, y la brecha se abre en inglés**
   (0,45 frente a 0,30). Como el doblaje es español→inglés, es el único caso
   que cuenta. El oído lo confirmó antes que el coseno: Qwen3 en inglés "suena
   muy bot" frente a un VibeVoice "un poco más natural".
2. **El WER de 1,4 % no compensa.** Era la apuesta razonable — el 82 % de los
   fallos residuales del doblaje son por WER, no por identidad — pero lo que se
   entiende perfectamente y suena a robot no es un doblaje.
3. **El techo manda sobre el motor.** Los dos rinden un porcentaje parecido del
   techo de cada voz (46-61 %). Ninguno arregla una grabación mala: con techo
   0,598 no hay motor que salve esa voz, y ahí el trabajo está en grabar mejor,
   no en cambiar de modelo. (Dos voces son dos puntos: es indicio, no ley.)
4. **RTF y memoria no se compararon**, y no se pueden con estos datos: Qwen3
   corrió nativo en ARM (RTF 1,02-1,44, RSS 200 MB) y VibeVoice en un
   contenedor x86 **emulado** (RTF 73,6, RSS 7,5 GB). La cifra de VibeVoice es
   del emulador, no del motor; en la c7i real va a ~1,0.

**Qwen3 se queda disponible, no descartado.** Su motor C funciona, cabe en
200 MB y en español pierde bastante menos. Para monolingüe español o como plan
B en una máquina pequeña tiene sitio. Para el doblaje al inglés, no.
