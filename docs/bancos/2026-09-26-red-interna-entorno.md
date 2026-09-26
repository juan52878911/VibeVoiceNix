# La red por dentro: qué se pudo medir en un contenedor sin modelo, y qué no (2026-09-26)

Ejecución del [plan de la red por dentro](../plan-red-interna-2026-09-26.md) en la máquina de la sesión, con
la regla de siempre: lo medido se marca **(M)**, lo estimado **(E)**, lo supuesto **(S)**. Todo el código nuevo
está en [`scripts/red/`](../../scripts/red/).

## La máquina y lo que no deja hacer

| | |
|---|---|
| CPU | 4 vCPU Xeon a 2,1 GHz con AVX-512, VNNI y AMX; sin GPU (M) |
| RAM / disco | 15 GB / 30 GB libres (M) |
| Python | 3.11; torch 2.8 (rueda de PyPI, CUDA sin uso) y transformers 4.57.6 instalados desde PyPI (M) |
| Código | el repo de VibeVoice al commit fijado `94da20d`, desde GitHub (M) |
| **Red** | **`huggingface.co` y `download.pytorch.org` denegados por la política del entorno** (M) |

La denegación de `huggingface.co` bloquea de raíz todo lo que necesita pesos: el modelo
(`microsoft/VibeVoice-Realtime-0.5B`), el codificador comunitario, el tokenizador de Qwen2, los jueces (whisper,
ECAPA de speechbrain, UTMOS) y los corpus (CML-TTS, LibriTTS-R, CREMA-D). No se ha buscado ningún rodeo: es una
decisión del entorno. Para desbloquearlo hay que añadir `huggingface.co` a los dominios permitidos del entorno
o abrir la política de red.

Lo que sí había: los **25 prefijos de voz oficiales** (`demo/voices/streaming_model/*.pt`, 96 MB, en el repo de
GitHub). Son cachés KV y estados de salida calculados con los pesos reales sobre audio real: la red por dentro,
sin la red.

## 1. Los prefijos como estados internos reales (M)

`scripts/red/analizar_prefijos.py`. Por voz: la salida del `tts_lm` en las N posiciones de latente (el flujo de
la condición sobre habla real, 99-287 fotogramas por voz) y las M de texto, más K y V de las 20 capas. 100
fotogramas por voz, 25 voces (13 hombres, 12 mujeres, 11 idiomas). Sexo con **deja-una-voz-fuera** (¿generaliza
a una voz nunca vista?), identidad como fracción de la varianza total que está entre voces.

| Qué | Valor |
|---|---|
| RMS de la condición en latentes / en texto | 3,64 / 4,21 (norma ~109 en 896 dimensiones; constante a lo largo del prefijo: 109,0 · 108,7 · 109,1 por tercios) |
| Identidad en la condición (latentes) | **24,9 %** de la varianza es entre voces |
| Sexo en la condición, deja-una-voz-fuera | **95,5 %** de acierto |
| PCA de la condición | 10 componentes explican el 28,5 %; 20, el 41,9 %; la 2.ª y 3.ª correlacionan con el sexo (0,64 y 0,54), la 1.ª no (0,09) |
| Vecino más cercano por media de voz | mismo sexo el 92 %; mismo idioma solo el 20 % |

Por capa, en la caché de valores (V, sin RoPE) y de claves (K, con RoPE) en las posiciones de latente:

| Capa | Identidad en V (fracción de varianza entre voces) | Sexo en V, deja una voz fuera | Identidad en K |
|---|---|---|---|
| 0 | 0,065 | 0,596 | 0,012 |
| 1 | 0,061 | 0,708 | 0,034 |
| 2 | 0,082 | 0,814 | 0,040 |
| 3 | 0,099 | 0,829 | 0,073 |
| 4 | 0,048 | 0,800 | 0,022 |
| 5 | 0,043 | 0,781 | 0,070 |
| 6 | 0,056 | 0,855 | 0,037 |
| 7 | 0,032 | 0,694 | 0,057 |
| 8 | 0,036 | 0,690 | 0,022 |
| 9 | 0,024 | 0,581 | 0,035 |
| 10 | 0,029 | 0,662 | 0,037 |
| 11 | 0,034 | 0,534 | 0,026 |
| 12 | 0,047 | 0,737 | 0,027 |
| 13 | 0,084 | 0,819 | 0,029 |
| 14 | 0,048 | 0,639 | 0,020 |
| 15 | 0,074 | 0,811 | 0,022 |
| 16 | 0,098 | 0,890 | 0,031 |
| 17 | 0,076 | 0,922 | 0,046 |
| 18 | 0,099 | 0,961 | 0,115 |
| 19 | 0,080 | 0,916 | 0,073 |

**Lectura:**

- **La identidad tiene forma de U en profundidad:** fuerte en las capas 2-3 y 16-19, mínima en las 9-11 (2-3 % de la
  varianza; el sexo cae al 53-66 %). Si un mando de prosodia se suma en el residual, las capas 8-11 son donde
  menos identidad hay que respetar y las 16-19 donde más se arriesga. Es la primera respuesta medida a "dónde
  vive la identidad" (I4 del plan), aunque venga de 25 voces y no de una intervención.
- **El registro (sexo, que aquí es sobre todo tono) es lineal y generaliza a voces nuevas** (95,5 % en la
  condición, 96 % en la V de la capa 18). Una dirección de "voz más grave/aguda" existe y es global; la pregunta
  para I2 es si al sumarla con λ pequeña mueve el tono sin mover la persona.
- **El espacio se organiza por voz, no por idioma:** el vecino más cercano comparte sexo el 92 % de las veces e
  idioma solo el 20 %. Apoya que el estilo aprendido en un idioma se aplique en otro (F7) y que las direcciones
  de prosodia sacadas de datos en inglés tengan sentido en español (S hasta medirlo).
- **Solo un cuarto de la varianza de la condición es identidad**; el resto es contenido y prosodia: hay sitio
  para direcciones intrahablante.

Salvedades: 25 voces con 2 por idioma; el sexo es un proxy grueso del tono; los prefijos llevan lectura limpia de
estudio, no conversación. Sin F0 por fotograma (no hay audio de las referencias) no se pueden entrenar aquí las
sondas de prosodia.

## 2. La mecánica de las herramientas, probada con pesos aleatorios (M)

`scripts/red/probar_mecanica.py` construye la arquitectura del 0.5B (Qwen2 de 896 × 24 capas partidas 4 + 20,
cabeza de 896, tokenizador acústico) con pesos al azar, carga un prefijo real y comprueba propiedades que **no
dependen de los pesos**. Las ocho pasan (67 s en esta CPU):

| Prueba | Qué comprueba | Resultado |
|---|---|---|
| paridad | `bucle.Generador` da el mismo audio muestra a muestra que `generate()` de Microsoft con la misma semilla | ✅ 38.400 muestras idénticas |
| bifurcar | foto del estado en el fotograma 6 y reponer: los 6 primeros fotogramas idénticos, la continuación cambia con otra semilla y se repite bit a bit con la misma | ✅ |
| dirigir | sumar una dirección a la condición: λ = 0 deja el audio de base; λ = 0,1 lo cambia; una rama frente a dos ramas difieren | ✅ |
| énfasis | escalar la ficha 7 del texto solo cambia el audio **desde** la ventana que la lee (fotogramas 0-5 idénticos) | ✅ causal por construcción |
| negativo | un `neg_tts_lm` con 12 latentes propios delante se acepta y cambia el audio | ✅ |
| registro | por fotograma: condición, negativa, latente, residual de 20 capas, p_fin | ✅ |
| sorpresa | pérdida v de la cabeza por fotograma, finita | ✅ |
| atención | masa por cabeza sobre [latentes del prefijo, texto del prefijo, generado] suma 1 | ✅ (20 × 14 × 3) |

Con pesos reales las mismas propiedades valen y, además, los números significan algo. `probar_mecanica.py
--modelo <ruta>` lo repite con el modelo de verdad.

**Coste por fotograma en esta máquina** (forma del 0.5B, torch fp32, 4 hilos, prefijo de 381 posiciones):
235 ms, de los que `tts_lm` (dos pasadas) 115, decodificador 107 y difusión (6 pasos) 35 → RTF 1,76 (M). Con
pesos reales sería lo mismo (mismas formas); es el doble que la VM con OpenVINO int8 y da la escala de una
campaña de sondas aquí: unos 2 s de CPU por segundo de audio.

## 3. La tubería completa, en seco (M)

`instrumentar.py → sondas.py → dirigir.py` corre de punta a punta con pesos aleatorios (8 clips de 24 fotogramas,
192 fotogramas, dos voces). Lo que enseña el ensayo, y que ya está escrito en las herramientas:

- **La red codifica la posición exacta.** Con audio que es ruido, "fotogramas hasta el fin" y "ventana leída" se
  leen con R² > 0,98 en cualquier capa, y la energía (que en el modelo al azar es función de la posición) al
  0,92. Por eso `sondas.py` resta a cada etiqueta lo que explica un polinomio cúbico de la posición más la fase
  dentro de la ventana de 6, e imprime una fila `posicion` como suelo. Aun así, con clips todos de la misma
  duración el reloj sigue leyéndose: la campaña real tiene que usar duraciones distintas y fiarse solo de las
  etiquetas que no son función del tiempo (tono, energía, pausa, pregunta).
- `dirigir.py` suma o proyecta fuera una dirección en la condición (dos líneas, cero coste) o en el residual de
  la capa que se pida (entra en la caché de los fotogramas siguientes), en una rama o en las dos, desde el
  fotograma que se pida (la prueba de tiempo real), y mide los descriptores de `perfil_vocal.py` por clip.

## 4. Lo que NO se pudo evaluar aquí, y dónde sí

| Qué | Por qué no aquí | Dónde y cómo |
|---|---|---|
| Sondas I1 con prosodia real (tono, energía, pausas por fotograma) | sin pesos ni codificador (huggingface.co denegado) | Mac: `instrumentar.py --modelo ~/.cache/vibevoice-nix/modelo --voces ~/.cache/vibevoice-nix/voces --voz … --grupos es,en,es_numeros --semillas 11,101 --salida red/inst` (≈ 2 s de CPU por s de audio en un i7; menos en el M4) y después `sondas.py --inst red/inst --salida red/sondas` |
| Intervención causal I2 con puerta (WER, ECAPA, UTMOS) | sin pesos ni jueces | Mac: `dirigir.py --clave condicion/f0_st --lambdas 0,0.05,0.1,0.2 --rama ambas …` y `scripts/juez_lote.py` sobre la carpeta; la puerta es la del plan de emoción, E2 |
| Prefijo negativo con habla "plana" real de la persona (§5.1) | sin codificador para audio real | Mac: latentes por `forzado.latentes()` y `prefijo_negativo.construir()`; guardar como `neg_tts_lm` en el `.pt` de la voz |
| Cabezas de identidad y alineador (I4, §5.6) | sin pesos | Mac: `atencion.py --modelo … --voz sp-Spk1_man --salida red/atencion.json` (atención eager, torch) |
| Sorpresa frente a WER (§5.5) | sin pesos ni whisper | Mac: `instrumentar.py` sobre los 216 segmentos de F1 con sus dos semillas y `sorpresa.por_fotograma()`; correlación con el WER de `juez_lote.py` |
| Bifurcación en producción (§5.4) | el motor de la VM es OpenVINO con estado en `query_state`; aquí solo torch | VM: llevar `foto()`/`reponer()` a `motor.CacheOV`/`AcusticoOV` y probarlo con `ws_fidelidad.py` |
| RTF y memoria de cualquier cambio | no es la máquina de producción | VM voz, `banco_rtf_vm.sh` y `/crono` |
| Direcciones de emoción de pares reales (CREMA-D, guion con consentimiento) | sin datos ni red | Mac + g4dn, plan de emoción §3.2 |
| El 1.5B como maestro (§5.9) | 5,4 GB en huggingface.co, denegado | g4dn con `scripts/deriva/gen15b.py` |
| Sexo/tono como dirección global: ¿mueve el tono sin mover la persona? | la dirección existe (§1) pero sin pesos no se puede generar | Mac: `dirigir.py --clave condicion/…` con la dirección de sexo de `analizar_prefijos.py` (falta guardarla: una línea) |

## 5. Qué cambia en el plan después de esto

1. **La U de la identidad por capa** entra en el plan de la red como dato: dirigir en el residual de las capas
   8-11 primero, y medir ECAPA con más cuidado si se toca de la 16 en adelante.
2. **Las sondas necesitan el control de reloj** que ya llevan, y clips de duraciones distintas.
3. **El bucle transparente sustituye a los parches sobre `generate()`** para todo lo de laboratorio: es paridad
   exacta, y da fotos, enganches y registro sin tocar el código de Microsoft. Para producción sigue mandando
   `voz_stream.py` con OpenVINO.
4. Nada de lo anterior gasta GPU. La primera sesión en el Mac con pesos reales (instrumentar 2 voces × 12
   frases × 2 semillas, sondas, un barrido de λ con 4 valores) son unas 3-4 h de CPU.
