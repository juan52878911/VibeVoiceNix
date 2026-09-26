# Plan de mejora del modelo de voz (2026-09-22)

Informe de lectura. No se ha medido nada nuevo ni se ha tocado código. Cada cifra lleva su
fuente. Las etiquetas son las de siempre: **(M)** medido en el proyecto, **(E)** estimado con la
cuenta a la vista, **(S)** suposición sin medir.

Restricciones que manda el proyecto y que este plan respeta:

- Producción sigue en CPU (i7-8700T, 12 vCPU, VM de 5 GB) con RTF < 1 y VmHWM ~1,9 GB. Todo lo que
  solo funcione con GPU en inferencia va marcado.
- GPU disponible: una g4dn.xlarge o g6.xlarge bajo demanda (cuota 8 vCPU, spot 0) y 50 USD en
  total (`homelab-topologia.md`; precios en `docs/ec2-y-coste.md` L82: 0,526 y 0,805 $/h).
- Solo identidades con consentimiento; los `.pt`, huellas y clips no entran en el repo.
- Cada fase lleva puerta fijada antes de medir. Una puerta que no pasa se anota y cierra la palanca.

---

## Estado (2026-09-23, fases 0-6 sin intervención humana)

| Fase | Resultado | Informe |
|---|---|---|
| F0 jueces | Juez de pronunciación (PER, solo comparable dentro de la misma voz) y de sonidos (AST); corpus congelado | [f0](bancos/2026-09-22-f0-jueces.md) |
| F1 referencia y segundos | **Repetir un clip de 5 s pasa** en los dos hablantes; codo ≤ 12 s; la referencia por estilo no generaliza y el ritmo no se transfiere; el autoarranque no aporta | [f1](bancos/2026-09-23-f1-referencias.md) |
| F2 texto | **El normalizador baja el WER real 4 puntos** (inglés 13 % → 6 %); **reintentar los clips malos: −27 % de WER con +21 % de síntesis**; el juez viejo inflaba el WER de los números | [f2 y f5](bancos/2026-09-23-f2-f5-texto-y-no-verbales.md) |
| F3 acento por prefijo | No pasa: mezcla las voces | [f3](bancos/2026-09-23-f3-acento-prefijo.md) |
| F4 acento por conversión | **Pasa en parte**: con minutos de audio real, identidad al 96-109 % del clon, acento mucho más nativo y más natural; con una nota de voz corta no. Solo para cambio de idioma (F4c) | [f4](bancos/2026-09-23-f4-conversion-acento.md) |
| F5 no verbales por texto | No pasa: el modelo lee las marcas | [f2 y f5](bancos/2026-09-23-f2-f5-texto-y-no-verbales.md) |
| F6 ambientes procedurales | No pasa como realismo; siguiente paso con decisión humana (CC0) | [f6](bancos/2026-09-23-f6-ambientes.md) |
| F7 LoRA multilingüe | **No pasa, cerrada tras 3 corridas**: bucle validado (puertas 0 y 1), pero generando el WER no baja (base 4,4 %) y la identidad cae −0,008 a −0,012. Efecto consistente: las voces españolas en inglés suenan más nativas (UTMOS +0,2-0,5, PER −0,02-0,05) a costa de identidad; F4 lo hace mejor | [f7](bancos/2026-09-23-f7-lora.md) |
| **F8 LoRA por voz** | **Pasa** (Carlos, 10,8 min mezclados con lectura, al 75 % de fuerza): identidad +0,049, estilo −0,21 desviaciones suyas, UTMOS +0,08, WER igual; sobrevive a int4; con 5 min también. Falta llevarlo a producción (IR por voz) y validarlo con otra persona | [f8](bancos/2026-09-24-f8-lora-por-voz.md) |
| Guía destilada | No pasa (identidad −0,04 en dos corridas); código apagado como posible modo rápido | [guía](bancos/2026-09-23-guia-destilada.md) |
| Decodificador destilado | No pasa: el mismo WER, pero UTMOS 3,34 → 1,24 e identidad 0,82 → 0,44 (alumno a mitad de canales, desde cero, sin discriminador). Si se retoma: quitar bloques de la etapa 0 partiendo del maestro | [decodificador](bancos/2026-09-25-decodificador-destilado.md) |
| QC con whisper medium | No mejora (WER 0,311 frente a 0,225 de small, identidad −0,06): se queda small | [decodificador](bancos/2026-09-25-decodificador-destilado.md) |

Gasto en GPU: 9,74 USD de un tope de 20 (F7 2,74; guía destilada ~1,8; F8 ~2,25; decodificador destilado y juez del QC 1,35).

Siguiente plan, de lectura (26-09): [emoción e intención controlables al generar y en tiempo real](plan-emocion-intencion-2026-09-26.md).

Probado en dobla (rama `mejoras-modelo`): normalizador, repetir la referencia corta, reintento por QC
y acento nativo por conversión, sin regresión en el caso de 4 voces (WER 0,316 → 0,306). La demo de
una charla de 91 s a 5 idiomas con acento nativo costó ~0,11 USD en Batch spot (0,014 USD por minuto
de vídeo e idioma); de paso aparecieron y se corrigieron dos fallos de dobla para fr/de/it/pt
(la limpieza borraba ã ç è ß; la huella de reanudación no incluía el acento).

## 1. Diagnóstico: dónde está hoy el límite de cada eje

### 1.1 WER (inteligibilidad)

| Hecho | Cifra | Fuente |
|---|---|---|
| Clones buenos en producción, cfg 3,5, semillas 17/101 | WER 0,3-2,9 % por voz; sp-Spk3_man 14,7 % (problema de esa voz) | `docs/bancos/2026-09-15-lm-int8.md` L98-104 (M) |
| La semilla mueve 40 puntos; los pasos, 1 | 101/17: 0 %; 42: 40,7 %; 6→10 pasos: 11,8→10,7 % | `docs/plan-determinismo-calidad.md` L180-184, L228-247 (M) |
| El arranque concentra el error | 4,2 % de WER en las tres primeras palabras frente a 0,0 % en el resto; se corrige con cfg 4,5 en rampa | `pkgs/vibevoice-cli/voz_stream.py` L897-915 (M) |
| Ruido de arranque semilla 1 | WER −1,4 puntos, UTMOS +0,27, ECAPA −0,016 (no pasa la puerta de identidad, adoptado por decisión) | `voz_stream.py` L956-960; `musica-inventada-ingles.md` (M) |
| Con material real, el clon ya se entiende mejor que el original | ΔWER −0,09 a −0,17 (whisper large-v3) | `banco-reconstruccion.md`; `~/Documents/dobla-recon/informe.txt` (M) |
| Ningún parámetro de inferencia baja el WER; dos lo suben | 10 pasos +0,040, arranque 23 +0,060 | `banco-reconstruccion.md` (M) |
| Doblaje real de 74 min | WER mediano 0,09; QC es el 47 % del coste del job | `dobla/README.md` L565-577 (M) |
| Referencia mala = WER catastrófico | Sebastián r60: 44 % en inglés; Juan Pablo (techo 0,51): 11-22 % según semilla | `docs/clonado-de-voz.md` L713-722, L838-841 (M) |
| Trozos cortos sin contexto inventan o repiten la transcripción de la referencia | fase 4 `trozos`/`saltos` | `docs/plan-personalidad-voz.md` L364-371 (M) |
| Otros idiomas con prefijo español | francés 43 % WER, alemán 22 % | `docs/comparativa-motores.md` L71-76 (M) |

**Límite real:** el WER residual no está en cfg, pasos ni motor; está en (1) la referencia, (2) el
sorteo de semilla por texto y voz, (3) el arranque, y (4) un puñado de frases que el modelo falla con
casi cualquier semilla (`plan-determinismo-calidad.md` L320-323). Las tres primeras ya tienen palanca
sin tocar pesos. La cuarta y el salto a otros idiomas solo se mueven con pesos.

### 1.2 Identidad (ECAPA)

| Hecho | Cifra | Fuente |
|---|---|---|
| Techo del códec (audio real, ida y vuelta) | 0,742-0,835 según hablante | `~/Documents/dobla-recon/controles.json` (M) |
| Clones con material real | 0,456-0,790 del techo; semilla: 16 semillas en 0,490-0,519 (σ 0,0087) | `banco-reconstruccion.md`, `informe.txt` (M) |
| Lo único que movió la identidad | centroide fijo al elegir clips: +0,10 en Sebastián, +0,048 [+0,022, +0,075] sobre la regla vieja | `dobla-fase1/pipeline/referencias.py` docstring; `banco-reconstruccion.md` (M) |
| El techo de la grabación manda sobre el motor | VibeVoice y Qwen3 rinden 46-61 % del techo de cada voz | `docs/comparativa-motores.md` L199-222 (M) |
| Solape con otra voz destruye el techo | 0,019 → 0,890 al quitar tramos pisados | `docs/clonado-de-voz.md` L663-668 (M) |
| Pérdida al cambiar de idioma | Avril 0,696 es / 0,571 en; Sebastián 0,699 / 0,488; Laura 0,65 / 0,45 | `docs/clonado-de-voz.md` L853-856; `comparativa-motores.md` L201 (M) |
| Corregir el tono a la salida no compensa | PSOLA: ECAPA −0,10 y WER +10 puntos | `docs/clonado-de-voz.md` L752-787 (M) |
| Adaptador FiLM en la difusión (fase 3b) | juez +0,22 pero UTMOS −0,33, WER +15,7, ECAPA −0,03 | `docs/plan-personalidad-voz.md` L266-282 (M) |

**Límite real:** la referencia (pureza, homogeneidad, SNR) y el idioma destino. La inferencia está
agotada. Subir el 60-80 % del techo actual exige pesos o un cambio de mecanismo (conversión de voz).

### 1.3 Naturalidad (UTMOS) y personalidad

| Hecho | Cifra | Fuente |
|---|---|---|
| Clones frente a voces de serie | UTMOS 1,9-3,1 frente a 3,0-3,6 | `docs/plan-personalidad-voz.md` L7-11 (M) |
| Lo que falta no es tono: es ritmo, pausas, dinámica y armonicidad | Carlos: pausas 20,9 → 10,2/min, sílabas/s 5,22 → 4,63, HNR −0,34 dB; recorrido tonal SUBE +0,99 st | `plan-personalidad-voz.md` L88-103 (M) |
| Salvedad que contamina la medida | el real es conversación espontánea separada con demucs; el clon lee | `plan-personalidad-voz.md` L105-107 |
| Juez de personalidad válido | AUC ida-y-vuelta/clon 0,999, real/ida-y-vuelta 0,628; pesos en la VM (perdidos con la reconstrucción del 15-09: `/var/lib/taller` ya no existe) | `plan-personalidad-voz.md` L146-165; `docs/plan-rendimiento.md` L581-582 |
| El juez solo se engaña con audio peor | siempre acompañado de WER, UTMOS y ECAPA | `plan-personalidad-voz.md` L291-293 (M) |
| `forma` (re-durar pausas por perfil) integrado, apagado por defecto | sílabas/s −0,215 [IC<0], WER −0,07, UTMOS −0,004 | `plan-personalidad-voz.md` L429-463 (M) |
| WSOLA de velocidad cuesta naturalidad | UTMOS −0,14 a −0,25 | `plan-personalidad-voz.md` L409 (M) |
| El respiro solo puede alargar donde el modelo ya pausó; no hay ancla de frontera | EOS plano en fronteras internas | `voz_stream.py` L2171-2178 (M) |

**Límite real:** el ritmo y las pausas los decide el LM; el adaptador sobre la difusión no
generaliza y el posproceso sobre la señal (WSOLA, PSOLA) rompe UTMOS o identidad. El único
posproceso que pasa es el que no toca la voz (`forma`). Lo que queda es (a) condicionar por texto y
referencia, (b) pesos del LM.

### 1.4 Acento

**Sin medir.** No hay juez de acento en el proyecto. Los indicios:

- El prefijo manda la fonética: con prefijo español, francés y alemán salen con el timbre de la
  persona pero con 43 % y 22 % de WER (`comparativa-motores.md` L89-91). El modelo imita en
  contexto los latentes acústicos de la referencia, y esos latentes llevan la pronunciación
  española.
- La identidad cae 0,1-0,2 al pasar al inglés (§1.2), coherente con que el modelo negocie entre
  "esta voz" y "esta lengua".
- El corpus del 0.5B es inglés y chino; el español es experimental (`README.md` L78-82).

En este modelo no existe un mando de acento: no hay token de lengua ni de estilo. El acento sale de
tres sitios: los latentes del prefijo (rama `tts_lm`, N posiciones), la transcripción del prefijo
(rama `lm`, M posiciones) y el texto que se lee (`docs/clonado-de-voz.md` L100-132).

### 1.5 Segundos de referencia

| Hecho | Cifra | Fuente |
|---|---|---|
| Curva con voces sintéticas del propio modelo | 11,7 s → 0,834-0,854; 23 s → 0,848-0,885; techo 0,88 | `docs/clonado-de-voz.md` L513-529 (M) |
| Nota de WhatsApp real, 6,8 s de voz, recortada, 17 kbps | clon 0,753 de media, por encima del techo de esa grabación (0,717) | `clonado-de-voz.md` L445-464 (M) |
| Demo en vivo, 14,3 s | 0,739 con techo 0,847; clonar en caliente 1,4 s; codificar 0,05 s por s de audio | `metricas-demo-clonado-vivo.md` (M) |
| Más audio dispar empeora | 31,4 s → 0,684; +9,6 s de otras salas → 0,592 | `clonado-de-voz.md` L556-571 (M) |
| Un prefijo corto genera más rápido | 168 posiciones: RTF 0,70-0,81; oficial largo: 1,12-1,14 (M4) | `metricas-demo-clonado-vivo.md` (M) |
| El denoiser sobre la referencia baja la identidad | 0,51 → 0,41; el paso alto a 100 Hz no | `proyecto-dobla.md` (realismo integrado) (M) |

**Límite real:** con 12 s homogéneos ya se está al 85-95 % del techo. El hueco entre 12 y 25 s vale
+0,015 a +0,03. "Menos segundos" es sobre todo elegir **qué** segundos y aguantar audio malo, no
un problema del codificador. Por debajo de ~8 s no hay medida con voz real limpia.

### 1.6 Sonidos no verbales y ambiente

- El modelo **sí genera** no-habla: inventa sintonías en intros en inglés (conducta del corpus de
  podcasts), y ninguna palanca de inferencia la quita salvo el ruido de arranque, que la convierte
  en "todo o nada por texto" (`voz_stream.py` L933-963) (M).
- La cola insistente hizo que el modelo **empezara otra palabra** al no dejarle parar
  (`voz_stream.py` L2422-2437) (M): pedirle audio que el texto no pide es inventar contenido.
- Las risas del original no se detectan de forma fiable (modulación 3-8 Hz indistinguible del
  habla); dobla las conserva por ventanas marcadas a mano (`proyecto-dobla.md`) (M).
- El ambiente en dobla ya se hace por mezcla: fondo de demucs, tono de sala del propio vídeo,
  sala sintética RT60 0,45 s, EQ por hablante; el usuario lo dio por convincente (`proyecto-dobla.md`).
- No hay ninguna medida de si "jaja", "(ríe)", "mmm" o "eh" producen el sonido en este modelo. (S)

---

## 2. Qué hay para cada punto (a-f)

Convención: **sin pesos** = condicionamiento, prompt de audio, preprocesado de la referencia,
posproceso. **Con pesos** = LoRA, adaptador, cabeza nueva. Un cambio de pesos del `tts_lm` o de la
cabeza **invalida todos los `.pt`** (son cachés KV calculadas con los pesos viejos): hay que
refabricarlos con el codificador comunitario, recuantizar, reconvertir los IR (4,6 GB, ~15 min) y
pasar `banco_ab.py` (3 h). No cambia el RTF: misma arquitectura, mismos bytes.

### a) Bajar el WER

Sin pesos, en orden de valor/coste:

1. **Semilla y arranque por voz y por corpus** (ya existe: `banco_semillas.py`, `elegir_arranque.py`,
   `perfiles.py`). Falta que dobla lo aplique a cada identidad nueva del banco y no solo a las tres del
   asistente. Coste: 15-20 min de VM por voz (M, `perfilador-voz-fase2.md`).
2. **Preprocesado del texto** en `voz_stream.py`: números a palabras, siglas deletreadas, extranjerismos
   con grafía fonética, normalizar puntuación (dobla ya hace `numeros_a_palabras` y `limpiar_texto`;
   el servicio no). Medible con `fidelidad.py` sobre las dos frases que fallan siempre
   (`plan-determinismo-calidad.md` L320-323). Coste: horas de Mac.
3. **Mejor de N en lote** (solo dobla, nunca en directo): sintetizar cada turno con 2 semillas de la
   lista buena de la voz y quedarse con el de menor WER. Hoy el QC solo re-tira si WER > 25 %. Coste:
   +1 síntesis por turno (~+33 % del tiempo de síntesis, que es el 33 % del job: +11 % del job) (E).
4. **Trozos con contexto**: el tope de 220 caracteres cerrando frase ya está. No bajar más: los
   trozos cortos inventan (M).

Con pesos:

5. **LoRA del `tts_language_model` (20 capas) y la cabeza de difusión en español latinoamericano e
   inglés**, con pérdida de difusión sobre latentes reales (v-prediction, como `fase3_adaptador.py`
   L20-25) y la pérdida del clasificador de EOS. Datos: los 16,9 min con consentimiento (76 % Carlos)
   más corpus públicos. Candidatos con licencia compatible: Common Voice es (CC0), MLS es (CC BY 4.0,
   lectura de LibriVox, acento peninsular mayoritario), Google Crowdsourced Latin American Spanish
   (CC BY-SA 4.0: revisar si el share-alike alcanza a los pesos). TEDx Spanish es CC BY-NC-ND: fuera.
   La cadena de datos existe a medias: `fase3_condiciones.py` ya fuerza latentes reales y guarda la
   condición del LM; el codificador comunitario cuesta 0,05 s por s de audio (M). El bucle de
   entrenamiento del Realtime-0.5B **no está publicado por Microsoft** y la receta comunitaria que
   existe es para el 1.5B (sin comprobar): hay que escribirlo, y es la parte cara en horas humanas.
   GPU: 0,5B en fp16 con LoRA cabe de sobra en una T4 (16 GB); una secuencia de 30 s de referencia +
   20 s de habla son ~550 posiciones. 10 h de audio a 20 s por muestra son 1.800 muestras; a ~0,3 s
   por muestra (E) una época son ~9 min; 10 épocas ~1,5 h; 3 corridas con ajuste de tasa ~5 h →
   **~3 USD en g4dn** (E). Lo que cuesta de verdad son las puertas: cada variante pasa `banco_ab.py`
   (3 h en el LXC 204) y refabricar las voces.
   Ganancia esperada: WER de las frases "malas" y de las voces con referencia floja; en las buenas no
   hay margen (0,3-2,9 %). (E) Riesgo: la identidad de las voces oficiales cambia (sus `z` sí se
   pueden recuperar, `clonado-de-voz.md` §3), y el estilo "leído" de MLS empuja en contra de la
   personalidad (b). Por eso el corpus debe ser conversacional o del propio banco de dobla.

### b) La capa de personalidad

Lo medido cierra dos caminos: FiLM sobre la condición de la difusión (rompe el habla) y posproceso
que toque la voz (WSOLA, PSOLA). Queda:

Sin pesos:

1. **Referencia por estilo, no solo por techo.** Hipótesis (S): el prefijo transfiere el ritmo del
   material. Con una referencia de conversación espontánea el clon debería pausar y acelerar como la
   persona; con una leída, leer. Se comprueba con Carlos (12,8 min) partiendo su material por
   pausas/min y sílabas/s del propio clip y midiendo el clon con los descriptores de la fase 1.
   Coste: 1 h de VM. Este experimento además limpia la salvedad "leer frente a conversar".
2. **Transcripción del prefijo con las muletillas reales.** La rama `lm` es el texto que se oye en la
   referencia. Si la transcripción literal incluye "eh", "o sea", "..." donde la persona los dice, el
   modelo ve el patrón. Hoy dobla las quita (`limpiar_texto`). Medible con el juez de personalidad,
   WER y cobertura. Coste: horas.
3. **Muletillas y respiraciones por perfil, insertadas en el texto** ("eh," cada K palabras según el
   perfil real). Riesgo conocido: cualquier cosa que cambie el texto pasa por la puerta de cobertura de
   la fase 4 (ningún clip con WER > 25 % si la base estaba ≤ 10 %; cobertura 0,85-1,15).
4. **Respiración real de la persona en el respiro.** El respiro inserta suelo de sala en espejo
   (`voz_stream.py` L2271-2292). Sustituirlo por una respiración recortada de la referencia (detectada
   con AST "Breathing" o por energía) no toca la voz, así que la garantía estructural de `forma` se
   mantiene. Medible con UTMOS y el juez. Coste: horas; CPU cero en producción.

Con pesos:

5. **Afinado por voz del `tts_lm` (LoRA de rango bajo), no de la difusión.** Es el camino estándar
   de "afinar una voz con minutos de audio": el LM decide ritmo y pausas, que es justo lo que falta. La
   fase 3a enseñó dos cosas: con 8,8 min (Carlos) se aprende, con 1,7 min no; y entrenar con la
   condición del audio real y generar con la del propio clon abre una brecha (`plan-personalidad-voz.md`
   L284-290). El LoRA sobre el LM con forzado de latentes reales sufre lo mismo (exposure bias); la
   defensa es regularizar hacia la base y pasar la puerta de la 3b tal cual (juez IC>0, distancia a
   los 8 descriptores IC<0, ECAPA ≥ −0,005, UTMOS ≥ −0,02, WER ≤ +0,5). Solo tiene datos Carlos.
   GPU: ~1 h por corrida en g4dn (E); en producción, el LoRA se funde en los pesos por voz, lo que
   significa **un IR del LM por voz** (int4, ~150 MB cada uno) o mantener el LoRA como MatMul aparte
   en el grafo (coste por fotograma sin medir; con rango 8 son 20 capas × 2 matrices de 896×8:
   despreciable en FLOPs, pero cada nodo nuevo en OpenVINO/AVX2 cuesta despacho). Esto solo es
   producto si el onboarding recoge ≥ 10 min por persona.
6. **Codificador de estilo → adaptador del LM** (generaliza a voz nueva): descartado con 5
   identidades (`plan-personalidad-voz.md` L209-212). Necesita cientos de hablantes con
   consentimiento. No es de esta etapa.

### c) Menos segundos de referencia

Sin pesos:

1. **Curva con voces reales, no sintéticas.** La curva de §7.8 está medida con clips generados por el
   modelo (`clonado-de-voz.md` L490-493). Falta la misma curva con Carlos, Liliana, Avril y Sebastián
   (5, 8, 12, 20, 30 s), juzgada contra apartados con `evaluar_clones.py`. Sin esto no se sabe dónde
   está el codo real. Coste: Mac clona (1,4 s por voz en caliente), VM sintetiza (20 min por voz).
2. **Elección automática de los mejores 10-12 s** dentro de lo que haya: consistencia al centroide fijo
   + SNR + pureza. Ya existe casi todo (`perfil_voz.py`, `referencias.py`, `mejor_referencia.py`);
   falta el modo "presupuesto de 10 s" y su puerta.
3. **Repetir los latentes** del mismo clip corto (5 s → 10 posiciones dobles). Hipótesis (S): el
   prefijo pesa más y el modelo se ancla mejor; riesgo de que el LM detecte la repetición. Experimento
   de una tarde: clon con 5 s, clon con 5 s ×2, clon con 5 s ×3; ECAPA contra apartados. Criterio de
   abandono: si ×2 no gana ≥ +0,02 sobre ×1 con IC separado, cerrar.
4. **Autoarranque** (bootstrapping): clonar con 6 s, sintetizar 25 s de texto neutro con la mejor
   semilla, re-codificar esa síntesis y fabricar el prefijo con 6 s reales + 20 s sintéticos. La curva
   de §7.8 ya demuestra que el modelo se clona a sí mismo con 0,85. Riesgo (S): deriva hacia la voz
   media del modelo (el sesgo de tono de las voces graves). Puerta: ECAPA contra apartados ≥ clon de
   6 s + 0,03 y tono dentro de ±0,5 st del real.
5. **No** aplicar denoiser (M: baja identidad); sí paso alto a 100 Hz e igualar RMS (ya en
   `clonar_voz.py` L206-218).

Con pesos:

6. **Prefijo destilado (soft prompt) por voz**: aprender por gradiente 60-80 posiciones de KV que
   reproduzcan la salida del prefijo largo de 30 s. Reduce caché y latencia (un prefijo corto genera
   más rápido, M) y podría concentrar identidad. Es entrenamiento por voz de ~1e5 parámetros con el
   modelo congelado; cabe en el Mac en minutos (S). Baja prioridad: resuelve latencia más que
   segundos de entrada.
7. **Generador de prefijo desde una huella** (ECAPA → KV): necesita cientos de hablantes. Fuera.

### d) Acento a voluntad

Ver §3.

### e) Sonidos no verbales y ambiente

Separar dos cosas que Juan junta:

**Sonidos de la persona** (risa, duda, respiración, carraspeo): tienen que salir de la síntesis o
de su propia grabación. Tres vías sin pesos, por orden:

1. **Marcadores en el texto**: "jaja", "ja, ja", "(risas)", "mmm", "eh...", "hmm", "ah". El modelo
   viene de podcasts y no está medido qué hace. Experimento cerrado: 4 voces × 8 marcadores × 6
   semillas = 192 clips en la VM (~15 min). Juez: AST de AudioSet (ya integrado en dobla) con las
   clases Laughter, Breathing, Speech, Music; WER del texto que rodea al marcador; UTMOS. Puerta
   fijada: el marcador produce la clase objetivo (AST > 0,3) en ≥ 70 % de los clips, cero clips con
   Music > 0,2, WER del resto sin subir más de 0,5 puntos. Si no pasa para risa, se cierra la vía
   generativa para risa y se va a la 3.
2. **Risa en la referencia**: un clip de 2-3 s de risa real de la persona, con transcripción "jajaja",
   como segundo clip del prefijo. Riesgo (M): los clips heterogéneos bajan la identidad 0,09. Puerta:
   ECAPA ≥ −0,02 frente al clon sin risa y la vía 1 mejora ≥ 20 puntos.
3. **Banco de no-verbales de la persona**: recortar de su grabación risas, respiraciones y "eh"
   (AST + energía) y colocarlos por montaje donde el original los tiene (dobla ya protege ventanas y
   ya tiene la anotación). Es lo que hoy se hace a mano con `original.json`; falta automatizar la
   detección con AST en vez de por modulación. Coste CPU: cero en síntesis.

**Ambiente** (oficina, teclado, calle): **no debe generarlo el modelo de voz.** Argumentos medidos:
la única no-habla que genera hoy es incontrolable (música inventada); cada fotograma de no-habla
cuesta lo mismo que uno de habla (RTF); y ensuciaría el prefijo y la identidad. El sitio correcto es
la mezcla, que ya existe en dobla (fondo separado, tono de sala, sala sintética, ducking) y que en
`voz_stream.py` sería una capa de posproceso barata: banco de ambientes CC0 en bucle sin costura
(freesound CC0 o grabados en casa), nivel relativo por perfil (−25 a −35 dB bajo la voz), reverb
corta compartida entre voz y ambiente para que "estén en la misma sala". Coste: convolución FFT de
un canal, < 1 % del RTF (E). Generar ambiente con un modelo de audio (AudioLDM2, Stable Audio Open)
exige GPU y sus pesos son no comerciales (CC BY-NC / licencia comunitaria de Stability): fuera.

Con pesos: enseñar al modelo a reír o respirar a partir de "(ríe)" es un afinado con datos
etiquetados de no-verbales (los del banco de dobla, si la vía 3 los extrae) sobre el mismo LoRA
del punto (a). Solo si la vía 1 falla y el volumen de datos lo permite (≥ 200 eventos). (E)

### f) Producto eficiente y muy natural

La palanca más grande, medida tres veces, es la grabación de referencia (46-61 % del techo, y el
techo lo pone el micro). El onboarding con consentimiento debe capturar 25-30 s seguidos, micro
cerca, sin música, y si se quiere personalidad, ≥ 10 min de conversación (no lectura). Sin eso
ningún punto de este plan llega a la zona buena. Lo dice también `licencias-y-legal-doblaje.md`.

---

## 3. El acento: qué hace falta para desacoplarlo del timbre

### 3.1 Dónde vive cada cosa en este modelo

- **Timbre**: latentes acústicos del prefijo (N posiciones de la rama `tts_lm`), 64 números a
  7,5/s. Es lo que el LM imita en contexto.
- **Pronunciación**: la decide el `tts_lm` al mapear los tokens de texto a latentes, **condicionado
  por esos mismos latentes**. No hay una variable separada. Por eso "voz española hablando inglés"
  sale con fonética española: el modelo continúa lo que oye.
- **Lengua del texto**: la ve el LM de texto de 4 capas; no hay token de idioma ni de estilo.

Desacoplar = darle al modelo latentes de timbre de la persona pero fonética de otro hablante, o
cambiar el timbre después de generar.

### 3.2 Un juez de acento antes de tocar nada

No existe y sin él no hay puerta. Propuesta barata y medible: **tasa de error de fonemas (PER)
contra la pronunciación nativa**. Reconocedor de fonemas IPA `facebook/wav2vec2-lv-60-espeak-cv-ft`
(MIT) sobre el audio, y `espeak-ng` como G2P del texto en la lengua destino con acento nativo
(en-us). Se calibra con controles: voz oficial inglesa del modelo diciendo el texto (PER bajo),
clon español diciendo el mismo inglés (PER alto), audio real de un nativo si lo hay. La medida
vale si separa los dos controles con IC. Complemento: clasificador de acento inglés de speechbrain
(CommonAccent, 16 acentos, ninguno "spanish": sirve solo como "cuánto de us/england suena").
Un juez L2 (L2-ARCTIC tiene 4 hablantes con L1 español) valdría para medir, pero su licencia es
CC BY-NC: solo como juez interno, nunca para pesos del producto, y a revisar con abogado.

### 3.3 Caminos baratos (sin pesos), en orden

| Camino | Qué se hace | Qué se espera (S) | Puerta |
|---|---|---|---|
| **B1. Referencia de la misma persona en la lengua destino** | si la persona habla inglés, 25-30 s suyos en inglés como prefijo para el inglés | acento real de esa persona en inglés (que puede seguir siendo español: es su acento); identidad en inglés sube porque desaparece el conflicto es/en | ECAPA en inglés ≥ +0,05 sobre el clon español; PER se anota |
| **B2. Prefijo mixto** | N latentes de la persona (es) + N' latentes de una voz oficial inglesa nativa con su transcripción; probar 30/10, 20/20 s y el orden | el modelo promedia: algo de timbre se pierde (M: −0,09 con clips dispares) y algo de fonética inglesa se gana | ECAPA ≥ −0,03 frente al clon puro **y** PER ≤ punto medio entre los dos controles; si no, cerrar |
| **B3. Transcripción del prefijo traducida** | mismos latentes españoles, rama `lm` con la traducción inglesa | casi seguro que falla (la receta exige transcripción literal, `clonar_voz.py` L39-41); se mide porque cuesta 20 min | igual que B2; criterio de abandono al primer barrido |
| **B4. Conversión de voz tras sintetizar** | generar con una voz que tenga el acento deseado (oficial inglesa nativa → inglés nativo; clon español de otra persona hablando inglés → inglés con acento español; voz oficial española → español con acento español; y para "español con acento inglés", un hablante inglés con consentimiento leyendo español) y **convertir el timbre** al de la persona con un modelo de conversión de voz | el acento lo pone la fuente; el timbre, la conversión. Es el único camino que da los cuatro cuadrantes sin datos apareados | ECAPA del convertido ≥ 0,9 × el del clon directo en esa lengua; PER ≤ el de la fuente + 10 %; UTMOS ≥ −0,10; WER = el de la fuente |

Sobre B4, lo concreto:

- Candidatos con licencia y CPU: **kNN-VC** (código MIT; WavLM-large MIT + regresión kNN + HiFi-GAN
  MIT): sin entrenamiento, 30 s de referencia bastan, calidad razonable en la literatura; WavLM-large
  son 315 M parámetros: RTF en el i7 sin medir, del orden de 0,3-0,6 (E) → **solo lote (dobla), no
  directo**. Seed-VC (GPL/MIT según versión, difusión, mejor identidad publicada) es de GPU.
  RVC exige entrenar por voz (10 min de audio, GPU). Empezar por kNN-VC en el Mac.
- La cadena es-con-acento-inglés necesita una fuente que hable español con acento inglés: no hay
  voz oficial así. Vale un hablante nativo inglés con consentimiento leyendo 30 s de español, o el
  clon de una voz oficial inglesa leyendo español (medido en francés: 43 % de WER con prefijo
  ajeno; en español el modelo sí tiene datos, así que puede salir mejor: sin medir). El WER manda.
- Coste en producción: cero en la VM (no se despliega); en dobla, +RTF de la conversión por segmento
  (E: ×1,3-1,6 del tiempo de síntesis). Memoria: WavLM-large ~1,3 GB en fp32; cabe en la m8a de
  16 GiB, no en la VM voz junto al motor.

### 3.4 Camino caro (con pesos)

**Afinar el `tts_lm` con un token de acento** exige datos etiquetados en las cuatro celdas
(es-nativo, es-con-acento-inglés, en-nativo, en-con-acento-español). Lo que existe:

- en-con-acento-español: Common Voice inglés tiene campo de acento (CC0, texto libre, hay que
  filtrar "Spanish"; volumen sin medir); L2-ARCTIC (CC BY-NC: no para pesos comerciales).
- es-con-acento-inglés: prácticamente **no hay corpus**. Habría que grabarlo o sintetizarlo con B4.
- Y aun con datos, el modelo nunca vio "esta persona con otro acento": el token aprendería acento
  medio, no acento de esa voz. Eso sí desacopla, porque el timbre sigue viniendo del prefijo.

Coste GPU (E): 20-40 h de audio, LoRA en g4dn, 3-6 h de entrenamiento (~2-3 USD); el coste real
es preparar y filtrar datos (días) y validar (banco_ab + juez de acento + refabricar voces).
**Veredicto:** no arrancar este camino hasta que B4 haya fallado con puerta, porque B4 da lo mismo
sin datos ni pesos. Y la celda es-con-acento-inglés es inviable con pesos por falta de datos.

---

## 4. Plan por fases con puertas

Orden por valor/coste. **Esta semana, sin GPU:** F0 a F5. Máquinas: Mac = clonar, juzgar (ECAPA,
whisper large-v3, UTMOS, AST, PER), minutos; VM voz = sintetizar con producción (candado: una
locución a la vez, avisar si la comparte otra sesión); LXC 204 de pve = `banco_ab.py` (3 h);
g4dn = solo F6-F8.

### F0 · Jueces y corpus de prueba (Mac, 3-4 h)

- Hipótesis: sin juez de acento ni de no-verbales no hay puerta posible.
- Experimento: montar PER (wav2vec2-espeak + espeak-ng) y el juez AST por clases; calibrar el PER con
  los tres controles (voz oficial en, clon es→en, real nativo si lo hay). Congelar un corpus de
  prueba bilingüe: las 8 frases de `evaluar_clones.py` + 4 con números y siglas + 4 con marcadores
  no verbales; apartados por hablante como en la charla del 14-09. Recuperar o reentrenar el juez de
  personalidad (los pesos de `/var/lib/taller/fase2b` se perdieron el 15-09; `fase2_idavuelta.py`
  lo rehace en ~30 min de VM en modo taller).
- Métrica: el PER separa los dos controles con IC 95 % disjunto. Si no, el PER no vale y se busca
  otro juez antes de F3.
- Coste: 3-4 h de Mac. Abandono: no aplica.

### F1 · Referencia por estilo y curva real de segundos (Mac + VM, 1 día)

- Hipótesis: el prefijo transfiere ritmo; el codo de la curva con voz real está por debajo de 12 s.
- Experimento: Carlos y Liliana. (a) tres referencias de 30 s: espontánea (pausas/min alto),
  neutra, la de dobla; (b) 5/8/12/20/30 s elegidos por centroide fijo; (c) 5 s ×1, ×2, ×3 latentes
  repetidos; (d) autoarranque 6 s + 20 s sintéticos. Clonar en el Mac (`--lote`), sintetizar en la VM
  con `evaluar_clones.py`, semillas de síntesis 11 y 101.
- Métrica: ECAPA contra apartados, WER (es/en), UTMOS, descriptores de la fase 1 (pausas/min,
  sílabas/s).
- Puerta: (a) pasa si la referencia espontánea acerca pausas/min y sílabas/s al real con IC < 0 y
  WER ≤ +0,5; (b) informa el codo; (c) y (d) pasan con ECAPA ≥ +0,02 [IC separado] sobre el clon de
  5 s sin subir WER ni mover el tono más de 0,5 st.
- Coste: ~2 h de Mac, ~2 h de VM. Abandono: (c) y (d) se cierran al primer barrido negativo.

### F2 · Preprocesado de texto y mejor-de-2 en dobla (Mac + VM, 1 día)

- Hipótesis: parte del WER residual es de grafía (números, siglas, extranjerismos) y de sorteo.
- Experimento: normalizador en `voz_stream.py` detrás de un campo `normalizar` (apagado por
  defecto); `fidelidad.py` con las 6 frases + 6 nuevas con números/siglas, 6 semillas. En dobla,
  mejor-de-2 sobre el tramo de prueba de `pruebas/tramos.py`.
- Puerta: WER medio −1 punto con IC < 0 y ningún clip peor de 25 % si antes ≤ 10 %; md5 idéntico
  con el campo apagado. Mejor-de-2: WER mediano del tramo −20 % relativo con +≤ 15 % de tiempo de
  job.
- Coste: 2 h Mac, 1 h VM, 1 job de Batch (~0,05 USD). Abandono: si el normalizador no mueve el WER
  medio, se queda solo como opción.

### F3 · Acento barato B1-B3 (Mac + VM, 1 día)

- Hipótesis: el prefijo mixto mueve el PER hacia el nativo sin hundir la identidad.
- Experimento: Avril y Sebastián (tienen apartados) + Carlos. Prefijos: puro es (control),
  mixto 30/10, 20/20, 10/30 con sp-Spk oficial inglesa (la que más se parezca por F0), transcripción
  traducida (B3), y si alguien tiene audio en inglés, B1. 8 frases en inglés, 2 semillas.
- Puerta: B2 pasa si PER ≤ punto medio entre controles **y** ECAPA en ≥ −0,03 frente al puro **y**
  WER ≤ +0,5. B3 igual. Se elige la mezcla con menor PER entre las que pasan.
- Coste: 1 h Mac, 2 h VM. Abandono: B3 se cierra al primer barrido; B2 si ninguna mezcla pasa.

### F4 · Conversión de voz B4 (Mac, 1-2 días)

- Hipótesis: kNN-VC conserva ≥ 90 % de la identidad del clon directo y hereda el acento de la
  fuente.
- Experimento: fuentes = voz oficial en (inglés nativo) y clon es de otra persona (inglés con
  acento español); destino = Avril, Sebastián, Carlos con 30 s de referencia real. Las mismas 8
  frases. Medir también RTF en el Mac y en la VM (sin desplegar; solo cronometrar).
- Puerta: ECAPA ≥ 0,9 × clon directo en inglés; PER de la conversión ≤ PER de la fuente × 1,1;
  UTMOS ≥ −0,10; WER = fuente ± 0,5. Para "español con acento inglés": una fuente con consentimiento
  leyendo español (30 s) y la misma puerta.
- Coste: 4-8 h Mac. Abandono: si la identidad cae por debajo de 0,8 × en las tres voces, probar
  Seed-VC en g4dn (1-2 USD) una sola vez; si tampoco, cerrar el acento sin pesos.

### F5 · No verbales por texto (VM + Mac, medio día)

- Hipótesis: algún marcador textual dispara risa/duda/respiración de forma controlable.
- Experimento: 4 voces × 8 marcadores × 6 semillas, ruido de arranque 1 y 0.
- Puerta: AST clase objetivo > 0,3 en ≥ 70 % de los clips del marcador, 0/192 con Music > 0,2,
  WER del texto circundante ≤ +0,5. Las risas de referencia (vía 2) solo si la 1 pasa a medias
  (40-70 %).
- Coste: 15 min VM, 1 h Mac. Abandono: si ningún marcador pasa el 40 %, la vía generativa se cierra
  y se hace el banco de no-verbales de la persona por montaje (vía 3).

### F6 · Respiración real y ambiente por mezcla (Mac + VM, 1-2 días)

- Hipótesis: sustituir el suelo en espejo del respiro por una respiración real de la persona, y
  añadir ambiente por mezcla, sube UTMOS y el juez sin tocar la voz.
- Experimento: en `voz_stream.py`, detrás de campos apagados por defecto; `ws_fidelidad.py` tiene
  que seguir dando md5 idéntico con los campos apagados y tramos de voz idénticos byte a byte con
  ellos encendidos (la misma garantía estructural de `forma`).
- Puerta: UTMOS ≥ 0 [IC], juez P(real) IC > 0 en Carlos, ECAPA ≥ −0,005, RTF ≤ +1 %, VmHWM sin
  subir más de 50 MB.
- Coste: 1 día Mac, 1 h VM. Abandono: UTMOS con IC < 0.

### F7 · LoRA es/en del `tts_lm` para WER (g4dn, 1-2 semanas de calendario)

- Prerrequisito: escribir el bucle de entrenamiento (forzado de latentes con el codificador
  comunitario, pérdida v-prediction de la cabeza + EOS), reutilizando `fase3_condiciones.py`.
  Validar el bucle antes de gastar un dólar: en el Mac, 50 pasos sobre 10 clips tienen que bajar la
  pérdida interna y la "comprobación de forzado" de la 3a tiene que salir bien.
- Datos: banco de dobla con consentimiento (16,9 min) + Google Crowdsourced Latin American Spanish
  (si el share-alike se acepta) o Common Voice es filtrado por SNR + 5-10 h de inglés conversacional
  CC0. Apartar el 10 % por hablante y las frases de `fidelidad.py`.
- GPU: g4dn.xlarge bajo demanda. Preparación de latentes en la propia GPU. 3 corridas × ~1,5 h +
  2 h de montaje = ~7 h → **~4 USD** (E). Tope autoimpuesto: 12 USD para toda la fase.
- Puertas, en orden y todas: (1) huella y md5 de la base intactos con el LoRA a cero;
  (2) `fidelidad.py` 18 semillas × 12 frases: WER medio −30 % relativo con IC < 0 y las dos frases
  "malas" por debajo del 10 %; (3) `banco_ab.py` 238 parejas con las voces **refabricadas** con los
  pesos nuevos frente a producción: UTMOS IC inf ≥ −0,02, identidad ±0,005 global, tono ±0,03 st;
  (4) RTF en la VM ≤ 1,02 × la base tras reconvertir los IR; (5) `evaluar_clones.py` en Avril y
  Sebastián: identidad ≥ −0,01 y WER en inglés sin subir.
- Abandono: si (2) no pasa con 3 corridas, se cierra y se documenta en `optimizacion.md`.
- Coste humano: el mayor del plan (bucle + refabricar 61 voces + IR + 3 h de banco por variante).

### F8 · LoRA por voz para personalidad (g4dn, después de F7, solo Carlos)

- Solo si F7 pasa (así el bucle ya existe) y si F1(a) no basta. Misma puerta que la 3b. ~1 USD.
  Producto solo si el onboarding recoge ≥ 10 min por persona.

### F9 · Token de acento con pesos

- Solo si F3 y F4 fallan con puerta. Antes, censo de datos de Common Voice en con acento español
  (horas disponibles, SNR). Sin datos para es-con-acento-inglés: esa celda queda en B4 o no existe.

---

## 5. Riesgos, callejones probables y lo que no merece la pena

**No repetir (medido y cerrado):** barrer cfg, pasos, `neg_cada`, ruido de arranque por identidad;
cabeza fp32 o int4; decodificador int4; LM int8; Karras; WSOLA/PSOLA para ritmo o tono; FiLM sobre la
condición de la difusión; "\n" o trozos cortos para pausar; cola insistente; iGPU; más hilos; B1/B2/C1/C2/C3
del plan de rendimiento; Qwen3-TTS para el inglés (identidad 0,30 frente a 0,45).

**Callejones probables de este plan:**

- B3 (transcripción traducida): la receta del prefijo exige texto literal; se mide por barato, se
  espera que falle.
- Prefijo mixto (B2): la medida de clips dispares (−0,09) apunta a que la identidad cae antes de que
  el acento se mueva.
- Repetir latentes (F1c): el LM puede leer la repetición como un bucle y desestabilizar el EOS.
- Exposure bias en F7/F8: entrenar con latentes reales y generar con los propios es justo lo que
  rompió la 3b. Hay que medir generando, no solo con pérdida.
- Marcadores no verbales (F5): el precedente es la música inventada, que no se controla por texto.
  Lo probable es un 20-40 % de acierto y risas de otra persona. Por eso la vía 3 (montaje de
  no-verbales reales) es el plan por defecto.
- kNN-VC en CPU: RTF sin medir; si sale > 1 en la m8a, el coste de dobla sube más de lo que la
  puerta de coste tolera. Se cronometra en F4 antes de decidir.
- Refabricar las 61 voces oficiales tras F7: sus `z` se recuperan por inversión de la caché
  (`clonado-de-voz.md` §3), pero es trabajo y hay que volver a elegir semillas por voz (la semilla
  buena es de la voz y del motor: cambian con los pesos).

**Lo que no merece la pena intentar con este modelo y este hardware:**

- Que el modelo genere ambiente. Coste por fotograma igual al del habla, sin control, ensucia la
  identidad. Mezcla.
- Acento por pesos para es-con-acento-inglés: no hay datos.
- Generador de prefijo desde huella, codificador de estilo → adaptador universal: hacen falta cientos
  de hablantes con consentimiento.
- GPU en inferencia de producción: los IR de OpenVINO no corren en NVIDIA y el camino torch fp16 es
  otro audio (`ec2-y-coste.md` L82). Solo entrenar.
- Cambiar de modelo por los no-verbales: los que traen etiquetas de risa (Dia 1,6B, Orpheus 3B,
  CSM-1B) son inglés y GPU; F5-TTS y Fish tienen pesos no comerciales; Chatterbox Multilingual (MIT,
  500 M) habla español pero no está medido en CPU ni en identidad cruzada. Si algún día se mide,
  tiene que superar la tabla de `comparativa-motores.md` §7 con las mismas voces y el mismo juez.

---

## 6. Tabla de prioridades

Ganancia (E) salvo que diga (M). Coste en horas de máquina y dólares; el coste humano va aparte y
es el que manda en F7.

| # | Eje | Palanca | Ganancia esperada | Coste | Riesgo | Cuándo |
|---|---|---|---|---|---|---|
| F0 | todos | jueces de acento (PER) y no-verbales (AST); corpus congelado | habilita todo lo demás | 4 h Mac | bajo | esta semana |
| F1a | personalidad | referencia espontánea frente a leída | pausas/min y sílabas/s del clon hacia el real; despeja la salvedad de la fase 1 | 2 h Mac + 2 h VM | bajo | esta semana |
| F1b-d | segundos | curva real 5-30 s; latentes repetidos; autoarranque | conocer el codo real; +0,02 ECAPA con 5-8 s si (c) o (d) pasan | incluido en F1 | medio (c/d) | esta semana |
| F2 | WER | normalizador de texto; mejor-de-2 en dobla | −1 punto de WER medio; −20 % en el WER mediano de dobla | 3 h Mac/VM + 0,05 USD | bajo | esta semana |
| F3 | acento | B1 referencia propia en destino; B2 prefijo mixto; B3 traducido | B1: +0,05 identidad en inglés si hay audio; B2: acento parcial con −0,03 de identidad | 3 h | alto (B2/B3 fallan probablemente) | esta semana |
| F4 | acento + identidad cruzada | conversión de voz kNN-VC sobre fuente con el acento deseado | los cuatro cuadrantes de acento; identidad 0,9× del clon directo | 4-8 h Mac; en dobla ×1,3-1,6 del tiempo de síntesis | medio | esta semana / la que viene |
| F5 | no verbales | marcadores en el texto | risa/duda/respiración en ≥ 70 % de los clips si pasa; si no, cierra la vía | 15 min VM + 1 h Mac | alto | esta semana |
| F6 | naturalidad | respiración real en el respiro; ambiente por mezcla | UTMOS +0,05-0,15; juez P(real) al alza; RTF +≤ 1 % | 1 día Mac + 1 h VM | bajo | la que viene |
| F7 | WER (+ naturalidad) | LoRA es/en del `tts_lm` y la cabeza | WER −30 % relativo en las frases malas y en voces con referencia floja; 0 en RTF | ~7 h g4dn (~4 USD, tope 12) + 3 h banco + refabricar voces; días humanos | medio-alto (bucle no publicado; exposure bias; refabricar 61 voces) | 1-2 semanas |
| F8 | personalidad | LoRA por voz (Carlos) | pasar la puerta de la 3b | ~1 h g4dn (~1 USD) | alto (solo 8,8 min de datos) | tras F7 |
| F9 | acento | token de acento con pesos | acento medio controlable en inglés | 3-6 h g4dn + días de datos | alto; es-con-acento-inglés sin datos | solo si F3 y F4 fallan |
| — | producto | onboarding con 25-30 s limpios y ≥ 10 min de conversación por persona | la palanca más grande medida (46-61 % del techo lo pone la grabación) (M) | 0 | legal: consentimiento grabado, cifrar `.pt` | ya |

Presupuesto GPU total del plan: ≤ 20 USD de los 50 (F7 12, F8 2, Seed-VC de reserva 2, margen 4).
Los otros 30 quedan para dobla y para repetir lo que no pase a la primera.
