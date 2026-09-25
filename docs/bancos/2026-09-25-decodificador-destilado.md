# Decodificador destilado y QC con whisper medium (2026-09-25)

## Decodificador destilado: no pasa

**Resultado:** el alumno se entiende igual que el decodificador actual, pero suena mucho peor y pierde la
voz. Se cierra esta vía tal como estaba planteada; producción sigue con `decoder_mm_int8.xml`.

| Mismo latente, 72 clips (de, en, es, fr; 24 lectores apartados) | Maestro (actual) | Alumno | Diferencia (IC 95 %) |
|---|---|---|---|
| WER normalizado | 0,051 | 0,051 | +0,000 [−0,015; +0,013] |
| Naturalidad (UTMOS) | 3,342 | 1,241 | **−2,101** [−2,255; −1,954] |
| Identidad (ECAPA frente al lector real) | 0,820 | 0,435 | **−0,386** [−0,405; −0,366] |
| Distancia de acento (PER) | 0,155 | 0,286 | +0,131 |

La puerta pedía UTMOS ≥ −0,02, ECAPA ±0,005 y WER ≤ +0,5 puntos. Solo cumple el WER, y el hundimiento
es parecido en los cuatro idiomas (UTMOS entre −1,6 y −3,0).

### Montaje

- **Alumno:** `pkgs/vibevoice-ov/decoder_alumno.py`, la misma arquitectura con la mitad de canales en
  todas las etapas (2048 → 1024, …, 32 → 16) y las mismas profundidades (8-3-3-3-3-3-3). Unos 86 M
  parámetros frente a unos 340 M. Con pesos aleatorios medía 10,8 ms frente a 83 del actual en el mismo
  banco de la VM (`banco_decoder_alumno.py`).
- **Entrenamiento:** `scripts/lora/destilar_decoder.py`.
  - Latentes reales de CML-TTS y LibriTTS-R (los de `datos.py`), pasados a la escala del decodificador.
  - El objetivo es la salida del maestro sobre el mismo latente.
  - Pérdida: STFT multirresolución, mel L1 y 0,1 × L1 de la onda.
  - 12 000 pasos, lote 8 × 16 latentes (2,1 s), lr 3e-4, 130 min en una g4dn.xlarge. Con lote 16 × 24
    se quedó sin memoria.
- **Validación (pérdida total y mel L1):**

  | Paso | Total | mel L1 |
  |---|---|---|
  | 1000 | 3,04 | 1,25 |
  | 2000 | 2,56 | 1,00 |
  | 4000 | 2,19 | 0,81 |
  | 8000 | 1,90 | 0,68 |
  | 10 000 | 1,82 | 0,65 |
  | 12 000 | 1,77 | 0,63 |

  Seguía bajando, pero despacio.
- **Puerta:** `scripts/lora/evaluar_decoder.py`. El audio real apartado se codifica y el mismo latente
  pasa por maestro y alumno. Los dos lotes se juzgan con `juez_lote.py` y se comparan con `comparar.py`.

### Por qué falla

1. **Solo pérdidas espectrales.** Los vocoders que suenan bien (HiFi-GAN, el propio σ-VAE) se entrenan con
   discriminadores adversarios. Sin ellos, la fase y la estructura fina salen como un zumbido: whisper
   sigue entendiendo las palabras (WER igual), pero UTMOS y ECAPA se hunden.
2. **Pocos pasos desde cero.** El alumno partía de pesos aleatorios. 12 000 pasos (2 h) quedan lejos
   de los cientos de miles que necesita un vocoder.

### Si se retoma

Es más barato **no partir de cero**. Casi toda la carga está en la etapa 0: 268 M de los ~340 M
parámetros en 8 bloques de 2048 canales. Opciones:

- **Quitar bloques de la etapa 0** (8 → 4) y dejar el resto idéntico, con pesos del maestro. Después, un
  ajuste corto con la misma pérdida más un discriminador.
- **Podar canales con los pesos del maestro** (SVD o importancia por canal) en vez de iniciar al azar.

Las dos parten de un decodificador que ya suena y se pueden medir con la misma puerta en ~1 h de g4dn.
No se hace ahora: queda a decisión de Juan.

Gasto: 1,35 USD (2,48 h de g4dn.xlarge, incluido el juez del QC).

## QC de dobla con whisper medium: se queda small

El mismo vídeo doblado dos veces con la imagen de producción, cambiando solo el modelo de whisper del QC.
Se juzgó con `scripts/lora/juzgar_historia.py`: whisper large-v3 frente al texto de cada tramo, ECAPA
frente a la voz original en la misma ventana y UTMOS.

| 11 tramos (destino inglés) | QC small (hoy) | QC medium |
|---|---|---|
| WER medio (large-v3) | **0,225** | 0,311 |
| WER mediano | **0,167** | 0,225 |
| Identidad | **0,445** | 0,384 |
| UTMOS | 1,268 | 1,263 |
| WER que midió el propio QC | 0,141 | 0,380 |
| Minutos del job | 10,8 | 9,7 |

Medium no mejora nada de lo que oye el juez independiente y es más severo consigo mismo. Son solo 11
tramos, pero todas las diferencias van en la misma dirección. **Se queda whisper small.**
