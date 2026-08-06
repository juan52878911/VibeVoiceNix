"""Cambiar la velocidad del habla sin cambiar el tono.

VibeVoice no tiene ningun parametro de ritmo ni length_scale: el tempo sale de
sus 6 ventanas acusticas por cada 5 tokens de texto y no se expone. Asi que la
velocidad hay que hacerla fuera, sobre el PCM.

POR QUE NO BASTA CON REMUESTREAR
Declarar otro ritmo en la cabecera WAV es gratis, pero mueve el tono junto con
la duracion, y a +-15% el cambio de tono se OYE mucho mas que el de velocidad:
el efecto que se percibe no es "lee mas rapido" sino "suena mas agudo". Por eso
esto estira el tiempo de verdad.

POR QUE WSOLA Y NO EL OLA DE LIBRO
La version ingenua -- leer ventanas a otro ritmo y solaparlas -- no funciona:
caen con fases distintas, se cancelan al sumarse y el tono acaba desplazandose.
Medido con un tono puro de 200 Hz: salia a 160. Con WSOLA sale a 200,2.

Numpy y nada mas, a proposito: la imagen Docker desinstala librosa, scipy,
soundfile y soxr, y no lleva torchaudio ni ffmpeg.
"""
import numpy as np

def estirar(x, factor, ventana=1024, salto=256, busqueda=384):
    """Cambia la duracion SIN tocar el tono (WSOLA).

    La version ingenua -- leer ventanas a otro ritmo y solaparlas -- NO vale:
    las ventanas caen con fases distintas, se cancelan al sumarse y el tono se
    desplaza. Comprobado con un tono puro de 200 Hz, que salia a 160.

    WSOLA lo arregla buscando, alrededor de donde tocaria leer, el punto que
    mejor EMPALMA con lo que venia sonando. Se compara con correlacion
    normalizada para que gane el parecido de forma y no el de volumen.
    """
    if abs(factor - 1.0) < 1e-3 or x.size < ventana + busqueda:
        return x
    w = np.hanning(ventana).astype(np.float32)
    avance = salto * factor
    n = int((x.size - ventana - busqueda) / avance)
    if n < 2:
        return x
    salida = np.zeros((n - 1) * salto + ventana, dtype=np.float32)
    norma = np.zeros_like(salida)
    referencia = x[:ventana]
    for i in range(n):
        ini = max(0, min(int(i * avance), x.size - ventana - 1))
        lo = max(0, ini - busqueda)
        hi = min(x.size - ventana, ini + busqueda)
        mejor, mejor_c = ini, -np.inf
        # de 8 en 8: a 24 kHz son 0,33 ms, mas fino no cambia lo que se oye y
        # multiplica el coste
        for c in range(lo, hi + 1, 8):
            seg = x[c:c + ventana]
            d = np.linalg.norm(seg)
            if d < 1e-6:
                continue
            corr = float(np.dot(seg, referencia)) / d
            if corr > mejor_c:
                mejor_c, mejor = corr, c
        seg = x[mejor:mejor + ventana]
        j = i * salto
        salida[j:j + ventana] += seg * w
        norma[j:j + ventana] += w
        # Lo que seguiria de forma natural: con eso se empalma la proxima.
        sig = x[mejor + salto:mejor + salto + ventana]
        if sig.size == ventana:
            referencia = sig
    return salida / np.maximum(norma, 1e-6)
