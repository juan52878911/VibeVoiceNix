#!/usr/bin/env python
"""Comparar dos audios que dicen LO MISMO, alineandolos de verdad.

    python scripts/espectro.py original.wav clonado.wav

POR QUE ESTE FICHERO EXISTE
El intento anterior de medir melodia (`prosodia.comparar`) fracaso, y esta
documentado por que: remuestreaba los contornos linealmente a 200 puntos. Con
duraciones que difieren un 10-15 % eso desliza las silabas unas sobre otras, y
la correlacion acaba midiendo poco mas que la declinacion de la frase. Medido:
emparejado 0,139 contra cruzado 0,131 en una matriz 5x5. Cero separacion.

Aqui se alinea con DTW sobre el espectro log-mel y TODO lo demas se mide a lo
largo de ese camino. Alinear por espectro y no por f0 es deliberado: el
espectro dice DONDE esta cada silaba (contenido), y una vez emparejadas las
silabas se puede preguntar si suenan igual de agudas. Alinear por f0 seria
circular.

QUE SE MIDE, UNA VEZ ALINEADO

  f0_corr       correlacion de los dos contornos de f0 en semitonos relativos a
                su propia mediana, sobre los tramos donde LOS DOS tienen voz.
                Esto es "¿sube y baja igual?" sin que influya el registro.
  f0_error      diferencia absoluta mediana en semitonos, tramo a tramo. Esto
                es "¿cuanto se desvia?", que la correlacion no dice: dos curvas
                paralelas separadas tres semitonos correlan 1,0.
  rango_ratio   recorrido del clon dividido por el del original. <1 = aplana.
                Es el numero que buscabamos desde el principio.
  mel_dist      distancia media en el espectro log-mel a lo largo del camino.
                Mezcla timbre y pronunciacion; sube con cualquier diferencia.
  ltas_dist     distancia entre los espectros promediados en el tiempo. NO usa
                el alineamiento: es la "huella tonal" global, la que dice si al
                clon le falta brillo o le sobran graves.
  tilt          inclinacion espectral en dB por octava, de cada uno. Un clon
                mas apagado que el original sale con tilt mas negativo.

Solo numpy, como el resto del repo.
"""
import argparse
import sys
import wave

import numpy as np

RITMO = 24000
N_FFT, SALTO = 1024, 256
N_MEL = 40


def leer_wav(ruta):
    with wave.open(str(ruta)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
        return x, w.getframerate()


def _banco_mel(hz, n_fft, n_mel, fmin=50, fmax=None):
    fmax = fmax or hz / 2
    def a_mel(f): return 2595 * np.log10(1 + f / 700)
    def a_hz(m): return 700 * (10 ** (m / 2595) - 1)
    puntos = a_hz(np.linspace(a_mel(fmin), a_mel(fmax), n_mel + 2))
    bins = np.floor((n_fft + 1) * puntos / hz).astype(int)
    banco = np.zeros((n_mel, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mel):
        izq, cen, der = bins[i], bins[i + 1], bins[i + 2]
        if cen > izq:
            banco[i, izq:cen] = np.linspace(0, 1, cen - izq)
        if der > cen:
            banco[i, cen:der] = np.linspace(1, 0, der - cen)
    return banco


def logmel(x, hz=RITMO):
    v = np.hanning(N_FFT).astype(np.float32)
    marcos = np.array([np.abs(np.fft.rfft(x[i:i + N_FFT] * v))
                       for i in range(0, max(1, len(x) - N_FFT), SALTO)])
    if marcos.size == 0:
        return np.zeros((0, N_MEL), dtype=np.float32)
    return np.log10((marcos ** 2) @ _banco_mel(hz, N_FFT, N_MEL).T + 1e-10)


def dtw(a, b):
    """Camino de minimo coste entre dos secuencias de vectores.

    Coste coseno y no euclideo: al comparar dos hablantes distintos la ENERGIA
    global difiere y el euclideo se dejaria llevar por eso en vez de por la
    forma del espectro, que es lo que empareja silabas.
    """
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    coste = 1 - an @ bn.T
    n, m = coste.shape
    ac = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    ac[0, 0] = 0
    for i in range(1, n + 1):
        fila_ant, fila = ac[i - 1], ac[i]
        c = coste[i - 1]
        for j in range(1, m + 1):
            fila[j] = c[j - 1] + min(fila_ant[j], fila[j - 1], fila_ant[j - 1])
    # rastro hacia atras
    camino, i, j = [], n, m
    while i > 0 and j > 0:
        camino.append((i - 1, j - 1))
        pasos = ((ac[i - 1, j], i - 1, j), (ac[i, j - 1], i, j - 1),
                 (ac[i - 1, j - 1], i - 1, j - 1))
        _, i, j = min(pasos)
    camino.reverse()
    return np.array(camino), float(ac[n, m] / max(1, len(camino)))


def contorno_f0(x, hz=RITMO):
    """f0 por marco, en la MISMA rejilla que logmel, con octavas corregidas."""
    from prosodia import corregir_octavas
    minimo, maximo = hz // 400, hz // 60
    f0 = []
    for i in range(0, max(1, len(x) - N_FFT), SALTO):
        t = x[i:i + N_FFT]
        if np.sqrt(np.mean(t ** 2)) < 0.01:
            f0.append(np.nan); continue
        t = t - t.mean()
        c = np.correlate(t, t, mode="full")[N_FFT - 1:]
        if c[0] <= 0:
            f0.append(np.nan); continue
        p = int(np.argmax(c[minimo:maximo])) + minimo
        f0.append(hz / p if c[p] / c[0] > 0.3 else np.nan)
    return corregir_octavas(np.array(f0))


def ltas(x, hz=RITMO):
    """Espectro promediado en el tiempo, en dB y normalizado a su propia media.
    Normalizar quita el volumen y deja la FORMA, que es lo que compara timbre."""
    v = np.hanning(N_FFT).astype(np.float32)
    m = np.array([np.abs(np.fft.rfft(x[i:i + N_FFT] * v))
                  for i in range(0, max(1, len(x) - N_FFT), SALTO)])
    if m.size == 0:
        return np.zeros(N_FFT // 2 + 1)
    e = 20 * np.log10(m.mean(0) + 1e-10)
    return e - e.mean()


def inclinacion(x, hz=RITMO):
    """dB por octava entre 100 Hz y 8 kHz: cuanto se apaga hacia los agudos."""
    e = ltas(x, hz)
    f = np.fft.rfftfreq(N_FFT, 1 / hz)
    m = (f >= 100) & (f <= 8000)
    if m.sum() < 5:
        return 0.0
    return float(np.polyfit(np.log2(f[m]), e[m], 1)[0])


def comparar(x_ref, x_cmp, hz=RITMO):
    a, b = logmel(x_ref, hz), logmel(x_cmp, hz)
    if len(a) < 5 or len(b) < 5:
        return {}
    camino, coste = dtw(a, b)
    f_ref, f_cmp = contorno_f0(x_ref, hz), contorno_f0(x_cmp, hz)

    pares = [(f_ref[i], f_cmp[j]) for i, j in camino
             if i < len(f_ref) and j < len(f_cmp)
             and not np.isnan(f_ref[i]) and not np.isnan(f_cmp[j])]
    r = {"mel_dist": coste, "cobertura": len(pares) / max(1, len(camino))}
    if len(pares) >= 10:
        pr = np.array([p[0] for p in pares]); pc = np.array([p[1] for p in pares])
        # semitonos relativos a la mediana de CADA UNO: quita el registro
        sr = 12 * np.log2(pr / np.median(pr))
        sc = 12 * np.log2(pc / np.median(pc))
        r["f0_corr"] = float(np.corrcoef(sr, sc)[0, 1])
        r["f0_error"] = float(np.median(np.abs(sr - sc)))
        rr = np.percentile(sr, 95) - np.percentile(sr, 5)
        rc = np.percentile(sc, 95) - np.percentile(sc, 5)
        r["rango_ref"], r["rango_cmp"] = float(rr), float(rc)
        r["rango_ratio"] = float(rc / rr) if rr > 0 else float("nan")
    er, ec = ltas(x_ref, hz), ltas(x_cmp, hz)
    f = np.fft.rfftfreq(N_FFT, 1 / hz)
    m = (f >= 100) & (f <= 10000)
    r["ltas_dist"] = float(np.sqrt(np.mean((er[m] - ec[m]) ** 2)))
    r["tilt_ref"], r["tilt_cmp"] = inclinacion(x_ref, hz), inclinacion(x_cmp, hz)
    return r


def bandas(x_ref, x_cmp, hz=RITMO):
    """Diferencia de energia por banda, en dB. Positivo = al clon le SOBRA."""
    er, ec = ltas(x_ref, hz), ltas(x_cmp, hz)
    f = np.fft.rfftfreq(N_FFT, 1 / hz)
    cortes = [(0, 200), (200, 500), (500, 1000), (1000, 2000),
              (2000, 4000), (4000, 8000), (8000, 12000)]
    return [(lo, hi, float(np.mean(ec[(f >= lo) & (f < hi)] - er[(f >= lo) & (f < hi)])))
            for lo, hi in cortes if ((f >= lo) & (f < hi)).sum()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original")
    ap.add_argument("clonados", nargs="+")
    ap.add_argument("--bandas", action="store_true", help="desglose por bandas de frecuencia")
    args = ap.parse_args()
    xr, hz = leer_wav(args.original)
    print(f"{'clon':30} {'f0_corr':>8} {'f0_err':>7} {'rango':>13} {'ratio':>6} "
          f"{'mel':>6} {'ltas':>6} {'tilt r/c':>13}")
    for r in args.clonados:
        xc, _ = leer_wav(r)
        m = comparar(xr, xc, hz)
        if not m:
            print(f"{r[-30:]:30}  (audio demasiado corto)"); continue
        print(f"{r.split('/')[-1][:30]:30} {m.get('f0_corr', float('nan')):8.3f} "
              f"{m.get('f0_error', float('nan')):7.2f} "
              f"{m.get('rango_ref', 0):5.1f}->{m.get('rango_cmp', 0):5.1f}st "
              f"{m.get('rango_ratio', float('nan')):6.2f} {m['mel_dist']:6.3f} "
              f"{m['ltas_dist']:6.2f} {m['tilt_ref']:6.2f}/{m['tilt_cmp']:6.2f}")
        if args.bandas:
            for lo, hi, d in bandas(xr, xc, hz):
                signo = "+" if d >= 0 else ""
                print(f"      {lo:5d}-{hi:5d} Hz: {signo}{d:5.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
