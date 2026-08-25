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

TEXTURA (--textura): tres firmas de lo que suena "robotico" cuando el tono y el
timbre ya coinciden. Con el triplete original / ciclo del codec / clon entero se
puede separar de quien es la culpa. MEDIDO sobre cuatro voces reales:

    metrica                  CODEC   disp.   GENERAR   suelo semilla
    nitidez de ataques        -1,5 %  +-6,5    -3,6 %       1,6 %
    planitud de sibilantes    -8,0 %  +-20     -5,5 %      16,8 %
    modulacion 4-8 Hz         -1,7 %  +-4,2    +1,5 %       3,1 %
    modulacion 8-16 Hz        -4,5 %  +-6,5    +0,3 %       4,6 %
    modulacion 16-32 Hz      -21,5 %  +-13     -3,6 %       2,7 %
    modulacion 32-64 Hz      -22,1 %  +-9,1    +4,8 %       5,5 %

Cada efecto contra el suelo de ruido QUE LE CORRESPONDE, y no vale mezclarlos:
el ciclo del codec es una reconstruccion determinista del MISMO audio, asi que
cualquier diferencia es real y su suelo es cero; el clon dice el mismo texto
pero es otra realizacion, asi que su suelo es la variabilidad entre semillas.

Lo que dicen estos numeros: **el codec NO emborrona los ataques ni el ritmo
silabico**, que era la prediccion fuerte de la hipotesis de los 7,5 latentes por
segundo. Lo que si pierde es una quinta parte de la modulacion RAPIDA (16-64 Hz)
-- el grano, la aspereza --, consistente en direccion en las cuatro voces en la
banda de 32-64 Hz. Y la generacion apenas resta nada mas alla de la loteria de
la semilla. Si ese 22 % es audible o no, estas metricas no lo dicen: eso lo
decide una escucha A/B entre el original y el ciclo.

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


# ==========================================================================
# TEXTURA: lo que suena "robotico" cuando el tono y el timbre ya coinciden
#
# El sospechoso es la tasa de fotogramas del codec acustico: 7,5 latentes por
# segundo, o sea uno cada 133 ms. Para comparar, EnCodec va a 75 Hz, DAC a 86 y
# Mimi a 12,5. Un ataque de oclusiva dura 5-20 ms y una /s/ es estructura fina
# de ruido: todo eso tiene que caber en 64 dimensiones por cada 133 ms.
#
# Si esa es la causa, tiene tres firmas medibles, y las tres se comparan mejor
# CONTRA EL CICLO del codec (audio -> z -> audio, sin modelo de lenguaje) que
# contra el clon entero: lo que ya aparezca en el ciclo es del codec y no tiene
# arreglo sin entrenar; lo que solo aparezca en el clon es de la generacion.
# ==========================================================================

ENV_VENTANA, ENV_SALTO = 512, 64          # envolvente a 24000/64 = 375 Hz
BANDAS_ENV = [(100, 800), (800, 2500), (2500, 8000)]


def envolventes(x, hz=RITMO):
    """Envolvente de energia por banda, muestreada a 375 Hz.

    Salto de 64 muestras y no 256: con 256 la envolvente va a 93,75 Hz y su
    Nyquist (46,9 Hz) corta justo la banda de transitorios que se quiere mirar.
    """
    v = np.hanning(ENV_VENTANA).astype(np.float32)
    n = max(1, (len(x) - ENV_VENTANA) // ENV_SALTO)
    esp = np.empty((n, ENV_VENTANA // 2 + 1), dtype=np.float32)
    for i in range(n):
        esp[i] = np.abs(np.fft.rfft(x[i * ENV_SALTO:i * ENV_SALTO + ENV_VENTANA] * v))
    f = np.fft.rfftfreq(ENV_VENTANA, 1 / hz)
    return {(lo, hi): np.sqrt(((esp[:, (f >= lo) & (f < hi)] ** 2).sum(1)))
            for lo, hi in BANDAS_ENV}, hz / ENV_SALTO


def espectro_modulacion(x, hz=RITMO, banda=(2500, 8000)):
    """Cuanta energia tiene la envolvente en cada rango de modulacion.

    Se normaliza por la energia total de la envolvente, asi que no depende del
    volumen: son proporciones. La banda 20-64 Hz es la de los transitorios; si
    el codec los emborrona, cae ahi.
    """
    envs, hz_env = envolventes(x, hz)
    e = envs[banda]
    if len(e) < 64:
        return {}
    e = e - e.mean()
    E = np.abs(np.fft.rfft(e * np.hanning(len(e)))) ** 2
    f = np.fft.rfftfreq(len(e), 1 / hz_env)
    total = E[(f > 0.5)].sum() or 1.0
    rangos = [(0.5, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64)]
    return {f"{lo:g}-{hi:g}Hz": float(E[(f >= lo) & (f < hi)].sum() / total)
            for lo, hi in rangos}


def nitidez_ataques(x, hz=RITMO, banda=(2500, 8000)):
    """Cuanto sube la envolvente en sus subidas mas bruscas, en dB por segundo.

    Un ataque de consonante es una subida casi vertical. Si el codec la
    redondea, este numero baja. Se toma el percentil 95 de la derivada positiva
    para quedarse con los ataques de verdad y no con el ruido de fondo.
    """
    envs, hz_env = envolventes(x, hz)
    e = envs[banda]
    if len(e) < 16:
        return 0.0
    e = 20 * np.log10(e + 1e-6)
    d = np.diff(e) * hz_env                # dB por segundo
    d = d[d > 0]
    return float(np.percentile(d, 95)) if len(d) else 0.0


def planitud_sibilantes(x, hz=RITMO, lo=4000, hi=9000, percentil=85):
    """Planitud espectral dentro de las fricativas, en 4-9 kHz.

    Una /s/ humana es ruido: su espectro es casi plano y la planitud (media
    geometrica / media aritmetica) se acerca a 1. Si el codec la reconstruye con
    estructura -- picos, armonicos falsos -- la planitud baja y se oye metalica.
    Se miran solo los fotogramas donde esa banda manda, que son las fricativas.
    """
    v = np.hanning(N_FFT).astype(np.float32)
    m = np.array([np.abs(np.fft.rfft(x[i:i + N_FFT] * v)) ** 2
                  for i in range(0, max(1, len(x) - N_FFT), SALTO)])
    if m.size == 0:
        return 0.0
    f = np.fft.rfftfreq(N_FFT, 1 / hz)
    alta = (f >= lo) & (f < hi)
    baja = (f >= 200) & (f < 2000)
    razon = m[:, alta].sum(1) / (m[:, baja].sum(1) + 1e-12)
    sel = razon >= np.percentile(razon, percentil)
    if sel.sum() < 3:
        return 0.0
    p = m[sel][:, alta] + 1e-12
    return float(np.mean(np.exp(np.log(p).mean(1)) / p.mean(1)))


def textura(x, hz=RITMO):
    """Las tres firmas juntas, para UN audio."""
    mod = espectro_modulacion(x, hz)
    return {"ataques_dB_s": nitidez_ataques(x, hz),
            "planitud_sib": planitud_sibilantes(x, hz),
            "mod_4_8": mod.get("4-8Hz", 0.0),
            "mod_8_16": mod.get("8-16Hz", 0.0),
            "mod_16_32": mod.get("16-32Hz", 0.0),
            "mod_32_64": mod.get("32-64Hz", 0.0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original")
    ap.add_argument("clonados", nargs="+")
    ap.add_argument("--bandas", action="store_true", help="desglose por bandas de frecuencia")
    ap.add_argument("--textura", action="store_true",
                    help="espectro de modulacion, nitidez de ataques y planitud de sibilantes")
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
        if args.textura:
            t = textura(xc)
            print(f"      ataques {t['ataques_dB_s']:6.1f} dB/s | sibilantes {t['planitud_sib']:.3f} | "
                  f"modulacion 4-8 {t['mod_4_8']:.3f}  8-16 {t['mod_8_16']:.3f}  "
                  f"16-32 {t['mod_16_32']:.3f}  32-64 {t['mod_32_64']:.3f}")
        if args.bandas:
            for lo, hi, d in bandas(xr, xc, hz):
                signo = "+" if d >= 0 else ""
                print(f"      {lo:5d}-{hi:5d} Hz: {signo}{d:5.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
