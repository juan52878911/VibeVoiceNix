#!/usr/bin/env python
"""Medir la MELODIA de una voz, no su timbre.

    python scripts/prosodia.py referencia.wav clon.wav

POR QUE HACE FALTA
La huella ECAPA de `scripts/oido.py` mide timbre y es ciega a la entonacion.
Un clon puede dar 0,86 de coseno -mismo locutor sin discusion- y sonar plano.
Eso paso con una voz expresiva: identidad correcta y "no se parece" al oido.

POR QUE NO VALE `tono()` DE sondeo_voz.py PARA ESTO
Su detector busca el pico de autocorrelacion entre 60 y 400 Hz y NO corrige
errores de octava. Cuando confunde 228 Hz con 114, el percentil 5 se hunde y el
recorrido sale inflado. MEDIDO sobre tres grabaciones reales de WhatsApp:

    voz          recorrido sin corregir    corregido
    masculina 1        16,5 st                7,4 st
    femenina           18,0 st               10,6 st
    masculina 2        14,0 st                5,1 st

Casi la mitad de lo que se estaba midiendo era ruido de subarmonico, y con opus
a 17-20 kbps eso pasa constantemente. Cualquier conclusion sacada del recorrido
sin corregir es sospechosa.

TRAMPA QUE YA MORDIO UNA VEZ
El primer intento de arreglo descartaba las ventanas con saltos mayores de 6
semitonos, de forma secuencial. Salia basura: si la primera ventana valida ya
era un error de subarmonico, el filtro conservaba los errores y tiraba las
buenas. Se vio porque la mediana se movia una octava entera (102 -> 369 Hz) y
descartaba el 98 % del material. Un filtro que mueve la mediana no esta
limpiando nada.

Lo correcto es CORREGIR, no descartar: para cada ventana se elige entre f0/2,
f0 y f0*2 el candidato mas cercano a la mediana global. Es robusto al orden y
no pierde material.

QUE SE MIDE, Y QUE DE ELLO ESTA VALIDADO

`descripcion()` -- tono, recorrido, desviacion y movimiento de UNA voz, con las
octavas corregidas. VALIDADO en el sentido que importa: arregla un sesgo
demostrado y deja las medianas donde estaban.

`comparar()` -- distancia de melodia entre dos audios. **NO VALIDADO: NO USAR
PARA CONCLUIR NADA.** Se probo contra los casos cuyo resultado ya se conoce y
NO los ordena. Se deja en el fichero porque el intento y su fracaso valen mas
documentados que borrados.

    prueba 1, distribucional (W1 sobre semitonos re-mediana), textos distintos:

        mitades de una misma grabacion (deberia ser lo MAS parecido)  0,65 / 1,09 / 0,26
        clones contra su referencia    (deberia quedar en medio)      0,37 / 0,53 / 0,30
        locutores distintos            (deberia ser lo MENOS)         0,54 / 0,58 / 1,07

    Las dos mitades de una misma grabacion salen MAS distintas entre si que el
    clon de su propia referencia, y dos locutores distintos salen mas parecidos
    que uno consigo mismo. Normalizar por la mediana quita el registro y lo que
    queda -- la forma de la distribucion -- varia mas con el CONTENIDO que con
    quien habla.

    prueba 2, correlacion de contornos con el texto IGUALADO (5 voces oficiales,
    su oraculo y su clon diciendo la misma frase, matriz 5x5):

        emparejado (clon de X contra oraculo de X)  media 0,139  minimo -0,012
        cruzado    (clon de X contra oraculo de Y)  media 0,131  maximo  0,278

    No separa. La sospecha: con duraciones que difieren un 10-15 %, remuestrear
    linealmente a 200 puntos desliza las silabas unas sobre otras y la
    correlacion mide poco mas que la declinacion de la frase. Un alineamiento
    de verdad (DTW sobre energia, o timestamps de palabra) sigue sin probarse.

Conclusion operativa: para juzgar si un clon "suena a la persona" seguimos
teniendo ECAPA (timbre) y los descriptores de `descripcion()`. La melodia sigue
sin metrica, y eso es un hueco abierto, no un problema resuelto.

Solo numpy: la imagen Docker de este repo desinstala librosa, scipy y soundfile
a proposito, y el detector de autocorrelacion ya esta calibrado en estas voces.
"""
import argparse
import sys
import wave

import numpy as np

RITMO = 24000
VENTANA, SALTO = 1024, 256


def leer_wav(ruta):
    with wave.open(str(ruta)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
        return x, w.getframerate()


def contorno(x, hz=RITMO, umbral_rms=0.01, umbral_corr=0.3, f_min=60, f_max=400):
    """f0 por ventana; NaN donde no hay sonido tonal. Mismo detector que
    sondeo_voz.py, pero devolviendo el contorno entero y no solo agregados.

    f_min/f_max acotan la busqueda. Con la banda entera (60-400 Hz) la
    autocorrelacion cae a veces en el subarmonico y el clip sale una octava
    grave (MEDIDO: -12,4 st en un clip de una voz de 247 Hz). Cuando se
    conoce la voz, acotar a [f0/1,5, f0*1,5] no deja sitio a la octava."""
    minimo, maximo = max(2, int(hz / f_max)), int(hz / f_min)
    f0 = []
    for i in range(0, len(x) - VENTANA, SALTO):
        t = x[i:i + VENTANA]
        if np.sqrt(np.mean(t ** 2)) < umbral_rms:
            f0.append(np.nan); continue
        t = t - t.mean()
        c = np.correlate(t, t, mode="full")[VENTANA - 1:]
        if c[0] <= 0:
            f0.append(np.nan); continue
        p = int(np.argmax(c[minimo:maximo])) + minimo
        f0.append(hz / p if c[p] / c[0] > umbral_corr else np.nan)
    return np.array(f0)


def corregir_octavas(f0, pasadas=2):
    """Lleva cada ventana a la octava mas cercana a la mediana global."""
    y = f0.copy()
    validas = y[~np.isnan(y)]
    if len(validas) < 5:
        return y
    ref = np.median(validas)
    for _ in range(pasadas):          # dos pasadas: la mediana mejora con la 1a
        for i in range(len(y)):
            if np.isnan(y[i]):
                continue
            cand = np.array([y[i] / 2, y[i], y[i] * 2])
            cand = cand[(cand > 50) & (cand < 500)]
            if len(cand):
                y[i] = cand[np.argmin(np.abs(np.log2(cand / ref)))]
        v = y[~np.isnan(y)]
        if len(v) < 5:
            return f0                  # la correccion se comio el material: no tocar
        ref = np.median(v)
    return y


def semitonos(f0):
    """Contorno en semitonos relativos a la mediana. Quita el registro y deja
    la melodia, que es lo que se quiere comparar entre dos voces distintas."""
    v = f0[~np.isnan(f0)]
    if len(v) < 5:
        return np.array([])
    return 12 * np.log2(v / np.median(v))


def _wasserstein(a, b, n=99):
    """W1 entre dos muestras 1D: diferencia media de sus cuantiles."""
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    q = np.linspace(1, 99, n)
    return float(np.mean(np.abs(np.percentile(a, q) - np.percentile(b, q))))


def descripcion(x, hz=RITMO):
    """Las cifras de UNA voz: tono, recorrido y cuanto se mueve."""
    f = corregir_octavas(contorno(x, hz))
    v = f[~np.isnan(f)]
    if len(v) < 5:
        return {"hz": 0.0, "recorrido": 0.0, "desviacion": 0.0, "movimiento": 0.0,
                "tonal": 0.0}
    st = semitonos(f)
    b, a = np.percentile(v, [5, 95])
    # derivada en semitonos por segundo, solo entre ventanas contiguas con valor
    d = []
    for i in range(1, len(f)):
        if not np.isnan(f[i]) and not np.isnan(f[i - 1]):
            d.append(12 * np.log2(f[i] / f[i - 1]) * (hz / SALTO))
    return {"hz": float(np.median(v)),
            "recorrido": float(12 * np.log2(a / b)),
            "desviacion": float(np.std(st)),
            "movimiento": float(np.mean(np.abs(d))) if d else 0.0,
            "tonal": float(len(v) / max(1, len(f)))}


def comparar(x_ref, x_cmp, hz=RITMO, mismo_texto=False):
    """Distancia de melodia entre dos audios. Menor es mas parecido."""
    f_ref = corregir_octavas(contorno(x_ref, hz))
    f_cmp = corregir_octavas(contorno(x_cmp, hz))
    st_ref, st_cmp = semitonos(f_ref), semitonos(f_cmp)

    def deriv(f):
        return np.array([12 * np.log2(f[i] / f[i - 1]) * (hz / SALTO)
                         for i in range(1, len(f))
                         if not np.isnan(f[i]) and not np.isnan(f[i - 1])])

    r = {"w1_tono": _wasserstein(st_ref, st_cmp),
         "w1_movimiento": _wasserstein(deriv(f_ref), deriv(f_cmp))}

    if mismo_texto:
        # Contorno interpolado sobre las ventanas sonoras y remuestreado a 200
        # puntos. Solo tiene sentido si los dos audios dicen LO MISMO.
        def curva(f, n=200):
            idx = np.where(~np.isnan(f))[0]
            if len(idx) < 5:
                return None
            st = 12 * np.log2(f[idx] / np.median(f[idx]))
            return np.interp(np.linspace(idx[0], idx[-1], n), idx, st)
        a, b = curva(f_ref), curva(f_cmp)
        r["corr_contorno"] = (float(np.corrcoef(a, b)[0, 1])
                              if a is not None and b is not None else float("nan"))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("referencia")
    ap.add_argument("comparados", nargs="*")
    ap.add_argument("--mismo-texto", action="store_true",
                    help="anade la correlacion de contornos; solo vale si dicen lo mismo")
    ap.add_argument("--crudo", action="store_true",
                    help="ademas, el recorrido SIN corregir octavas, para ver el sesgo")
    args = ap.parse_args()

    xr, hz = leer_wav(args.referencia)
    dr = descripcion(xr, hz)
    print(f"{'audio':34} {'Hz':>7} {'recorr':>7} {'desv':>6} {'movim':>7}", end="")
    if args.crudo:
        print(f" {'recorr crudo':>13}", end="")
    print(f" {'W1 tono':>8} {'W1 movim':>9}", end="")
    if args.mismo_texto:
        print(f" {'corr cont':>10}", end="")
    print()

    def fila(nombre, d, c=None, crudo=None):
        print(f"{nombre[-34:]:34} {d['hz']:7.1f} {d['recorrido']:7.1f} "
              f"{d['desviacion']:6.2f} {d['movimiento']:7.1f}", end="")
        if args.crudo:
            print(f" {crudo:13.1f}", end="")
        if c is None:
            print(f" {'—':>8} {'—':>9}", end="")
            if args.mismo_texto:
                print(f" {'—':>10}", end="")
        else:
            print(f" {c['w1_tono']:8.2f} {c['w1_movimiento']:9.1f}", end="")
            if args.mismo_texto:
                print(f" {c.get('corr_contorno', float('nan')):10.3f}", end="")
        print()

    def recorrido_crudo(x, hz):
        v = contorno(x, hz); v = v[~np.isnan(v)]
        if len(v) < 5:
            return 0.0
        b, a = np.percentile(v, [5, 95])
        return float(12 * np.log2(a / b))

    fila(args.referencia + "  (REFERENCIA)", dr, None, recorrido_crudo(xr, hz))
    for r in args.comparados:
        x, h = leer_wav(r)
        fila(r, descripcion(x, h), comparar(xr, x, h, args.mismo_texto),
             recorrido_crudo(x, h))
    return 0


if __name__ == "__main__":
    sys.exit(main())
