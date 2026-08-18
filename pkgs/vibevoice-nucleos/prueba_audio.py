"""A/B audible del nucleo int8: audio real por la conv transpuesta real.

QUE ES Y QUE NO ES
Esto NO sintetiza voz: no carga VibeVoice ni necesita el checkpoint. Lo que
hace es pasar audio de verdad por el MISMO nucleo (vv_convtr_din) y con la
MISMA geometria que la subida grande del decodificador -- k=16, stride 8, la
que sube de 3 a 24 kHz y que en produccion iba en fp32 -- y dejar oir la
diferencia entre el camino fp32 de hoy y el nucleo int8.

Sirve para contestar "¿se nota la cuantizacion?" sin el modelo delante. Para
juzgar la voz de verdad hace falta scripts/fidelidad.py con los pesos reales.

LOS PESOS DEL BANCO
Cada par (canal entrada, canal salida) lleva el mismo interpolador sinc
repartido con ganancias positivas aleatorias que suman 1 por canal de salida.
Asi la salida fp32 es EXACTAMENTE el audio remuestreado (se puede escuchar y
comparar), pero los pesos son diversos, que es lo que importa para que el
error de cuantizacion se comporte como en una capa aprendida: con pesos
identicos los errores se sumarian en fase y el banco mentiria a la baja.
La profundidad de reduccion (512 canales) es la de una subida real.

DOS MODOS, Y LA DIFERENCIA IMPORTA
Las activaciones se cuantizan por tensor y POR LLAMADA. El decodificador real
se llama una vez por fotograma (133 ms), asi que la escala se readapta 7,5
veces por segundo; cuantizar una locucion entera de una sola vez es otra cosa.
El banco mide las dos y las compara, porque la diferencia entre ambas es
justo el margen que da el streaming:

  llamada unica   una escala para todo el audio  (pesimista: no es produccion)
  por fotogramas  una escala cada 133 ms         (lo que hace el servicio)

En el tramo casi mudo del banco (-46 dB) la primera hunde la senal y la
segunda no: es la demostracion de que el troceado en streaming no solo sirve
para la latencia, tambien protege a la cuantizacion.

SALIDA
  referencia_fp32.wav   el camino de hoy
  nucleo_int8.wav       el camino con el nucleo, por fotogramas (produccion)
  diferencia_x50.wav    lo que anade el int8, amplificado 50 veces para
                        que sea audible; si aqui se oye la voz y no ruido,
                        la cuantizacion esta coloreando la senal

Uso:
    python prueba_audio.py [--salida DIR] [--segundos 4]
"""
import argparse
import os
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RITMO = 24000          # el del modelo
RATIO = 8              # subida0: k=16, stride 8
K = 16
CANALES_ENT = 512      # profundidad de reduccion de una subida real
CANALES_SAL = 32


# ------------------------------------------------------------------ senal

def _vocal(t, f0, formantes, vibrato=0.0):
    """Sintesis fuente-filtro: tren de pulsos glotal por resonadores.

    No es voz de verdad, pero tiene estructura armonica y formantes, que es
    lo que hace que un artefacto de cuantizacion se OIGA en vez de esconderse
    bajo ruido blanco.
    """
    f = f0 * (1.0 + vibrato * np.sin(2 * np.pi * 4.5 * t))
    fase = 2 * np.pi * np.cumsum(f) / RITMO
    # pulso glotal suave: unos pocos armonicos con caida -12 dB/octava
    fuente = sum(np.sin(fase * n) / (n ** 1.6) for n in range(1, 12))
    salida = np.zeros_like(fuente)
    for frecuencia, ancho, ganancia in formantes:
        # resonador de dos polos
        r = np.exp(-np.pi * ancho / RITMO)
        theta = 2 * np.pi * frecuencia / RITMO
        a1, a2 = -2 * r * np.cos(theta), r * r
        y = np.zeros_like(fuente)
        for i in range(2, len(fuente)):
            y[i] = fuente[i] - a1 * y[i - 1] - a2 * y[i - 2]
        salida += ganancia * y
    return salida


def senal_prueba(segundos):
    """Cuatro tramos, cada uno pensado para destapar un fallo distinto."""
    n = int(segundos * RITMO)
    t = np.arange(n) / RITMO
    x = np.zeros(n)
    corte = lambda a, b: slice(int(a * n), int(b * n))          # noqa: E731

    # 1) vocal sostenida: armonicos limpios; un error de cuantizacion aparece
    #    como armonicos que no estaban
    s = corte(0.00, 0.30)
    x[s] = _vocal(t[s], 118.0, [(520, 60, 1.0), (1180, 90, 0.5)])

    # 2) barrido: recorre el espectro util (por debajo de los 1,5 kHz de
    #    Nyquist del ritmo bajo) y delata errores dependientes de frecuencia
    s = corte(0.30, 0.55)
    tt = t[s] - t[s][0]
    f0, f1, dur = 90.0, 1350.0, tt[-1]
    x[s] = 0.6 * np.sin(2 * np.pi * (f0 * tt + (f1 - f0) / (2 * dur) * tt ** 2))

    # 3) casi silencio: el caso duro de la cuantizacion dinamica por tensor.
    #    Un tono a -46 dB; si el escalado de activaciones fuera por bloque
    #    grande, aqui se lo comeria el redondeo
    s = corte(0.55, 0.72)
    x[s] = 0.005 * np.sin(2 * np.pi * 320 * t[s])

    # 4) vocal con vibrato que se apaga: transitorios y decaimiento, donde se
    #    oye cualquier suelo de ruido anadido
    s = corte(0.72, 1.00)
    tt = t[s] - t[s][0]
    x[s] = _vocal(t[s], 145.0, [(700, 70, 1.0), (1220, 110, 0.6)],
                  vibrato=0.03) * np.exp(-2.2 * tt)

    pico = np.abs(x).max()
    return (x / pico * 0.85).astype(np.float32) if pico > 0 else x


def bajar_ritmo(x, ratio):
    """A ritmo/ratio con filtro anti-alias, para luego volver a subir."""
    taps = 8 * ratio + 1
    n = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(n / ratio) * np.hanning(taps)
    h /= h.sum()
    return np.convolve(x, h, mode="same")[::ratio].astype(np.float32)


# ------------------------------------------------------------------ pesos

def interpolador():
    """Sinc enventanado de K taps para subir x RATIO, normalizado por rama.

    conv_transpose1d con stride s reparte cada muestra de entrada en s ramas
    polifase; normalizar cada rama a suma 1 conserva la amplitud y evita que
    la comparacion mida un cambio de ganancia en vez del ruido de cuantizacion.
    """
    n = np.arange(K)
    h = np.sinc((n - (K - 1) / 2.0) / RATIO) * np.hanning(K + 2)[1:-1]
    for rama in range(RATIO):
        idx = np.arange(rama, K, RATIO)
        suma = h[idx].sum()
        if abs(suma) > 1e-12:
            h[idx] = h[idx] / suma
    return h.astype(np.float32)


def capa_subida(semilla=11):
    """ConvTranspose1d(512->32, k16, s8) que remuestrea de verdad."""
    generador = torch.Generator().manual_seed(semilla)
    conv = nn.ConvTranspose1d(CANALES_ENT, CANALES_SAL, K, stride=RATIO)
    h = torch.from_numpy(interpolador())
    # ganancias positivas y diversas que suman 1 por canal de salida: la
    # salida sigue siendo el remuestreo exacto, pero cada fila de pesos tiene
    # su propia escala, como en una capa aprendida
    g = torch.rand(CANALES_ENT, CANALES_SAL, generator=generador) + 0.05
    g = g / g.sum(dim=0, keepdim=True)
    conv.weight.data = g[:, :, None] * h[None, None, :]
    conv.bias.data = torch.zeros(CANALES_SAL)
    return conv.eval()


# ------------------------------------------------------------------- wav

def escribir_wav(ruta, x):
    y = np.clip(x, -1.0, 1.0)
    with wave.open(ruta, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RITMO)
        f.writeframes((y * 32767.0).astype("<i2").tobytes())


def main():
    from nucleos_torch import ConvTrNativa, cargar_nucleos
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", default=".")
    parser.add_argument("--segundos", type=float, default=4.0)
    argumentos = parser.parse_args()

    lib = cargar_nucleos()
    if lib is None:
        raise SystemExit(
            "no hay nucleos nativos utilizables aqui (¿VIBEVOICE_NUCLEOS_SO?, "
            "¿x86_64 con AVX2?, ¿VIBEVOICE_SIN_NUCLEOS=1?)")
    os.makedirs(argumentos.salida, exist_ok=True)

    original = senal_prueba(argumentos.segundos)
    bajo = bajar_ritmo(original, RATIO)
    print(f"senal: {len(original)} muestras a {RITMO} Hz -> "
          f"{len(bajo)} a {RITMO // RATIO} Hz -> subida x{RATIO} por la capa")

    conv = capa_subida()
    nativa = ConvTrNativa(conv, lib)
    # el audio va replicado en los 512 canales: los pesos suman 1 por canal
    # de salida, asi que cada canal de salida es el remuestreo del audio
    x = torch.from_numpy(bajo)[None, None, :].repeat(1, CANALES_ENT, 1).contiguous()

    # Troceado como el decodificador: 133 ms por llamada, con los k-1 ultimos
    # muestreos de ENTRADA como estado, que es exactamente lo que hace
    # SConvTranspose1d en streaming (ver decoder_manual.py, SubidaTr).
    por_fotograma = int(RITMO / RATIO * 3200 / RITMO)          # 400 = 133 ms
    with torch.no_grad():
        fp32 = F.conv_transpose1d(x, conv.weight, conv.bias, stride=RATIO)
        entera = nativa(x)                                     # una sola escala
        trozos, estado = [], torch.zeros(1, CANALES_ENT, K - 1)
        for ini in range(0, x.shape[2], por_fotograma):
            trozo = x[:, :, ini:ini + por_fotograma]
            lleno = torch.cat([estado, trozo], dim=2)
            estado = lleno[:, :, -(K - 1):]
            y = nativa(lleno)[:, :, : -(K - RATIO)]
            trozos.append(y[:, :, -(trozo.shape[2] * RATIO):])
        troceada = torch.cat(trozos, dim=2)

    borde = K                       # fuera los bordes, iguales en todos
    a = fp32[0, 0, borde:-borde].numpy()
    b = troceada[0, 0, borde:borde + len(a)].numpy()
    c = entera[0, 0, borde:borde + len(a)].numpy()

    # el tramo casi mudo aparte: es donde una cuantizacion mal hecha canta
    ini, fin = int(0.57 * len(a)), int(0.70 * len(a))

    def informe(nombre, y):
        ruido = a - y
        snr = 10 * np.log10(float((a ** 2).sum())
                            / float((ruido ** 2).sum() + 1e-30))
        tramo, ruido_tramo = a[ini:fin], ruido[ini:fin]
        snr_mudo = 10 * np.log10(float((tramo ** 2).sum())
                                 / float((ruido_tramo ** 2).sum() + 1e-30))
        print(f"  {nombre:22s} SNR {snr:6.2f} dB   "
              f"correlacion {float(np.corrcoef(a, y)[0, 1]):.8f}   "
              f"tramo a -46 dB {snr_mudo:6.2f} dB")
        return ruido

    print(f"\nfp32 vs nucleo int8 (pico de la senal {np.abs(a).max():.3f}):")
    ruido = informe(f"por fotogramas ({por_fotograma * 1000 // (RITMO // RATIO)} ms)", b)
    informe("llamada unica (4 s)", c)
    print("\n  la fila de arriba es la de produccion; la de abajo esta solo "
          "para\n  ver cuanto protege el troceado en streaming")

    rutas = {
        "referencia_fp32.wav": a,
        "nucleo_int8.wav": b,
        "diferencia_x50.wav": ruido * 50.0,
    }
    for nombre, datos in rutas.items():
        ruta = os.path.join(argumentos.salida, nombre)
        escribir_wav(ruta, datos)
        print(f"  escrito {ruta}")


if __name__ == "__main__":
    main()
