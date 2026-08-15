#!/usr/bin/env python
"""Genera las variantes de pausa a partir de UN solo audio base.

El respiro es un post-proceso puro sobre la cadena de fotogramas (el texto se
cose siempre con espacio, asi que el habla no depende de el). Por eso todas las
variantes salen del MISMO base_espacio.wav: misma voz, misma semilla, mismos
fotogramas, y lo unico que cambia es que se mete entre frases. La comparacion
es limpia por construccion, no por buena suerte. Que la simulacion es fiel lo
ata comprobar_servidor.py por md5: 01_actual_espejo tiene que ser bit a bit lo
que hoy emite la sesion con respiro=True.

Dos ejes:
  MATERIAL  que se inserta: el fotograma entero en espejo (lo de hoy), suelo de
            sala de verdad, o silencio con entradas y salidas suaves.
  SITIO     donde se inserta: detras del fotograma entero (lo de hoy, que cae
            DESPUES del ataque de la palabra siguiente) o delante del ataque.

No reproduce nada: escribe WAV y una tabla de medidas.
"""
import pathlib
import wave

import numpy as np

from analizar import FOT, RITMO, UMBRAL, fotogramas, leer, rms

AQUI = pathlib.Path(__file__).parent
PICO = 0.03
PRERROLLO = 240
ALARGA = 2
DISPARO = 2       # RESPIRO_FOTOGRAMAS
TOPE = 8


# ------------------------------------------------------------------ utiles
def escribir(ruta, x):
    # x32768 y no x32767: leer() divide por 32768, asi que asi la ida y vuelta
    # es la identidad y el md5 se puede comparar con el del servidor. Con 32767
    # cada muestra se movia un bit y no coincidia nada.
    d = np.round(np.asarray(x, dtype=np.float64) * 32768)
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RITMO)
        w.writeframes(np.clip(d, -32768, 32767).astype("<i2").tobytes())


PIE = 960           # 40 ms: RESPIRO_PIE


def partir_ataque(v, retroceder=True):
    """(cabeza callada, resto con el ataque dentro).

    La primera muestra que pasa de PICO no es el principio de la palabra: es
    donde el ataque YA esta a -30 dBFS. Medido en el fotograma de la costura,
    la envolvente empieza a subir 20-30 ms antes de eso (de -56 a -48 a -37
    dBFS en bloques de 10 ms). Si se corta ahi, esos 20 ms de rampa se quedan
    al otro lado de la pausa y se oyen sueltos: el principio de la palabra,
    luego el aire, y luego la palabra otra vez.

    Por eso el corte del aire retrocede PIE muestras y no PRERROLLO. Se probo
    tambien un retroceso adaptativo (hacia atras por bloques de 5 ms mientras
    sigan por encima del suelo del propio fotograma) y da EL MISMO corte, asi
    que se queda el numero fijo.

    retroceder=False da el corte historico (solo PRERROLLO), que es el que usa
    el recorte del tope y no hay motivo para cambiar."""
    fuertes = np.nonzero(np.abs(v) >= PICO)[0]
    if len(fuertes) == 0:
        return v, v[:0]
    corte = max(0, int(fuertes[0]) - (PIE if retroceder else PRERROLLO))
    return v[:corte], v[corte:]


def tejer(fuente, largo, solape=480):
    """`largo` muestras de aire hechas con `fuente`, por solapado-suma de
    potencia constante. Sin invertir y sin escalones: en la zona de solape las
    dos copias son ruido incorrelado y las ventanas van en raiz de coseno, asi
    que la envolvente sale plana."""
    fuente = np.asarray(fuente, dtype=np.float32)
    if len(fuente) == 0:
        return np.zeros(largo, dtype=np.float32)
    if len(fuente) >= largo:
        return fuente[:largo].copy()
    solape = max(1, min(solape, len(fuente) // 3))
    paso = len(fuente) - solape
    t = np.linspace(0, np.pi / 2, solape, dtype=np.float32)
    sube, baja = np.sin(t), np.cos(t)
    out = np.zeros(largo + len(fuente), dtype=np.float32)
    pos = 0
    while pos < largo:
        pieza = fuente.copy()
        if pos:
            pieza[:solape] *= sube
            out[pos:pos + solape] *= baja
        out[pos:pos + len(pieza)] += pieza
        pos += paso
    return out[:largo]


def espejar(fuente, largo):
    """`largo` muestras de aire hechas con `fuente` en espejo alternado.

    Es el truco que ya usa el codigo -- la copia invertida EMPIEZA por la
    ultima muestra del original, asi que cada junta es continua por
    construccion --, pero aplicado al SUELO DE SALA y no al fotograma entero.
    Sobre ruido plano el espejo no tiene nada que invertir; sobre un ataque de
    palabra, si, y eso es lo que suena al reves.

    Ventaja sobre tejer(): no hay ni una operacion aritmetica, solo copiar
    muestras del reves. Un segundo implementador (ws_fidelidad.py) lo reproduce
    bit a bit sin depender de como redondee su coma flotante."""
    fuente = np.asarray(fuente, dtype=np.float32)
    if len(fuente) == 0 or largo <= 0:
        return np.zeros(max(largo, 0), dtype=np.float32)
    piezas, n = [], 0
    k = 0
    while n < largo:
        piezas.append(fuente[::-1] if k % 2 == 0 else fuente)
        n += len(fuente)
        k += 1
    return np.concatenate(piezas)[:largo].copy()


def valle(x, ms=25):
    """El aire se apaga del todo y vuelve: silencio de verdad en medio, con
    entrada y salida suaves para que no haya escalon en los bordes."""
    n = min(int(RITMO * ms / 1000), len(x) // 2)
    if n <= 0:
        return np.zeros_like(x)
    v = np.zeros_like(x)
    r = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, n, dtype=np.float32))
    v[:n] = x[:n] * r
    v[-n:] = x[-n:] * r[::-1]
    return v


# ------------------------------------------------- el respiro, parametrizado
def respirar(fs, material="espejo", sitio="despues", alarga=ALARGA,
             jitter=None):
    """Reproduce ColaAudioSesion.put() con la estrategia de pausa que se pida.

    material: "ninguno" | "espejo" (hoy) | "suelo" | "silencio"
    sitio:    "despues" (hoy) | "antes" (el aire va delante del ataque)
    jitter:   fotogramas por pausa, ciclico. None = `alarga` fijo.

    Devuelve (audio, [(inicio, fin)] de cada tramo INSERTADO).
    """
    salida, metidos = [], []
    seguidos, sonado, pausa, n = 0, False, 0, 0

    def mete(trozo, insertado=False):
        nonlocal n
        if len(trozo) == 0:
            return
        if insertado:
            metidos.append((n, n + len(trozo)))
        salida.append(trozo)
        n += len(trozo)

    for k, v in enumerate(fs):
        ultimo = k == len(fs) - 1
        if rms(v) >= UMBRAL:
            seguidos, sonado = 0, True
            mete(v)
            continue
        seguidos += 1
        if seguidos > TOPE:
            # el recorte del tope no cambia: mismo corte de siempre
            _, resto = partir_ataque(v, retroceder=False)
            mete(resto)
            continue
        dispara = (seguidos == DISPARO and sonado and not ultimo
                   and material != "ninguno")
        if not dispara:
            mete(v)
            continue
        cuantos = alarga if jitter is None else jitter[pausa % len(jitter)]
        pausa += 1
        cabeza, resto = partir_ataque(v)
        largo = cuantos * FOT
        if material == "espejo":
            # EXACTAMENTE lo de hoy: el fotograma entero, invertido y derecho
            # alternando. Sin retoques, que esta variante es la referencia del
            # defecto y tiene que coincidir por md5 con el servidor.
            aire = (np.concatenate([v[::-1] if i % 2 == 0 else v
                                    for i in range(cuantos)])
                    if cuantos > 0 else v[:0])
        elif material in ("suelo", "suelo_espejo", "silencio"):
            # suelo de sala DE VERDAD: la cabeza callada de este fotograma, que
            # es el aire que el modelo acaba de dar en este mismo punto. Si el
            # ataque cae tan pronto que no queda cabeza, se usa el fotograma
            # anterior, que en una pausa ya esta callado entero.
            fuente = cabeza if len(cabeza) >= FOT // 4 else fs[k - 1]
            aire = (espejar(fuente, largo) if material == "suelo_espejo"
                    else tejer(fuente, largo))
            if material == "silencio":
                aire = valle(aire)
        else:
            aire = v[:0]
        if sitio == "antes" and len(resto):
            mete(cabeza)
            mete(aire, insertado=True)
            mete(resto)
        else:
            mete(v)
            mete(aire, insertado=True)
    audio = np.concatenate(salida) if salida else np.zeros(0, np.float32)
    return audio, metidos


# ------------------------------------------------------------------ medidas
def envolvente_db(x, sub=240):
    n = len(x) // sub
    if n == 0:
        return np.zeros(0)
    e = np.sqrt((x[:n * sub].reshape(n, sub).astype(np.float64) ** 2).mean(1))
    return 20 * np.log10(e + 1e-12)


def golpes(x, ini, fin, umbral_db=-45):
    """Arranques de energia por encima de `umbral_db` dentro de un tramo que
    deberia ser aire. Cada uno de mas es un ataque de palabra repetido."""
    e = envolvente_db(x[ini:fin])
    alto = e > umbral_db
    if len(alto) == 0:
        return 0
    return int((alto[1:] & ~alto[:-1]).sum() + (1 if alto[0] else 0))


def modulacion(x, ini, fin, hz=7.5, ancho=1.5):
    """Modulacion de amplitud a `hz` en el tramo (mas 2 fotogramas a cada lado).
    Es la periodicidad audible que delata un trozo repetido: golpes de energia
    separados 133 ms son exactamente 7,5 Hz."""
    tr = x[max(0, ini - 2 * FOT):min(len(x), fin + 2 * FOT)]
    e = envolvente_db(tr, sub=120)
    if len(e) < 16:
        return 0.0
    e = e - e.mean()
    esp = np.abs(np.fft.rfft(e * np.hanning(len(e)))) / len(e)
    f = np.fft.rfftfreq(len(e), 120 / RITMO)
    sel = np.abs(f - hz) <= ancho
    return float(esp[sel].max() * 2) if sel.any() else 0.0


def rango_db(x, ini, fin):
    """Cuanto sube y baja la envolvente DENTRO del aire. Un aire de verdad es
    plano; uno con un ataque dentro tiene 25 dB de recorrido."""
    e = envolvente_db(x[ini:fin])
    return float(e.max() - e.min()) if len(e) else 0.0


def salto_borde(x, pos, margen=4):
    return float(np.abs(np.diff(x[max(0, pos - margen):pos + margen])).max())


def eco_db(x, ini, fin, ms=30):
    """dB que el audio JUSTO ANTES del aire esta por encima del propio aire.

    Es la medida del ataque abandonado al otro lado de la pausa: si el aire se
    mete DETRAS del fotograma de la costura, los ultimos 20-30 ms de ese
    fotograma ya son el arranque de la palabra siguiente, y quedan sueltos --
    se oye el principio de la palabra, luego la pausa, y luego la palabra otra
    vez. Con el aire delante del ataque esto tiene que dar ~0 dB."""
    n = int(RITMO * ms / 1000)
    antes, dentro = x[max(0, ini - n):ini], x[ini:fin]
    if len(antes) == 0 or len(dentro) == 0:
        return 0.0
    return float(20 * np.log10((rms(antes) + 1e-12) / (rms(dentro) + 1e-12)))


def rachas(x, umbral=UMBRAL):
    out, r = [], 0
    for f in fotogramas(x):
        if rms(f) < umbral:
            r += 1
        else:
            if r:
                out.append(r)
            r = 0
    if r:
        out.append(r)
    return out


def habla_intacta(x, base):
    """Quitando los fotogramas que son suelo de sala, ¿queda el mismo habla?
    Es la prueba de que el aire se INSERTA y no se sintetiza."""
    def sonoros(y):
        trozos = [f for f in fotogramas(y) if rms(f) >= UMBRAL]
        return np.concatenate(trozos) if trozos else np.zeros(0, np.float32)
    a, b = sonoros(x), sonoros(base)
    if len(a) != len(b):
        return f"NO({len(a)-len(b):+d})"
    return "si" if np.array_equal(a, b) else "NO"


VARIANTES = [
    ("00_sin_aire", dict(material="ninguno"),
     "referencia: no se inserta nada, la pausa es la que da el modelo"),
    ("01_actual_espejo", dict(material="espejo", sitio="despues"),
     "LO DE HOY: el fotograma entero en espejo, detras del ataque"),
    ("02_espejo_antes", dict(material="espejo", sitio="antes"),
     "espejo pero delante del ataque (aisla el SITIO)"),
    ("03_suelo_despues", dict(material="suelo", sitio="despues"),
     "suelo de sala de verdad, en el sitio de hoy (aisla el MATERIAL)"),
    ("04_suelo_antes", dict(material="suelo", sitio="antes"),
     "suelo de sala de verdad, delante del ataque"),
    ("05_silencio_antes", dict(material="silencio", sitio="antes"),
     "silencio de verdad con entrada y salida suaves, delante del ataque"),
    ("06_suelo_antes_variable", dict(material="suelo", sitio="antes",
                                     jitter=[3, 2, 4, 2, 3, 1]),
     "suelo delante, con la pausa de duracion variable"),
    ("07_suelo_antes_largo", dict(material="suelo", sitio="antes", alarga=3),
     "suelo delante, un fotograma mas de aire"),
    ("08_suelo_espejo_antes", dict(material="suelo_espejo", sitio="antes"),
     "suelo de sala en espejo alternado, delante del ataque"),
    ("09_suelo_espejo_variable", dict(material="suelo_espejo", sitio="antes",
                                      jitter=[3, 2, 4, 2, 3, 1]),
     "suelo en espejo delante, con la pausa de duracion variable"),
]


def main():
    base = leer(AQUI / "base_espacio.wav")
    fs = fotogramas(base)
    print(f"base: {len(base)/RITMO:.2f} s, {len(fs)} fotogramas, "
          f"pausas del modelo {rachas(base)} fotogramas")
    cab = (f"{'variante':<24} {'dur s':>6} {'pausas':>13} {'golpes':>7} "
           f"{'rango dB':>9} {'eco dB':>7} {'mod 7,5Hz':>10} {'salto':>7} "
           f"{'habla':>6}")
    print("\n" + cab)
    print("-" * len(cab))
    for nombre, kw, _ in VARIANTES:
        x, metidos = respirar(fs, **kw)
        escribir(AQUI / f"{nombre}.wav", x)
        g = sum(golpes(x, i, f) for i, f in metidos)
        r = max((rango_db(x, i, f) for i, f in metidos), default=0.0)
        m = max((modulacion(x, i, f) for i, f in metidos), default=0.0)
        s = max((max(salto_borde(x, i), salto_borde(x, f)) for i, f in metidos),
                default=0.0)
        ec = max((eco_db(x, i, f) for i, f in metidos), default=0.0)
        print(f"{nombre:<24} {len(x)/RITMO:>6.2f} {str(rachas(x)):>13} "
              f"{g:>7} {r:>9.1f} {ec:>7.1f} {m:>10.2f} {s:>7.4f} "
              f"{habla_intacta(x, base):>6}")

    print("\ngolpes    = arranques por encima de -45 dBFS DENTRO del aire "
          "insertado. Tiene que ser 0: ahi no debe empezar nada.")
    print("rango dB  = recorrido de la envolvente dentro del aire. Un aire de "
          "verdad es plano (<10 dB); 25 dB es un ataque de palabra metido "
          "dentro.")
    print("eco dB    = cuanto suenan los 30 ms de DELANTE del aire por "
          "encima del aire. >10 dB = el arranque de la palabra se quedo al "
          "otro lado de la pausa y se oye dos veces.")
    print("mod 7,5Hz = modulacion de amplitud a 7,5 Hz (un golpe por "
          "fotograma) = 'esto se repite'.")
    print("salto     = mayor salto entre muestras consecutivas en las juntas "
          "del aire (el habla llega a 0,20).")
    print("habla     = quitando los fotogramas de suelo, ¿queda el mismo "
          "habla que en la base?\n")
    for nombre, _, texto in VARIANTES:
        print(f"  {nombre:<24} {texto}")


if __name__ == "__main__":
    main()
