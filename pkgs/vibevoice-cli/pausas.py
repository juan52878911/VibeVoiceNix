"""Pausas por voz («forma»): medirlas en el audio real de una persona y conformarlas al sintetizar.

EL PORQUÉ (docs/plan-personalidad-voz.md, fases 4, 4b y 4c)
El clon de Carlos pausaba menos y más largo que Carlos. Meter pausas por el TEXTO rompe la cobertura
("\\n" dispara el fin de locución y el modelo se come el último tramo; trocear en trozos cortos inventa
palabras). Lo que no rompe nada es no cambiar lo que se genera: cada pausa que el modelo YA hace pasa a
durar lo que duran las pausas de esa persona. Medido con 4 semillas nuevas (4c): la velocidad por clip
se acerca a la real (−0,215 sílabas/s, IC bajo 0), WER −0,07 puntos, UTMOS −0,004, ECAPA +0,007 y cero
catástrofes; el único clip que la puerta marcó fue whisper dejando de oír tres palabras que el audio
conserva byte a byte.

LA GARANTÍA: NUNCA SE TOCA UNA MUESTRA CON VOZ
Una pausa es una racha de ventanas calladas de al menos 150 ms ENTRE dos tramos con voz. Los tramos con
voz, los huecos cortos y el silencio de antes de la primera palabra y de después de la última salen tal
cual. De una pausa se conservan sus primeras muestras reales y sus últimos 50 ms reales (ahí puede ir la
rampa del ataque siguiente); si hay que alargarla, el relleno es la propia racha en espejo, y si hay que
acortarla, se tira su centro. `tramos_identicos()` lo comprueba sobre dos audios.

EL DETECTOR ES CAUSAL, Y EL MISMO PARA MEDIR Y PARA CONFORMAR
Ventanas de 10 ms; una ventana está callada si su RMS queda 35 dB por debajo del RMS máximo visto HASTA
AHÍ (con un piso de 0,0015). Es lo que hacía perfil_vocal.py con el máximo del clip entero, pero sin
necesitar el futuro, así que vale en streaming. La distribución de una voz se mide con `perfil()` sobre
su audio real usando exactamente este detector: si se midiera con otro, los números no casarían.

LATENCIA: la voz sale con como mucho 10 ms de retención (la ventana a medio llenar). En silencio se
retienen hasta 150 ms antes de decidir que es pausa, y durante la pausa se van soltando las muestras
reales dejando 50 ms de reserva. Alargar retrasa el resto de la locución; acortar lo adelanta.

Solo numpy, como estirar.py: la imagen Docker no lleva librosa ni scipy.
"""
import re

import numpy as np

RITMO = 24000
VENTANA = 240                # 10 ms
MIN_VENTANAS = 15            # 150 ms: la pausa de perfil_vocal.py
BORDE = 1200                 # 50 ms reales a cada lado de una pausa conformada
RELATIVO_DB = -35.0
PISO = 0.0015
DUR_MIN, DUR_MAX = 0.15, 1.20
VERSION = "pausas.py v1"
_VOCALES = re.compile(r"[aeiouáéíóúü]+", re.I)


class _Detector:
    def __init__(self):
        self.maximo = 0.0
        self._factor = 10 ** (RELATIVO_DB / 20)

    def callada(self, v):
        rms = float(np.sqrt(np.mean(np.square(v, dtype=np.float64))))
        self.maximo = max(self.maximo, rms)
        return rms < max(PISO, self.maximo * self._factor)


def rachas(x):
    """Pausas interiores (>= 150 ms entre tramos con voz) de un audio entero, en muestras [ini, fin)."""
    x = np.asarray(x, np.float32).reshape(-1)
    det, n = _Detector(), len(x) // VENTANA
    calladas = [det.callada(x[i * VENTANA:(i + 1) * VENTANA]) for i in range(n)]
    voz = [i for i, c in enumerate(calladas) if not c]
    if not voz:
        return []
    out, ini = [], None
    for i in range(voz[0], voz[-1] + 1):
        if calladas[i]:
            ini = i if ini is None else ini
        else:
            if ini is not None and i - ini >= MIN_VENTANAS:
                out.append((ini * VENTANA, i * VENTANA))
            ini = None
    return out


def perfil(clips, textos=None):
    """Perfil de pausas de una persona a partir de su audio REAL (24 kHz, mono, float).

    Devuelve la distribución de duraciones (recortada a 0,15-1,2 s, que es lo que se sortea al
    conformar), pausas por minuto y, si se pasan los textos, sílabas por segundo."""
    dist, dur, n_vocales = [], 0.0, 0
    for i, x in enumerate(clips):
        x = np.asarray(x, np.float32).reshape(-1)
        dist += [(b - a) / RITMO for a, b in rachas(x)]
        dur += len(x) / RITMO
        if textos is not None:
            n_vocales += len(_VOCALES.findall(textos[i]))
    if not dist:
        raise ValueError("no hay ni una pausa de 150 ms en esos clips: hace falta más audio real")
    out = {"dist": [round(float(d), 4) for d in np.clip(dist, DUR_MIN, DUR_MAX)],
           "n": len(dist), "pausas_min": len(dist) / (dur / 60), "mediana_s": float(np.median(dist)),
           "segundos": dur, "detector": VERSION}
    if textos is not None:
        out["silabas_s"] = n_vocales / dur
    return out


class ConformadorPausas:
    """Streaming: `empujar(trozo)` y `cerrar()` devuelven listas de trozos float32 listos para emitir."""

    def __init__(self, dist, semilla=0):
        d = np.clip(np.asarray(dist, np.float64), DUR_MIN, DUR_MAX)
        if d.size == 0:
            raise ValueError("distribución de pausas vacía")
        self.dist = d
        self.rng = np.random.default_rng(semilla)
        self.det = _Detector()
        self.resto = np.zeros(0, np.float32)
        self.hubo_voz = False
        self._vaciar_racha()

    def _vaciar_racha(self):
        self.racha, self.n_racha, self.objetivo, self.emitidas = [], 0, None, 0

    def empujar(self, trozo):
        x = np.asarray(trozo, np.float32).reshape(-1)
        datos = np.concatenate([self.resto, x]) if self.resto.size else x
        n = len(datos) // VENTANA
        self.resto = datos[n * VENTANA:].copy()
        out = []
        for i in range(n):
            v = datos[i * VENTANA:(i + 1) * VENTANA]
            callada = self.det.callada(v)
            if not self.hubo_voz:
                self.hubo_voz = not callada
                out.append(v)
            elif callada:
                self._callada(v, out)
            else:
                self._voz(v, out)
        return _juntar(out)

    def cerrar(self):
        """Fin de la locución: el silencio que quede es el final, y sale tal cual."""
        out = []
        if self.n_racha:
            out.append(np.concatenate(self.racha)[self.emitidas:])
            self._vaciar_racha()
        if self.resto.size:
            out.append(self.resto)
            self.resto = np.zeros(0, np.float32)
        return _juntar(out)

    def _callada(self, v, out):
        self.racha.append(v)
        self.n_racha += len(v)
        if self.objetivo is None and self.n_racha >= MIN_VENTANAS * VENTANA:
            self.objetivo = max(2 * BORDE, int(self.rng.choice(self.dist) * RITMO))
        if self.objetivo is not None:
            # Se sueltan muestras reales guardando siempre 50 ms de reserva para la cola, y nunca
            # más de lo que cabe antes de ella en la duración objetivo.
            limite = min(self.n_racha, self.objetivo) - BORDE
            if limite > self.emitidas:
                todo = np.concatenate(self.racha)
                self.racha = [todo]
                out.append(todo[self.emitidas:limite])
                self.emitidas = limite

    def _voz(self, v, out):
        if self.n_racha:
            todo = np.concatenate(self.racha)
            if self.objetivo is None:
                out.append(todo)                      # hueco corto: tal cual
            else:
                cola = todo[len(todo) - BORDE:]
                falta = self.objetivo - self.emitidas - len(cola)
                if falta > 0:
                    out.append(_relleno(todo, falta))
                out.append(cola)
            self._vaciar_racha()
        out.append(v)


def _relleno(racha, n):
    """n muestras hechas con el centro de la racha en espejo: empieza invertido, así continúa sin salto
    desde la última muestra real emitida."""
    centro = racha[BORDE:len(racha) - BORDE]
    if len(centro) < VENTANA:
        centro = racha
    piezas, total, invertido = [], 0, True
    while total < n:
        piezas.append(centro[::-1] if invertido else centro)
        total += len(centro)
        invertido = not invertido
    return np.concatenate(piezas)[:n].astype(np.float32)


def _juntar(trozos):
    trozos = [t for t in trozos if len(t)]
    if not trozos:
        return []
    return [np.concatenate(trozos).astype(np.float32, copy=False)]


def conformar(x, dist, semilla=0):
    """Versión offline: el audio entero por el mismo conformador que usa el servicio."""
    c = ConformadorPausas(dist, semilla)
    partes = c.empujar(x) + c.cerrar()
    return np.concatenate(partes) if partes else np.zeros(0, np.float32)


def tramos_identicos(base, conformado):
    """La garantía: cada tramo con voz de `base` (lo que queda entre sus pausas, con los huecos cortos)
    aparece idéntico y en el mismo orden dentro de `conformado`. Devuelve (ok, detalle)."""
    base = np.asarray(base, np.float32).reshape(-1)
    conf = np.asarray(conformado, np.float32).reshape(-1)
    previo, pos, tramos = 0, 0, []
    for a, b in rachas(base):
        tramos.append(base[previo:a + BORDE])     # la cabeza real de la pausa sale siempre
        previo = b - BORDE                        # y su cola real también
    tramos.append(base[previo:])
    fb = conf.tobytes()
    for i, t in enumerate(tramos):
        j = fb.find(t.tobytes(), pos)
        while j >= 0 and j % 4:
            j = fb.find(t.tobytes(), j + 1)
        if j < 0:
            return False, f"tramo {i} de {len(tramos)} ({len(t) / RITMO:.2f} s) no aparece en orden"
        pos = j + len(t.tobytes())
    return True, f"{len(tramos)} tramos identicos y en orden"
