#!/usr/bin/env python
"""Compara narrar por frases sueltas contra narrarlas en UNA sesion viva.

    python scripts/sesiones_fidelidad.py --voz-url http://127.0.0.1:8092

QUE COMPARA, Y POR QUE ASI
El RMS no vale para esto. Un intento anterior de encadenar frases daba 0 frases
"mudas" por volumen y aun asi repetia el texto de la frase anterior: el audio
sonaba, pero decia otra cosa. Asi que aqui se TRANSCRIBE todo y se compara
palabra a palabra.

Cuatro modos, mismas frases y misma semilla en los cuatro:

  sueltas      una peticion por frase, como hoy en produccion. Es la referencia.
  junta        las 6 frases en UNA peticion. Es el techo: el modelo ve todo el
               texto por delante y nunca dispara EOS a mitad.
  sesion-golpe una sesion viva a la que se le meten las 6 de una vez. Tiene que
               salir IDENTICA a 'junta', bit a bit: es la prueba de que el
               alimentador reparte el texto igual que el tensor fijo.
  sesion       una sesion viva alimentada frase a frase, esperando a que el
               modelo se quede PARADO sin texto antes de meter la siguiente.
               Es el caso real: poner voz a un LLM segun escribe.

En los tres ultimos el audio es una locucion continua, asi que no hay "el WAV
de la frase 3". Para poder dar WER por frase se alinea la transcripcion entera
contra la concatenacion de las 6 referencias (Levenshtein con rastro) y se
corta la hipotesis por donde caen los limites de cada frase. De ahi salen las
tres cosas que importan: WER por frase, frases mudas (no queda ni una palabra
alineada) y frases que repiten texto de la anterior.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fidelidad import FRASES, distancia, normalizar, transcribir  # noqa: E402

RITMO = 24_000


# --------------------------------------------------------------------- HTTP
def _pedir(url, token, cuerpo=None, metodo=None, tiempo=600):
    pet = urllib.request.Request(
        url, method=metodo or ("POST" if cuerpo is not None else "GET"),
        data=json.dumps(cuerpo).encode() if cuerpo is not None else None,
        headers={"content-type": "application/json",
                 **({"authorization": f"Bearer {token}"} if token else {})})
    return urllib.request.urlopen(pet, timeout=tiempo)


def _guardar(pcm, ruta):
    with wave.open(ruta, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RITMO)
        w.writeframes(pcm)
    return len(pcm) / 2 / RITMO


def sintetizar(texto, url, token, voz, cfg, semilla, ruta):
    cuerpo = {"texto": texto, "voz": voz, "cfg_scale": cfg, "semilla": semilla}
    pcm = _pedir(f"{url}/tts/stream", token, cuerpo).read()[44:]
    return pcm, _guardar(pcm, ruta)


# ------------------------------------------------------------------ sesiones
class Sesion:
    """Cliente de una sesion viva: el audio por un hilo, el texto por otro."""

    def __init__(self, url, token, nombre, voz, cfg, semilla):
        self.url, self.token, self.nombre = url, token, nombre
        self.voz, self.cfg, self.semilla = voz, cfg, semilla
        self.pcm = bytearray()
        self._hilo = None

    def texto(self, frase, fin=False):
        cuerpo = {"texto": frase, "voz": self.voz, "cfg_scale": self.cfg,
                  "semilla": self.semilla, "fin": fin}
        return json.load(_pedir(f"{self.url}/tts/sesion/{self.nombre}", self.token, cuerpo))

    def escuchar(self):
        def leer():
            r = _pedir(f"{self.url}/tts/sesion/{self.nombre}/audio", self.token)
            r.read(44)                      # cabecera WAV
            while True:
                trozo = r.read(8192)
                if not trozo:
                    break
                self.pcm += trozo
        self._hilo = threading.Thread(target=leer, daemon=True)
        self._hilo.start()

    def estado(self):
        return json.load(_pedir(f"{self.url}/tts/sesion/{self.nombre}", self.token))

    def esperar_parada(self, limite=120):
        """Hasta que el modelo se queda quieto por falta de texto."""
        t0 = time.time()
        while time.time() - t0 < limite:
            try:
                e = self.estado()
            except urllib.error.HTTPError:
                return False            # la sesion ya termino
            if e.get("esperando"):
                return True
            time.sleep(0.2)
        return False

    def fin(self, ruta):
        _pedir(f"{self.url}/tts/sesion/{self.nombre}/fin", self.token, {})
        self._hilo.join(timeout=300)
        return bytes(self.pcm), _guardar(bytes(self.pcm), ruta)


# ----------------------------------------------------------------- alineado
def alinear(ref, hip):
    """Levenshtein con rastro. Devuelve, para cada posicion de la referencia,
    donde cae en la hipotesis."""
    R, H = len(ref), len(hip)
    d = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        d[i][0] = i
    for j in range(H + 1):
        d[0][j] = j
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hip[j - 1]))
    # rastro hacia atras: corte[i] = indice en la hipotesis donde empieza ref[i]
    corte = [H] * (R + 1)
    i, j = R, H
    corte[R] = H
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hip[j - 1]):
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
        corte[i] = min(corte[i], j)
    return corte


def por_frases(frases, oido):
    """WER de cada frase dentro de una transcripcion continua."""
    trozos = [normalizar(f).split() for f in frases]
    ref = [p for t in trozos for p in t]
    hip = normalizar(oido).split()
    corte = alinear(ref, hip)
    salida, base = [], 0
    for t in trozos:
        ini, fin = corte[base], corte[base + len(t)]
        base += len(t)
        seg = hip[ini:fin]
        salida.append({"ref": t, "hip": seg,
                       "wer": distancia(t, seg) / len(t) if t else 0.0,
                       "muda": len(seg) == 0 or not set(seg) & set(t)})
    return salida


def repite_anterior(res):
    """Frases que sueltan un cacho literal de la anterior (>=2 palabras)."""
    marcadas = []
    for k in range(1, len(res)):
        previa, propia = set(res[k - 1]["ref"]), set(res[k]["ref"])
        ajenas, racha = 0, 0
        for p in res[k]["hip"]:
            racha = racha + 1 if (p in previa and p not in propia) else 0
            ajenas = max(ajenas, racha)
        if ajenas >= 2:
            marcadas.append(k)
    return marcadas


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voz-url", default=os.environ.get("VOZ_STREAM_URL", "http://127.0.0.1:8092"))
    ap.add_argument("--api-url", default=os.environ.get("VOZ_API_URL", "http://127.0.0.1:8080"))
    ap.add_argument("--token", default=os.environ.get("VOZ_TOKEN", ""))
    ap.add_argument("--token-voz", default=os.environ.get("VOZ_STREAM_TOKEN", ""))
    ap.add_argument("--voz", default="sp-Spk3_man")
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--semillas", default="11,7,3,23")
    ap.add_argument("--audios", default="/tmp/sesiones")
    ap.add_argument("--modos", default="sueltas,junta,sesion-golpe,sesion")
    a = ap.parse_args()

    semillas = [int(s) for s in a.semillas.split(",")]
    modos = a.modos.split(",")
    os.makedirs(a.audios, exist_ok=True)
    frases = FRASES
    resumen = {m: [] for m in modos}
    md5 = {}

    for semilla in semillas:
        for modo in modos:
            t0 = time.time()
            ruta = os.path.join(a.audios, f"{modo}_{semilla}.wav")
            if modo == "sueltas":
                res, dur = [], 0.0
                for i, f in enumerate(frases):
                    r = os.path.join(a.audios, f"suelta_{semilla}_{i}.wav")
                    _, d = sintetizar(f, a.voz_url, a.token_voz, a.voz, a.cfg,
                                      semilla, r)
                    dur += d
                    oido = transcribir(r, a.api_url, a.token)
                    ref = normalizar(f).split()
                    hip = normalizar(oido).split()
                    res.append({"ref": ref, "hip": hip,
                                "wer": distancia(ref, hip) / len(ref),
                                "muda": len(hip) == 0 or not set(hip) & set(ref)})
            else:
                if modo == "junta":
                    pcm, dur = sintetizar("\n".join(frases), a.voz_url,
                                          a.token_voz, a.voz, a.cfg, semilla, ruta)
                else:
                    s = Sesion(a.voz_url, a.token_voz, f"{modo}-{semilla}",
                               a.voz, a.cfg, semilla)
                    s.texto(frases[0])
                    s.escuchar()
                    for f in frases[1:]:
                        if modo == "sesion":
                            # Solo cuando el modelo ya no tiene texto por
                            # delante: asi se ejerce de verdad la pausa.
                            s.esperar_parada()
                        s.texto(f)
                    if modo == "sesion":
                        s.esperar_parada()
                    pcm, dur = s.fin(ruta)
                md5[(modo, semilla)] = hashlib.md5(pcm).hexdigest()
                oido = transcribir(ruta, a.api_url, a.token)
                res = por_frases(frases, oido)
            repes = repite_anterior(res)
            resumen[modo].append({"semilla": semilla, "res": res, "dur": dur,
                                  "repes": repes, "seg": time.time() - t0})
            print(f"[{modo} semilla {semilla}] {dur:.1f} s de audio en "
                  f"{time.time()-t0:.0f} s · WER medio "
                  f"{statistics.mean(x['wer'] for x in res):.1%} · "
                  f"mudas {sum(x['muda'] for x in res)} · repes {len(repes)}")
            for x in res:
                if x["wer"] > 0:
                    print(f"    {' '.join(x['ref'])[:38]:40s} -> {' '.join(x['hip'])}")
            sys.stdout.flush()

    print("\n" + "=" * 74)
    print(f"{'modo':14s} {'WER medio':>10s} {'WER peor':>9s} {'mudas':>8s} "
          f"{'repiten':>8s} {'audio s':>8s}")
    for modo in modos:
        wers = [x["wer"] for p in resumen[modo] for x in p["res"]]
        if not wers:
            continue
        mudas = sum(x["muda"] for p in resumen[modo] for x in p["res"])
        repes = sum(len(p["repes"]) for p in resumen[modo])
        dur = statistics.mean(p["dur"] for p in resumen[modo])
        print(f"{modo:14s} {statistics.mean(wers):9.1%} {max(wers):9.1%} "
              f"{mudas:5d}/{len(wers):<3d} {repes:8d} {dur:8.1f}")

    # La prueba fuerte: si alimentar por trozos diera algo distinto de pasar el
    # texto entero de golpe, aqui se veria. No es "parecido", es el mismo md5.
    for modo in modos:
        if modo in ("sueltas", "junta") or ("junta", semillas[0]) not in md5:
            continue
        iguales = sum(md5.get(("junta", s)) == md5.get((modo, s)) for s in semillas)
        print(f"audio identico bit a bit 'junta' vs '{modo}': "
              f"{iguales}/{len(semillas)} semillas")


if __name__ == "__main__":
    main()
