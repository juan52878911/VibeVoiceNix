#!/usr/bin/env python
"""Quién habla: huellas de voz y perfiles de locutor.

Convierte unos segundos de voz en un vector de 192 números que captura el
TIMBRE, no el contenido (ECAPA-TDNN de SpeechBrain, entrenado en VoxCeleb),
y compara por coseno contra los perfiles guardados. No es autenticación:
es un diferenciador para separar personas y colgarles información -- nadie
se defiende aquí de una suplantación.

POR QUE ECAPA-TDNN Y NO OTRO
  - Resemblyzer (GE2E, 256d): arrastra librosa -> scipy/soxr/soundfile,
    justo las librerías que la imagen Docker de este repo desinstala a
    propósito. Y su EER en VoxCeleb es ~4 veces peor que el de ECAPA.
  - pyannote.audio: el más pesado, y sus modelos buenos piden token de HF.
  - ECAPA (SpeechBrain): EER 0,80 % en VoxCeleb, ~83 MB de modelo, y
    speechbrain 1.1 solo añadió CINCO paquetes al venv de vibevoice
    (speechbrain, torchaudio, hyperpyyaml, ruamel-yaml x2): torch ya estaba
    porque VibeVoice lo necesita. Sin librosa, sin scipy, sin soundfile.
    Esto corre en el PUENTE (el Mac), no en la imagen Docker: no la engorda.

MEDIDO con las 6 voces españolas de ejemplos-voces/ partidas por la mitad
(mitades distintas del mismo locutor contra todos los demás):

    mismo locutor : coseno mínimo 0,626   (media 0,682)
    distinto      : coseno máximo 0,446   (media 0,162)

El umbral 0,55 parte ese hueco por el medio. Sacar una huella de un trozo de
~3 s cuesta ~24 ms en CPU; cargar el modelo, ~6 s UNA vez (se hace en un hilo
al arrancar el puente para que la primera intervención no lo pague).

EL PERFIL DEL ASISTENTE SE APRENDE DE SU PROPIO AUDIO, NO DEL PROMPT
Es el hallazgo que decide el diseño. La voz que VibeVoice GENERA clonando
sp-Spk1_man da coseno 0,24-0,38 contra la grabación de referencia del prompt:
el clon no conserva la huella del original. Matricular al asistente con el WAV
del prompt NO detectaría su propia voz saliendo por el altavoz. En cambio, el
audio generado es consistente consigo mismo: contra la huella MEDIA de varias
locuciones generadas, cada locución da 0,76-0,94. Así que el puente, cada vez
que habla, guarda unos segundos del PCM que acaba de retransmitir y refresca
con ellos el perfil 'asistente' (aprender_asistente). Verdad de terreno por
construcción: da igual qué voz esté configurada, se aprende la que SUENA.

TROZOS CORTOS: por debajo de ~0,8 s la huella es ruido (medido: un trozo de
0,1 s dio coseno -0,04 contra su propio locutor). Se devuelve "indeterminado"
en vez de adivinar.
"""
import json
import os
import tempfile
import threading
import time
import wave

# Umbral de asignación: por debajo de esto, el locutor es desconocido y se le
# abre perfil nuevo.
#
# 0,55 SALIA DE VOCES SINTETICAS y no vale para voz real por microfono. Las
# generadas por VibeVoice son consistentisimas consigo mismas (0,626 el mismo
# locutor, 0,446 distintos), pero una persona de verdad cambia de postura, de
# distancia y de entonacion. Medido con nueve intervenciones reales del mismo
# hablante: consigo mismo 0,274-0,506, contra el asistente 0,050-0,239. Con
# 0,55 no se reconocia NUNCA y se abria un perfil nuevo cada vez.
#
# 0,25 cae en ese hueco, pero es estrecho (0,035). Con mas muestras de
# matriculacion el margen se ensancha; con pocas, dos personas distintas
# pueden confundirse. Si se anade a alguien mas a la casa, hay que rematricular
# a los dos con mas muestras y recalibrar esto.
UMBRAL_PERFIL = float(os.environ.get("VIBEVOICE_UMBRAL_PERFIL", "0.25"))
# Solo se añade una huella nueva a un perfil existente si el parecido es
# holgado: reforzar con casos dudosos degradaría el perfil con el tiempo.
UMBRAL_REFUERZO = 0.70
# Mientras el asistente habla, el micrófono puede oír una MEZCLA de su voz y
# la de quien interrumpe. Para aceptar una interrupción se exige que gane un
# perfil humano por este margen sobre el parecido con el asistente.
MARGEN_HABLANDO = 0.10
# Huellas que se conservan por perfil. Con la media de varias sobra; más solo
# ocupa y hace la comparación más cara.
MAX_HUELLAS = 8
# Por debajo de esto no hay timbre que medir (ver cabecera).
MINIMO_SEGUNDOS = 0.8

RITMO_HUELLA = 16_000


def _decodificar_wav(datos: bytes):
    """WAV PCM -> tensor float32 mono a 16 kHz. Sin ffmpeg: la página manda
    WAV s16 crudo que ella misma construye, y el PCM del sintetizador ya es
    s16 mono. torchaudio solo se usa para remuestrear."""
    import io

    import numpy as np
    import torch
    with wave.open(io.BytesIO(datos)) as w:
        crudo = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        canales, ritmo = w.getnchannels(), w.getframerate()
    señal = crudo.astype(np.float32) / 32768.0
    if canales > 1:
        señal = señal.reshape(-1, canales).mean(axis=1)
    t = torch.from_numpy(señal)
    if ritmo != RITMO_HUELLA:
        import torchaudio
        t = torchaudio.functional.resample(t, ritmo, RITMO_HUELLA)
    return t


def _pcm_a_tensor(pcm: bytes, ritmo: int):
    """PCM s16 mono crudo (lo que baja de la sesión de voz) -> tensor 16 kHz."""
    import numpy as np
    import torch
    señal = np.frombuffer(pcm[:len(pcm) - len(pcm) % 2],
                          dtype=np.int16).astype(np.float32) / 32768.0
    t = torch.from_numpy(señal)
    if ritmo != RITMO_HUELLA:
        import torchaudio
        t = torchaudio.functional.resample(t, ritmo, RITMO_HUELLA)
    return t


class Oido:
    """Los perfiles y el modelo, con un candado: el puente atiende en hilos."""

    def __init__(self, ruta_perfiles: str):
        self.ruta = ruta_perfiles
        self.candado = threading.Lock()
        self._modelo = None
        self._error = None
        self._datos = self._cargar_perfiles()

    # ---- el modelo ------------------------------------------------------
    def precargar(self):
        """Carga el modelo en un hilo aparte: ~6 s que no debe pagar nadie."""
        threading.Thread(target=self._cargar_modelo, daemon=True).start()

    def _cargar_modelo(self):
        with self.candado:
            if self._modelo is not None or self._error is not None:
                return self._modelo
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            modelo = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.expanduser("~/.cache/asistente-huellas/ecapa"),
                run_opts={"device": "cpu"})
        except Exception as e:      # sin speechbrain el puente sigue andando
            with self.candado:
                self._error = f"{type(e).__name__}: {e}"
            return None
        with self.candado:
            self._modelo = modelo
        return modelo

    def listo(self) -> bool:
        with self.candado:
            return self._modelo is not None

    def estado(self) -> dict:
        with self.candado:
            return {"disponible": self._modelo is not None,
                    "error": self._error,
                    "perfiles": len(self._datos["perfiles"])}

    # ---- perfiles en disco ---------------------------------------------
    def _cargar_perfiles(self):
        try:
            with open(self.ruta) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {"version": 1, "perfiles": []}

    def _guardar(self):
        """Escritura atómica: un corte a mitad no puede dejar el JSON roto."""
        d = os.path.dirname(os.path.abspath(self.ruta)) or "."
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._datos, f, ensure_ascii=False)
            os.replace(tmp, self.ruta)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def lista(self):
        with self.candado:
            return [{"id": p["id"], "nombre": p["nombre"], "tipo": p["tipo"],
                     "muestras": len(p["huellas"])}
                    for p in self._datos["perfiles"]]

    def renombrar(self, id_, nombre):
        with self.candado:
            for p in self._datos["perfiles"]:
                if p["id"] == id_:
                    p["nombre"] = nombre.strip() or p["nombre"]
                    self._guardar()
                    return True
        return False

    def _nuevo(self, nombre, tipo, huella):
        base = ("desconocido" if tipo == "desconocido"
                else nombre.strip().lower().replace(" ", "-") or "perfil")
        usados = {p["id"] for p in self._datos["perfiles"]}
        id_, n = base, 1
        while id_ in usados:
            n += 1
            id_ = f"{base}-{n}"
        perfil = {"id": id_, "nombre": nombre.strip() or id_, "tipo": tipo,
                  "huellas": [huella.tolist()], "creado": round(time.time()),
                  # Los huecos que pide el diseño: aún nadie escribe en ellos,
                  # pero el sitio donde colgarán preferencias y memoria ya
                  # existe y no habrá que migrar el fichero.
                  "preferencias": {}, "memoria": []}
        self._datos["perfiles"].append(perfil)
        self._guardar()
        return perfil

    # ---- huellas --------------------------------------------------------
    def _huella(self, t):
        """Tensor de voz -> vector unitario de 192d, o None si es muy corto."""
        import torch
        if len(t) < MINIMO_SEGUNDOS * RITMO_HUELLA:
            return None
        modelo = self._cargar_modelo()
        if modelo is None:
            return None
        with torch.no_grad():
            e = modelo.encode_batch(t.unsqueeze(0)).squeeze()
        return (e / e.norm()).cpu()

    def _parecidos(self, huella):
        """[(perfil, mejor_coseno)] contra cada perfil. El coseno de un perfil
        es el MÁXIMO sobre sus huellas: una persona con varias tomas (cerca,
        lejos, resfriada) debe casar si se parece a cualquiera de ellas."""
        import torch
        salida = []
        for p in self._datos["perfiles"]:
            m = max(float(huella @ torch.tensor(h)) for h in p["huellas"])
            salida.append((p, m))
        return salida

    def identificar(self, wav: bytes, hablando: bool = False) -> dict:
        """Decide de quién es la voz del WAV, y si hay que descartarla.

        Devuelve siempre un dict con `descartada` y, si hay perfil, quién.
        Reglas, en orden:
          - modelo no disponible o trozo corto -> indeterminado, NO se
            descarta (decidirá la compuerta con el texto)... salvo que el
            asistente esté hablando: ahí lo indeterminado se descarta,
            porque lo más probable es que sea él mismo o una mezcla.
          - gana el perfil 'asistente' -> descartada (es su propia voz).
          - hablando y no gana un humano con margen -> descartada.
          - gana un humano por encima del umbral -> ese es; refuerzo si sobra.
          - no gana nadie -> perfil desconocido-N nuevo, para ponerle nombre.
        """
        t0 = time.perf_counter()
        try:
            t = _decodificar_wav(wav)
        except Exception:
            return {"descartada": bool(hablando), "perfil": None,
                    "motivo": "audio ilegible", "s": 0.0}
        huella = self._huella(t)
        if huella is None:
            motivo = ("modelo de huellas no disponible"
                      if not self.listo() else "trozo demasiado corto")
            return {"descartada": bool(hablando), "perfil": None,
                    "motivo": motivo, "s": round(time.perf_counter() - t0, 3)}
        with self.candado:
            pares = self._parecidos(huella)
            cos_asistente = max((c for p, c in pares
                                 if p["tipo"] == "asistente"), default=-1.0)
            humanos = [(p, c) for p, c in pares if p["tipo"] != "asistente"]
            mejor, cos = max(humanos, key=lambda x: x[1], default=(None, -1.0))

            if cos_asistente >= UMBRAL_PERFIL and cos_asistente >= cos:
                return {"descartada": True, "perfil": "asistente",
                        "nombre": "el propio asistente",
                        "cos": round(cos_asistente, 3), "motivo": "es su propia voz",
                        "s": round(time.perf_counter() - t0, 3)}
            if hablando and (cos < UMBRAL_PERFIL
                             or cos - max(cos_asistente, 0.0) < MARGEN_HABLANDO):
                return {"descartada": True, "perfil": None,
                        "cos": round(cos, 3), "motivo":
                        "sin voz humana clara mientras el asistente habla",
                        "s": round(time.perf_counter() - t0, 3)}
            if mejor is not None and cos >= UMBRAL_PERFIL:
                if cos >= UMBRAL_REFUERZO and len(mejor["huellas"]) < MAX_HUELLAS:
                    mejor["huellas"].append(huella.tolist())
                    self._guardar()
                perfil = mejor
            else:
                perfil = self._nuevo("", "desconocido", huella)
                cos = 1.0
            return {"descartada": False, "perfil": perfil["id"],
                    "nombre": perfil["nombre"], "tipo": perfil["tipo"],
                    "cos": round(cos, 3),
                    "s": round(time.perf_counter() - t0, 3)}

    def matricular(self, nombre: str, wav: bytes) -> dict:
        """Alta (o refuerzo) a mano desde la página: nombre + unos segundos."""
        t = _decodificar_wav(wav)
        huella = self._huella(t)
        if huella is None:
            return {"error": "no hay modelo de huellas o la grabación es "
                             f"demasiado corta (mínimo {MINIMO_SEGUNDOS} s)"}
        with self.candado:
            for p in self._datos["perfiles"]:
                if p["nombre"].strip().lower() == nombre.strip().lower():
                    if len(p["huellas"]) < MAX_HUELLAS:
                        p["huellas"].append(huella.tolist())
                    self._guardar()
                    return {"perfil": p["id"], "muestras": len(p["huellas"])}
            p = self._nuevo(nombre, "persona", huella)
            return {"perfil": p["id"], "muestras": 1}

    def aprender_asistente(self, pcm: bytes, ritmo: int):
        """Refresca el perfil del asistente con el audio que ACABA de decir.

        Se llama al terminar cada respuesta con el PCM que el puente
        retransmitió. Huellas rodantes: entra la nueva, sale la más vieja.
        Así el perfil sigue a la voz que suena aunque cambien la voz, la
        expresividad o la semilla desde la página.
        """
        try:
            t = _pcm_a_tensor(pcm, ritmo)
        except Exception:
            return
        # Con más de ~20 s sobra, y el remuestreo no es gratis.
        t = t[:20 * RITMO_HUELLA]
        huella = self._huella(t)
        if huella is None:
            return
        with self.candado:
            for p in self._datos["perfiles"]:
                if p["tipo"] == "asistente":
                    p["huellas"].append(huella.tolist())
                    p["huellas"] = p["huellas"][-MAX_HUELLAS:]
                    self._guardar()
                    return
            self._datos["perfiles"].insert(0, {
                "id": "asistente", "nombre": "Asistente", "tipo": "asistente",
                "huellas": [huella.tolist()], "creado": round(time.time()),
                "preferencias": {}, "memoria": []})
            self._guardar()
