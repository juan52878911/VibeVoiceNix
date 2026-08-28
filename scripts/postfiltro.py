#!/usr/bin/env python
"""Post-filtro que deshace lo que el codec acustico le quita a la voz.

    python scripts/postfiltro.py datos    --corpus ~/corpus --salida ~/pares
    python scripts/postfiltro.py entrenar --pares ~/pares --salida postfiltro.pt
    python scripts/postfiltro.py aplicar  --modelo postfiltro.pt entrada.wav salida.wav

QUE ARREGLA, Y POR QUE ESTO Y NO OTRA COSA
Medido con `espectro.py --textura` sobre el triplete original / ciclo del codec /
clon entero, en cuatro voces reales:

    el codec NO toca los ataques (-1,5 %) ni el ritmo silabico (-1,7 %)
    el codec SI pierde ~22 % de la modulacion rapida de 16-64 Hz
    la generacion apenas resta nada mas alla de la loteria de la semilla

O sea: el defecto vive ENTERO en el ciclo encoder->decoder, que no pasa por el
modelo de lenguaje. Confirmado tambien de oido: el ciclo ya suena robotico.

Eso hace que la reparacion sea un problema mucho mas facil que "mejorar el TTS":

  * los pares de entrenamiento son GRATIS e ILIMITADOS -- cualquier grabacion
    de habla real pasada por el ciclo da un par (degradado, limpio) sin
    etiquetar nada
  * el post-filtro va a la SALIDA, no toca el modelo, y no hay que reentrenar
    un TTS para el que ni siquiera existe un trainer que funcione
  * se puede entrenar en un portatil, no hace falta alquilar una GPU

POR QUE APRENDE EL RESIDUO
La red suma una correccion a la entrada en vez de generar el audio desde cero.
Asi el punto de partida es la identidad -- una red sin entrenar no estropea
nada -- y toda su capacidad se dedica a lo que falta, que es poca cosa: una
quinta parte de una banda de modulacion.

QUE FUNCIONA, Y QUE HIZO FALTA PARA QUE FUNCIONARA
Seis intentos. Los dos primeros fracasaron y de ahi salieron las dos piezas que
hacen que el tercero en adelante funcione. Todo medido sobre las CUATRO VOCES
REALES, que el modelo nunca vio: el corpus es de otros hablantes.

  intento                       dist. al original   mejora    mel    8-12 kHz
  1 solo multi-STFT                     0,255        +0,3 %  -130 %  +8,03 dB
  2 + perdida de modulacion             0,165       +35,4 %   -95 %  +6,64 dB
  3 + residuo acotado a 0,5             0,188       +26,7 %    -2 %  -1,80 dB
  4 residuo + mod x3, 196 min           0,124       +51,6 %    -2 %  -1,48 dB
  5 residuo + mod x1, 196 min           0,115       +54,9 %    -3 %  -1,50 dB

(la distancia del ciclo sin filtrar al original es 0,256)

LA PERDIDA DE MODULACION, porque una multi-STFT no ve el grano: compara
magnitudes por ventana y la microestructura vive en lo que ese promedio borra.
El intento 1 bajo su perdida un 6 % sin tocar el problema -- recupero un +7 % y
un -2 % de las bandas de 16-64 Hz -- porque encontro un atajo mas barato.

EL LIMITE DEL RESIDUO, porque ese atajo era subir 11,8 dB por encima de 8 kHz,
donde el original casi no tiene energia. Penalizarlo no bastaba; hay que
hacerlo imposible. Con el recorte por celda tiempo-frecuencia el exceso pasa de
+8 dB a -1,5 dB y la distancia mel de -130 % a -3 %.

Y MAS DATOS: de 90 a 196 minutos de corpus, la mejora sube de +26,7 % a +54,9 %.
La curva no estaba agotada; se paro por disco, no por rendimientos decrecientes.

TRANSFIERE AL TTS DE VERDAD, que era el riesgo de abajo. Sobre audio generado,
con la z viniendo de la difusion y no del encoder:

  voz        dist. TTS   + filtro    mejora    ECAPA     ECAPA + filtro
  juan          0,127      0,211    -66,2 %   0,8796        0,8848
  isis          0,332      0,222    +32,9 %   0,9127        0,9104
  santiago      0,336      0,116    +65,4 %   0,9453        0,9390
  andres        0,363      0,227    +37,5 %   0,9083        0,9127
  MEDIA         0,289      0,194    +32,9 %   0,9115        0,9117

La identidad no se mueve y el WER se queda en 0,000 en las diez frases de
prueba. Juan es la excepcion: su TTS ya salia cerca (0,127 frente a 0,33-0,36
de los otros) y el filtro se pasa de frenada. Su referencia es la mas limitada
en banda de las cuatro, y probablemente por eso.

DONDE VIVE EL MODELO
Pesa 33 MB, o sea que NO va al repositorio, igual que las voces. Vive en
~/.cache/vibevoice-nix/postfiltro.pt y se usa con `decir.py --postfiltro`.
Cuesta RTF 0,014 en CPU con 6 hilos: un 1,5 % del presupuesto del motor.

OJO CON LA LICENCIA: entrenado con OpenSLR 72 (espanol colombiano), que es
CC BY-SA 4.0. Para uso propio da igual; si algun dia se distribuyen los pesos,
el share-alike viaja con ellos.

LA VIA ADVERSARIAL: PROBADA Y DESCARTADA POR PRESUPUESTO, NO POR LA IDEA
Se anadio un discriminador multi-periodo de HiFi-GAN (1,06 M) y se ajusto v5
con el durante 8 epocas. EMPEORA, y de forma monotona:

    v5 (sin discriminador)   +54,9 %
    adv, epoca 2             +51,2 %
    adv, epoca 8             +41,9 %

La causa esta medida, no supuesta. El discriminador se quedo clavado en 0,497
desde la primera epoca, que en LSGAN es exactamente el equilibrio de azar. Con
el generador congelado y entrenando SOLO el discriminador:

    tarea                        separacion tras 150 pasos
    real vs filtrado por v5              +0,000
    real vs ciclo crudo                  +0,000
    real vs real + ruido 0,02            +0,001
    real vs ruido puro                   +0,180

Ni con normalizacion de pesos ni con 5x el learning rate. Y alargando a 1.500
pasos sobre la tarea facil (real vs ciclo crudo) se ve lo que pasa de verdad:

    paso   100   separacion +0,0001
    paso   300              +0,0004
    paso   600              +0,0010
    paso  1000              +0,0019
    paso  1500              +0,0047

APRENDE, pero a un ritmo que necesitaria del orden de 100.000 pasos para dar
una senal util -- que es justamente lo que entrena HiFi-GAN. El ajuste fino de
aqui hizo 2.792. A ese ritmo la separacion era ~0,008: gradiente de ruido, que
arrastro al generador fuera del optimo de reconstruccion sin darle nada a
cambio. De ahi que empeore cuanto mas entrena.

Conclusion: la via adversarial no esta descartada por equivocada -- es la
tecnica estandar para textura -- sino por PRESUPUESTO. Pide dias de GPU, no
horas de portatil. Si algun dia hay una GPU decente, es lo primero que hay que
retomar, y el arnes (`postfiltro.py adversarial`) ya esta escrito y probado.

EL RIESGO QUE HAY QUE VIGILAR
Los pares se hacen con z del ENCODER, pero en produccion la z viene muestreada
por la cabeza de difusion. Son distribuciones parecidas pero no iguales, asi que
un post-filtro que funcione de maravilla sobre el ciclo puede no transferir a
la salida del TTS. Por eso `aplicar` mide siempre antes y despues con
`espectro.textura`, y hay que comprobarlo sobre audio GENERADO, no solo sobre
ciclos.
"""
import argparse
import math
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RITMO = 24000
TROZO = 24000 * 2          # 2 s por muestra de entrenamiento


# ---------------------------------------------------------------- audio --
def leer_wav(ruta):
    with wave.open(str(ruta)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
        return x, w.getframerate()


def escribir_wav(ruta, x):
    with wave.open(str(ruta), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RITMO)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


# ---------------------------------------------------------------- la red --
class Bloque(nn.Module):
    """Conv dilatada con puerta. La puerta deja que la red decida DONDE
    corregir: en las zonas que el codec reconstruye bien, aprende a no tocar."""

    def __init__(self, canales, dilatacion):
        super().__init__()
        self.conv = nn.Conv1d(canales, canales * 2, 5, padding=2 * dilatacion,
                              dilation=dilatacion)
        self.salida = nn.Conv1d(canales, canales, 1)

    def forward(self, x):
        a, b = self.conv(x).chunk(2, dim=1)
        return x + self.salida(torch.tanh(a) * torch.sigmoid(b))


class PostFiltro(nn.Module):
    """U-Net 1D pequena sobre la forma de onda, con salida residual.

    Cuatro niveles a stride 4 dan un campo receptivo de ~85 ms en el cuello,
    suficiente para la modulacion de 16-64 Hz que hay que reconstruir (un ciclo
    de 16 Hz son 62 ms) sin irse a un modelo que no quepa en la CPU del homelab.
    """

    def __init__(self, base=24, niveles=4, bloques=4, limite=0.0, n_fft=512, salto=64):
        super().__init__()
        # LIMITE DEL RESIDUO POR BANDA
        # El intento 2 recuperaba la modulacion pero seguia metiendo +10,6 dB
        # por encima de 8 kHz: subir energia donde el original casi no tiene es
        # la forma mas barata de bajar cualquier perdida espectral, y ni la
        # perdida de modulacion lo impedia.
        #
        # Esto no lo penaliza: lo hace IMPOSIBLE. En cada celda de tiempo y
        # frecuencia el residuo se recorta a `limite` veces la magnitud de la
        # ENTRADA en esa misma celda. Donde la entrada no tiene energia, el
        # residuo tampoco puede tenerla, por mucho que le convenga.
        #
        # Es una proyeccion, no un termino de coste, asi que la garantia se
        # cumple tambien fuera de entrenamiento.
        self.limite, self.n_fft, self.salto = limite, n_fft, salto
        self.entrada = nn.Conv1d(1, base, 7, padding=3)
        cs = [base * (2 ** i) for i in range(niveles + 1)]
        self.baja = nn.ModuleList(
            [nn.Conv1d(cs[i], cs[i + 1], 8, stride=4, padding=2) for i in range(niveles)])
        self.cuello = nn.Sequential(
            *[Bloque(cs[-1], 2 ** i) for i in range(bloques)])
        self.sube = nn.ModuleList(
            [nn.ConvTranspose1d(cs[i + 1], cs[i], 8, stride=4, padding=2) for i in range(niveles)][::-1])
        self.junta = nn.ModuleList(
            [nn.Conv1d(cs[i] * 2, cs[i], 3, padding=1) for i in range(niveles)][::-1])
        self.salida = nn.Conv1d(base, 1, 7, padding=3)
        nn.init.zeros_(self.salida.weight)      # arranca en la identidad
        nn.init.zeros_(self.salida.bias)
        self.act = nn.LeakyReLU(0.2)

    def _recortar(self, r, x):
        """Recorta el residuo a `limite` veces la entrada, celda a celda."""
        v = torch.hann_window(self.n_fft, device=x.device)
        n = x.shape[-1]
        R = torch.stft(r.squeeze(1), self.n_fft, self.salto, window=v, return_complex=True)
        X = torch.stft(x.squeeze(1), self.n_fft, self.salto, window=v, return_complex=True)
        tope = self.limite * X.abs()
        g = torch.clamp(tope / (R.abs() + 1e-8), max=1.0)
        return torch.istft(R * g, self.n_fft, self.salto, window=v, length=n).unsqueeze(1)

    def forward(self, x):                        # (B, 1, T)
        h = self.act(self.entrada(x))
        saltos = []
        for c in self.baja:
            saltos.append(h)
            h = self.act(c(h))
        h = self.cuello(h)
        for c, j, s in zip(self.sube, self.junta, reversed(saltos)):
            h = self.act(c(h))
            if h.shape[-1] != s.shape[-1]:
                h = h[..., :s.shape[-1]] if h.shape[-1] > s.shape[-1] else \
                    nn.functional.pad(h, (0, s.shape[-1] - h.shape[-1]))
            h = self.act(j(torch.cat([h, s], 1)))
        r = self.salida(h)                       # RESIDUO
        if self.limite > 0:
            r = self._recortar(r, x)
        return x + r


# ---------------------------------------------------------------- perdida --
class PerdidaMultiSTFT(nn.Module):
    """STFT a tres resoluciones. La ventana corta (256 con salto 64) es la que
    importa aqui: su envolvente va a 375 Hz y por tanto SI resuelve la banda de
    16-64 Hz que el codec se come. Con una sola ventana larga esa banda queda
    promediada y la perdida no la ve."""

    RESOLUCIONES = [(256, 64), (512, 128), (1024, 256)]

    def __init__(self, disp):
        super().__init__()
        self.ventanas = [torch.hann_window(n, device=disp) for n, _ in self.RESOLUCIONES]

    def forward(self, y, obj):
        total = 0.0
        for (n, salto), v in zip(self.RESOLUCIONES, self.ventanas):
            Y = torch.stft(y.squeeze(1), n, salto, window=v, return_complex=True).abs()
            O = torch.stft(obj.squeeze(1), n, salto, window=v, return_complex=True).abs()
            # magnitud relativa + log: la primera cuida los picos, la segunda
            # los valles, que es donde vive el grano
            total = total + (Y - O).norm() / (O.norm() + 1e-7)
            total = total + nn.functional.l1_loss(torch.log(Y + 1e-5), torch.log(O + 1e-5))
        return total / len(self.RESOLUCIONES)


class PerdidaModulacion(nn.Module):
    """La banda que el codec se come, DENTRO de la funcion de coste.

    El primer intento fallo porque se le pedia a la red recuperar modulacion de
    16-64 Hz con una perdida que no la mide: la multi-STFT compara magnitudes
    por ventana y el grano vive justo en lo que ese promedio borra. La red hizo
    lo racional -- buscar el atajo mas barato para bajarla, que era ecualizar.

    Aqui se mide lo mismo que `espectro.textura`, pero derivable:

      1. STFT con salto de 64 -> envolvente a 375 Hz, que SI resuelve 16-64 Hz
      2. energia por banda acustica -> tres envolventes
      3. FFT de cada envolvente -> espectro de modulacion
      4. energia en cada banda de modulacion, en log, y L1 contra el objetivo

    Se comparan LOGARITMOS de energia y no energias crudas porque el hueco a
    recuperar es una quinta parte de una cantidad ya pequena: en lineal esa
    diferencia no mueve el gradiente frente a las bandas gordas.
    """

    VENTANA, SALTO = 512, 64
    BANDAS_HZ = [(100, 800), (800, 2500), (2500, 8000)]
    # el peso va donde esta el problema; las bajas entran con poco para que la
    # red no destroce el ritmo silabico persiguiendo el grano
    BANDAS_MOD = [(4, 8, 0.25), (8, 16, 0.5), (16, 32, 1.0), (32, 64, 1.0)]

    def __init__(self, disp, hz=RITMO):
        super().__init__()
        self.hz = hz
        self.ventana = torch.hann_window(self.VENTANA, device=disp)
        f = torch.fft.rfftfreq(self.VENTANA, 1 / hz).to(disp)
        self.mascaras = [((f >= lo) & (f < hi)).float() for lo, hi in self.BANDAS_HZ]
        self.hz_env = hz / self.SALTO

    def _energias(self, x):
        S = torch.stft(x.squeeze(1), self.VENTANA, self.SALTO, window=self.ventana,
                       return_complex=True).abs() ** 2          # (B, F, T)
        fuera = []
        for m in self.mascaras:
            env = torch.sqrt((S * m[None, :, None]).sum(1) + 1e-12)   # (B, T)
            env = env - env.mean(dim=1, keepdim=True)
            M = torch.fft.rfft(env * torch.hann_window(env.shape[-1], device=env.device)).abs() ** 2
            fm = torch.fft.rfftfreq(env.shape[-1], 1 / self.hz_env).to(env.device)
            total = M[:, fm > 0.5].sum(1, keepdim=True) + 1e-12
            fuera.append(torch.stack(
                [(M[:, (fm >= lo) & (fm < hi)].sum(1) / total.squeeze(1))
                 for lo, hi, _ in self.BANDAS_MOD], dim=1))          # (B, mod)
        return torch.stack(fuera, dim=1)                             # (B, banda, mod)

    def forward(self, y, obj):
        a, b = self._energias(y), self._energias(obj)
        w = torch.tensor([p for _, _, p in self.BANDAS_MOD], device=a.device)
        d = (torch.log(a + 1e-8) - torch.log(b + 1e-8)).abs()
        return (d * w[None, None, :]).sum() / (d.numel() * w.mean())


# ==========================================================================
# DISCRIMINADOR
#
# La tercera pieza. Las dos perdidas de reconstruccion -- multi-STFT y
# modulacion -- comparan promedios, y el grano de una voz es justo lo que un
# promedio borra: no hay una respuesta "correcta" que copiar, hay una
# TEXTURA que tiene que ser plausible. Eso es lo que un discriminador sabe
# exigir y una L1 espectral no.
#
# Se usa el discriminador multi-periodo de HiFi-GAN, no uno espectral: la onda
# se dobla en 2D con periodos primos y cada rama ve la estructura periodica a
# esa escala. Los periodos pequenos (2, 3, 5 muestras a 24 kHz) caen justo en
# la microestructura de ciclo a ciclo, que es donde vive el jitter que el
# codec plancha.
# ==========================================================================

class RamaPeriodo(nn.Module):
    """Una rama del discriminador: dobla la onda cada `periodo` muestras."""

    def __init__(self, periodo, canales=(16, 64, 128, 256)):
        super().__init__()
        self.periodo = periodo
        capas, ant = [], 1
        for c in canales:
            capas.append(nn.Conv2d(ant, c, (5, 1), (3, 1), padding=(2, 0)))
            ant = c
        self.convs = nn.ModuleList(capas)
        self.final = nn.Conv2d(ant, 1, (3, 1), padding=(1, 0))
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):                       # (B, 1, T)
        b, c, t = x.shape
        if t % self.periodo:
            x = nn.functional.pad(x, (0, self.periodo - t % self.periodo), "reflect")
            t = x.shape[-1]
        x = x.view(b, c, t // self.periodo, self.periodo)
        rasgos = []
        for conv in self.convs:
            x = self.act(conv(x))
            rasgos.append(x)                    # para la perdida de rasgos
        return self.final(x), rasgos


class Discriminador(nn.Module):
    """Varias ramas con periodos primos, para que no compartan alineamiento."""

    def __init__(self, periodos=(2, 3, 5, 7, 11)):
        super().__init__()
        self.ramas = nn.ModuleList([RamaPeriodo(p) for p in periodos])

    def forward(self, x):
        salidas, rasgos = [], []
        for r in self.ramas:
            s, f = r(x)
            salidas.append(s); rasgos.append(f)
        return salidas, rasgos


def perdida_rasgos(reales, falsos):
    """Distancia entre las activaciones internas del discriminador.

    Es el ancla que evita que el generador persiga solo al discriminador: le
    pide que las representaciones intermedias coincidan, no solo el veredicto.
    Sin esto un GAN de audio se va a ruido plausible pero equivocado.
    """
    total = 0.0
    n = 0
    for fr, ff in zip(reales, falsos):
        for a, b in zip(fr, ff):
            total = total + nn.functional.l1_loss(b, a.detach())
            n += 1
    return total / max(1, n)


# ------------------------------------------------------------------ datos --
def subcomando_datos(args):
    """Corpus de habla real -> pares (ciclo del codec, original) en .npy."""
    from banco_duracion import cargar_modelo
    disp = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"cargando el tokenizador acustico en {disp}...", flush=True)
    _, modelo = cargar_modelo(args.modelo, args.cache, disp, 10, con_encoder=True)
    tok = modelo.model.acoustic_tokenizer

    # Se excluyen los `._*`: macOS deja uno por fichero al escribir en volumenes
    # que no soportan atributos extendidos, y son metadatos AppleDouble, no
    # audio. Sin este filtro la lista se DUPLICA -- 4.903 wav reales y 4.903 de
    # basura -- y cada uno gasta un ffmpeg que falla. Medido: la generacion baja
    # de 7x a 3x tiempo real. No corrompe nada, porque el try/except los
    # descarta, pero tarda el doble.
    fuentes = sorted(p for ext in ("*.wav", "*.flac", "*.mp3", "*.opus")
                     for p in Path(args.corpus).rglob(ext)
                     if not p.name.startswith("._"))
    if args.mezclar:
        # los ficheros vienen ordenados y eso pone todas las voces femeninas
        # primero: con --max-min se entrenaria casi solo con ellas
        import random
        random.Random(7).shuffle(fuentes)
    if args.limite:
        fuentes = fuentes[:args.limite]
    if not fuentes:
        raise SystemExit(f"no hay audio en {args.corpus}")
    salida = Path(args.salida); salida.mkdir(parents=True, exist_ok=True)
    print(f"{len(fuentes)} ficheros")

    sucio, limpio, segundos, t0 = [], [], 0.0, time.perf_counter()
    for i, f in enumerate(fuentes, 1):
        try:
            x = cargar_a_24k(f, args.ffmpeg)
        except Exception as e:
            print(f"  [salto] {f.name}: {e}"); continue
        if len(x) < TROZO // 2:
            continue
        # recorte de silencio por los extremos: entrenar sobre silencio no
        # ensena nada y el corpus viene con margen generoso
        x = recortar(x)
        if len(x) < TROZO // 2:
            continue
        with torch.no_grad():
            a = torch.from_numpy(x)[None, None].to(disp, torch.float32)
            z = tok.encode(a).mean                    # la MEDIA, sin ruido:
            y = tok.decode(z).squeeze().cpu().numpy() # el par tiene que ser limpio
        n = min(len(x), len(y))
        for k in range(0, n - TROZO + 1, TROZO):
            sucio.append(y[k:k + TROZO]); limpio.append(x[k:k + TROZO])
        segundos += n / RITMO
        if i % 25 == 0 or i == len(fuentes):
            vel = segundos / (time.perf_counter() - t0)
            print(f"  {i}/{len(fuentes)}  {segundos/60:.1f} min de audio  "
                  f"{len(sucio)} trozos  ({vel:.0f}x tiempo real)", flush=True)
        if args.max_min and segundos / 60 >= args.max_min:
            print("  alcanzado --max-min"); break

    np.save(salida / "sucio.npy", np.stack(sucio).astype(np.float32))
    np.save(salida / "limpio.npy", np.stack(limpio).astype(np.float32))
    print(f"\n{len(sucio)} pares de {TROZO/RITMO:.0f} s "
          f"({len(sucio)*TROZO/RITMO/60:.1f} min) en {salida}")


def cargar_a_24k(ruta, ffmpeg="ffmpeg"):
    import subprocess, tempfile
    if ruta.suffix.lower() == ".wav":
        x, hz = leer_wav(ruta)
        if hz == RITMO:
            return x
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / "a.wav"
        r = subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(ruta),
                            "-ar", str(RITMO), "-ac", "1", "-c:a", "pcm_s16le", str(d)],
                           capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode()[:120])
        return leer_wav(d)[0]


def recortar(x, umbral=0.005, margen=int(0.05 * RITMO)):
    k = (len(x) // 240) * 240
    if not k:
        return x
    rms = np.sqrt((x[:k].reshape(-1, 240) ** 2).mean(1))
    vivo = np.where(rms > umbral)[0]
    if len(vivo) < 2:
        return x
    return x[max(0, vivo[0] * 240 - margen):min(len(x), (vivo[-1] + 1) * 240 + margen)]


# -------------------------------------------------------------- entrenar --
def subcomando_entrenar(args):
    disp = "mps" if torch.backends.mps.is_available() else "cpu"
    sucio = torch.from_numpy(np.load(Path(args.pares) / "sucio.npy"))
    limpio = torch.from_numpy(np.load(Path(args.pares) / "limpio.npy"))
    n = len(sucio)
    corte = max(1, int(n * 0.05))
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(7))
    val, ent = idx[:corte], idx[corte:]
    print(f"{n} trozos: {len(ent)} de entrenamiento, {len(val)} de validacion")

    red = PostFiltro(base=args.base, limite=args.limite).to(disp)
    par = sum(p.numel() for p in red.parameters())
    print(f"post-filtro: {par/1e6:.2f} M parametros")
    opt = torch.optim.AdamW(red.parameters(), lr=args.lr, weight_decay=1e-4)
    perdida_stft = PerdidaMultiSTFT(disp)
    perdida_mod = PerdidaModulacion(disp)

    def perdida(y, obj):
        return perdida_stft(y, obj) + args.peso_mod * perdida_mod(y, obj)
    pasos_total = args.epocas * math.ceil(len(ent) / args.lote)
    plan = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=pasos_total)

    mejor = float("inf")
    for ep in range(1, args.epocas + 1):
        red.train()
        perm = ent[torch.randperm(len(ent))]
        acum, nl, t0 = 0.0, 0, time.perf_counter()
        for k in range(0, len(perm) - args.lote + 1, args.lote):
            b = perm[k:k + args.lote]
            x = sucio[b].unsqueeze(1).to(disp); y = limpio[b].unsqueeze(1).to(disp)
            l = perdida(red(x), y)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(red.parameters(), 1.0)
            opt.step(); plan.step()
            acum += l.item(); nl += 1
        red.eval(); vac, vn = 0.0, 0
        with torch.no_grad():
            for k in range(0, len(val), args.lote):
                b = val[k:k + args.lote]
                x = sucio[b].unsqueeze(1).to(disp); y = limpio[b].unsqueeze(1).to(disp)
                vac += perdida(red(x), y).item(); vn += 1
                if vn == 1:      # linea base: no tocar nada
                    base_l = perdida(x, y).item()
        vl = vac / max(1, vn)
        marca = ""
        if vl < mejor:
            mejor = vl
            torch.save({"estado": red.state_dict(), "base": args.base,
                        "limite": args.limite}, args.salida)
            marca = "  <- guardado"
        print(f"epoca {ep:3d}  entren {acum/max(1,nl):.4f}  valid {vl:.4f}  "
              f"(sin filtro {base_l:.4f})  {time.perf_counter()-t0:.0f} s{marca}", flush=True)
    print(f"\nmejor validacion {mejor:.4f} en {args.salida}")


# --------------------------------------------------------------- aplicar --
def subcomando_aplicar(args):
    from espectro import textura
    disp = "cpu" if args.cpu else ("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(args.modelo, map_location=disp, weights_only=False)
    red = PostFiltro(base=ck.get("base", 24), limite=ck.get("limite", 0.0)).to(disp)
    red.load_state_dict(ck["estado"]); red.eval()

    x, hz = leer_wav(args.entrada)
    if hz != RITMO:
        raise SystemExit(f"el audio va a {hz} Hz y hace falta {RITMO}")
    t0 = time.perf_counter()
    with torch.no_grad():
        y = red(torch.from_numpy(x)[None, None].to(disp)).squeeze().cpu().numpy()
    dt = time.perf_counter() - t0
    escribir_wav(args.salida, y)
    print(f"{args.salida}  {len(x)/RITMO:.2f} s  RTF del filtro {dt/(len(x)/RITMO):.4f}")

    a, b = textura(x), textura(y)
    print(f"\n{'metrica':22} {'antes':>9} {'despues':>9} {'cambio':>9}")
    for k, nom in (("ataques_dB_s", "nitidez ataques"), ("planitud_sib", "planitud sibil."),
                   ("mod_4_8", "modulacion 4-8 Hz"), ("mod_8_16", "modulacion 8-16 Hz"),
                   ("mod_16_32", "modulacion 16-32 Hz"), ("mod_32_64", "modulacion 32-64 Hz")):
        c = 100 * (b[k] - a[k]) / a[k] if a[k] else 0.0
        print(f"{nom:22} {a[k]:9.4f} {b[k]:9.4f} {c:+8.1f}%")
    if args.referencia:
        r = textura(*leer_wav(args.referencia))
        print(f"\n{'':22} {'objetivo':>9}   (la referencia real)")
        for k, nom in (("mod_16_32", "modulacion 16-32 Hz"), ("mod_32_64", "modulacion 32-64 Hz")):
            print(f"{nom:22} {r[k]:9.4f}")


# ---------------------------------------------------------- adversarial --
def subcomando_adversarial(args):
    """Ajuste fino de un post-filtro que YA funciona, con discriminador.

    Se parte de un checkpoint entrenado, no de cero. Un GAN de audio desde
    cero es inestable y aqui no hace falta: v5 ya cierra el 55 % del hueco, y
    lo que se busca es la textura que las perdidas de reconstruccion no saben
    pedir. Ademas el limite del residuo sigue puesto, asi que por mucho que el
    discriminador empuje, el generador no puede inventar energia donde no la
    hay -- es un estabilizador estructural, no una esperanza.
    """
    disp = "mps" if torch.backends.mps.is_available() else "cpu"
    sucio = torch.from_numpy(np.load(Path(args.pares) / "sucio.npy"))
    limpio = torch.from_numpy(np.load(Path(args.pares) / "limpio.npy"))
    n = len(sucio)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(7))
    corte = max(1, int(n * 0.05))
    val, ent = idx[:corte], idx[corte:]

    ck = torch.load(args.partir_de, map_location="cpu", weights_only=False)
    gen = PostFiltro(base=ck.get("base", 24), limite=ck.get("limite", 0.0)).to(disp)
    gen.load_state_dict(ck["estado"])
    dis = Discriminador().to(disp)
    print(f"generador {sum(p.numel() for p in gen.parameters())/1e6:.2f} M "
          f"(desde {args.partir_de}), discriminador "
          f"{sum(p.numel() for p in dis.parameters())/1e6:.2f} M")
    print(f"{len(ent)} trozos de entrenamiento, {len(val)} de validacion")

    # lr bajo en el generador: viene de un optimo y solo hay que moverlo un poco
    og = torch.optim.AdamW(gen.parameters(), lr=args.lr_gen, betas=(0.8, 0.99))
    od = torch.optim.AdamW(dis.parameters(), lr=args.lr_dis, betas=(0.8, 0.99))
    p_stft, p_mod = PerdidaMultiSTFT(disp), PerdidaModulacion(disp)

    def reconstruccion(y, obj):
        return p_stft(y, obj) + args.peso_mod * p_mod(y, obj)

    mejor = float("inf")
    for ep in range(1, args.epocas + 1):
        gen.train(); dis.train()
        perm = ent[torch.randperm(len(ent))]
        ac_g = ac_d = ac_a = 0.0
        pasos = 0
        t0 = time.perf_counter()
        for k in range(0, len(perm) - args.lote + 1, args.lote):
            b = perm[k:k + args.lote]
            x = sucio[b].unsqueeze(1).to(disp)
            y = limpio[b].unsqueeze(1).to(disp)
            gy = gen(x)

            # --- discriminador: LSGAN, reales a 1 y falsos a 0 ---
            sr, _ = dis(y)
            sf, _ = dis(gy.detach())
            ld = sum(((r - 1) ** 2).mean() + (f ** 2).mean() for r, f in zip(sr, sf)) / len(sr)
            od.zero_grad(); ld.backward()
            torch.nn.utils.clip_grad_norm_(dis.parameters(), 1.0)
            od.step()

            # --- generador: reconstruccion + adversarial + rasgos ---
            sr, fr = dis(y)
            sf, ff = dis(gy)
            la = sum(((f - 1) ** 2).mean() for f in sf) / len(sf)
            lrec = reconstruccion(gy, y)
            lg = lrec + args.peso_adv * la + args.peso_rasgos * perdida_rasgos(fr, ff)
            og.zero_grad(); lg.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            og.step()

            ac_g += lrec.item(); ac_d += ld.item(); ac_a += la.item(); pasos += 1

        # la validacion se juzga SOLO por reconstruccion: la perdida
        # adversarial no es comparable entre epocas porque el discriminador
        # cambia bajo ella
        gen.eval(); vac, vn = 0.0, 0
        with torch.no_grad():
            for k in range(0, len(val), args.lote):
                b = val[k:k + args.lote]
                x = sucio[b].unsqueeze(1).to(disp); y = limpio[b].unsqueeze(1).to(disp)
                vac += reconstruccion(gen(x), y).item(); vn += 1
        vl = vac / max(1, vn)
        marca = ""
        if vl < mejor:
            mejor = vl
            torch.save({"estado": gen.state_dict(), "base": ck.get("base", 24),
                        "limite": ck.get("limite", 0.0)}, args.salida)
            marca = "  <- guardado"
        # siempre se guarda el ultimo tambien: en un GAN el mejor por
        # reconstruccion no tiene por que ser el que mejor suena
        torch.save({"estado": gen.state_dict(), "base": ck.get("base", 24),
                    "limite": ck.get("limite", 0.0)}, str(args.salida) + ".ultimo")
        print(f"epoca {ep:3d}  recon {ac_g/pasos:.4f}  disc {ac_d/pasos:.4f}  "
              f"adv {ac_a/pasos:.4f}  valid {vl:.4f}  "
              f"{time.perf_counter()-t0:.0f} s{marca}", flush=True)
    print(f"\nmejor validacion por reconstruccion: {mejor:.4f}")
    print(f"guardados {args.salida} (mejor) y {args.salida}.ultimo")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("datos", help="corpus -> pares (ciclo, original)")
    d.add_argument("--corpus", required=True)
    d.add_argument("--salida", required=True)
    d.add_argument("--limite", type=int, default=0, help="solo N ficheros")
    d.add_argument("--mezclar", action="store_true",
                   help="baraja el orden con semilla fija; imprescindible con --max-min")
    d.add_argument("--max-min", type=float, default=0, help="parar a los N minutos de audio")
    d.add_argument("--ffmpeg", default=os.environ.get("VOZ_FFMPEG", "ffmpeg"))
    d.add_argument("--modelo", default=os.environ.get(
        "VIBEVOICE_MODELO", str(Path.home() / ".cache/vibevoice-nix/modelo")))
    d.add_argument("--cache", default=str(Path.home() / ".cache/vibevoice-nix"))
    d.set_defaults(f=subcomando_datos)

    e = sub.add_parser("entrenar", help="pares -> post-filtro")
    e.add_argument("--pares", required=True)
    e.add_argument("--salida", default="postfiltro.pt")
    e.add_argument("--epocas", type=int, default=40)
    e.add_argument("--lote", type=int, default=16)
    e.add_argument("--lr", type=float, default=3e-4)
    e.add_argument("--base", type=int, default=24, help="canales del primer nivel")
    e.add_argument("--limite", type=float, default=0.0,
                   help="tope del residuo como fraccion de la entrada, por celda "
                        "tiempo-frecuencia. 0 = sin limite")
    e.add_argument("--peso-mod", type=float, default=1.0,
                   help="peso de la perdida de modulacion frente a la multi-STFT")
    e.set_defaults(f=subcomando_entrenar)

    v = sub.add_parser("adversarial", help="ajuste fino con discriminador")
    v.add_argument("--pares", required=True)
    v.add_argument("--partir-de", required=True, help="checkpoint que ya funciona")
    v.add_argument("--salida", default="postfiltro-adv.pt")
    v.add_argument("--epocas", type=int, default=10)
    v.add_argument("--lote", type=int, default=16)
    v.add_argument("--lr-gen", type=float, default=5e-5)
    v.add_argument("--lr-dis", type=float, default=2e-4)
    v.add_argument("--peso-mod", type=float, default=1.0)
    v.add_argument("--peso-adv", type=float, default=1.0)
    v.add_argument("--peso-rasgos", type=float, default=2.0)
    v.set_defaults(f=subcomando_adversarial)

    a = sub.add_parser("aplicar", help="pasar un wav por el post-filtro")
    a.add_argument("entrada"); a.add_argument("salida")
    a.add_argument("--modelo", default="postfiltro.pt")
    a.add_argument("--referencia", help="wav real, para ver el objetivo")
    a.add_argument("--cpu", action="store_true")
    a.set_defaults(f=subcomando_aplicar)

    args = ap.parse_args()
    return args.f(args) or 0


if __name__ == "__main__":
    sys.exit(main())
