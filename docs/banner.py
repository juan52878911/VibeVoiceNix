"""Genera docs/banner.svg: la portada ochentera (synthwave) del proyecto.

El SVG del repositorio es la salida de este script, no un fichero dibujado a mano.
Se versiona el generador para poder rehacer el banner —cambiar colores, texto o los
chips de los motores— sin partir del XML.

    python3 docs/banner.py

Sin dependencias: solo la biblioteca estandar. Las posiciones de las estrellas salen
de una secuencia de Halton y la onda de una envolvente de gaussianas, asi que la
salida es determinista y dos ejecuciones dan el mismo fichero.
"""
import math
import os

W, H = 1200, 520
HY = 320                     # horizonte
SUN_CX, SUN_CY, SUN_R = 600, HY, 158

out = []
a = out.append

a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="VibeVoiceNix - stack de voz local para el homelab en NixOS y Proxmox">')
a('<title>VibeVoiceNix — voz local para el homelab (NixOS + Proxmox)</title>')

# ----------------------------------------------------------------- defs
a('<defs>')
a('''
  <linearGradient id="cielo" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%"   stop-color="#08010f"/>
    <stop offset="38%"  stop-color="#1b0533"/>
    <stop offset="72%"  stop-color="#4a0d5c"/>
    <stop offset="100%" stop-color="#8d1a67"/>
  </linearGradient>

  <linearGradient id="sol" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%"   stop-color="#fff2a8"/>
    <stop offset="28%"  stop-color="#ffcf4d"/>
    <stop offset="58%"  stop-color="#ff8330"/>
    <stop offset="100%" stop-color="#ff1f6d"/>
  </linearGradient>

  <linearGradient id="suelo" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%"   stop-color="#2a0533"/>
    <stop offset="100%" stop-color="#0b0118"/>
  </linearGradient>

  <linearGradient id="cromo" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%"   stop-color="#ffffff"/>
    <stop offset="34%"  stop-color="#cdefff"/>
    <stop offset="50%"  stop-color="#7fd4ff"/>
    <stop offset="52%"  stop-color="#ffb3dd"/>
    <stop offset="76%"  stop-color="#ff4f9a"/>
    <stop offset="100%" stop-color="#ff1f6d"/>
  </linearGradient>

  <radialGradient id="halo" cx="50%" cy="50%" r="50%">
    <stop offset="0%"   stop-color="#ff3d8b" stop-opacity="0.55"/>
    <stop offset="55%"  stop-color="#c02a8f" stop-opacity="0.18"/>
    <stop offset="100%" stop-color="#c02a8f" stop-opacity="0"/>
  </radialGradient>

  <filter id="brillo" x="-60%" y="-60%" width="220%" height="220%">
    <feGaussianBlur stdDeviation="7" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>

  <filter id="brilloSuave" x="-60%" y="-60%" width="220%" height="220%">
    <feGaussianBlur stdDeviation="3" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>

  <pattern id="scan" width="4" height="4" patternUnits="userSpaceOnUse">
    <rect width="4" height="2" fill="#000000" opacity="0.16"/>
  </pattern>

  <linearGradient id="desvanece" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%"   stop-color="#000000"/>
    <stop offset="14%"  stop-color="#555555"/>
    <stop offset="55%"  stop-color="#ffffff"/>
    <stop offset="100%" stop-color="#dddddd"/>
  </linearGradient>
''')
a(f'  <mask id="mSuelo"><rect x="0" y="{HY}" width="{W}" height="{H - HY}" fill="url(#desvanece)"/></mask>')

# Mascara del sol: disco blanco con franjas negras (las ranuras clasicas).
a('<mask id="mSol">')
a(f'  <circle cx="{SUN_CX}" cy="{SUN_CY}" r="{SUN_R}" fill="#ffffff"/>')
y, h, hueco = float(HY), 13.0, 5.0
while y > HY - SUN_R * 0.82:
    a(f'  <rect x="{SUN_CX - SUN_R}" y="{y - h:.1f}" width="{SUN_R * 2}" height="{h:.1f}" fill="#000000"/>')
    y -= h + hueco
    h *= 0.80
    hueco *= 1.16
a('</mask>')
a('</defs>')

# ----------------------------------------------------------------- cielo
a(f'<rect width="{W}" height="{H}" fill="url(#cielo)"/>')

# Estrellas: posiciones deterministas via halton, sin random.
def halton(i, base):
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r

a('<g fill="#ffffff">')
for i in range(1, 130):
    sx = halton(i, 2) * W
    sy = halton(i, 3) * (HY - 30)
    # ni dentro del sol ni pegadas al horizonte
    if math.hypot(sx - SUN_CX, sy - SUN_CY) < SUN_R + 16:
        continue
    r = 0.7 + (i % 5) * 0.28
    op = 0.25 + ((i * 37) % 60) / 100
    a(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{r:.2f}" opacity="{op:.2f}"/>')
a('</g>')

# Halo y sol
a(f'<ellipse cx="{SUN_CX}" cy="{SUN_CY}" rx="{SUN_R * 2.9:.0f}" ry="{SUN_R * 1.5:.0f}" fill="url(#halo)"/>')
a(f'<g mask="url(#mSol)"><circle cx="{SUN_CX}" cy="{SUN_CY}" r="{SUN_R}" fill="url(#sol)"/></g>')

# Sierra lejana: da profundidad y tapa la base del sol.
picos = [(0, 22), (90, 8), (160, 30), (250, 12), (330, 36), (430, 17),
         (520, 26), (600, 9), (690, 29), (780, 14), (880, 33), (980, 13),
         (1070, 25), (1150, 10), (1200, 20)]
d = f'M 0 {HY} '
for x, alto in picos:
    d += f'L {x} {HY - alto} '
d += f'L {W} {HY} Z'
a(f'<path d="{d}" fill="#150329" opacity="0.92"/>')
a(f'<path d="{d}" fill="none" stroke="#ff2a6d" stroke-width="1.2" opacity="0.45"/>')

# ----------------------------------------------------------------- suelo
a(f'<rect x="0" y="{HY}" width="{W}" height="{H - HY}" fill="url(#suelo)"/>')
a('<g mask="url(#mSuelo)" stroke-linecap="round">')
# Fugas hacia el punto de fuga
for k in range(-24, 25):
    xb = SUN_CX + k * 96
    ancho = 1.9 if k % 4 == 0 else 1.0
    color = "#05d9e8" if k % 4 == 0 else "#ff2a6d"
    a(f'  <line x1="{SUN_CX}" y1="{HY}" x2="{xb}" y2="{H}" stroke="{color}" stroke-width="{ancho}" opacity="0.85"/>')
# Transversales con espaciado en perspectiva
n = 11
for i in range(1, n + 1):
    yy = HY + (H - HY) * (i * i) / (n * n)
    ancho = 1.0 + 1.6 * (i / n)
    a(f'  <line x1="0" y1="{yy:.1f}" x2="{W}" y2="{yy:.1f}" stroke="#ff2a6d" stroke-width="{ancho:.2f}" opacity="0.8"/>')
a('</g>')

# ----------------------------------------------------------------- onda
# El horizonte ES la senal de voz: envolvente tipo habla sobre una portadora.
def envolvente(x):
    picos_voz = [(120, 70), (300, 95), (470, 60), (760, 88), (930, 66), (1080, 44)]
    v = 0.0
    for c, w in picos_voz:
        v += math.exp(-((x - c) ** 2) / (2.0 * w * w))
    return min(v, 1.25)

pts = []
for x in range(0, W + 1, 3):
    amp = 17 * envolvente(x)
    y = HY + amp * math.sin(x / 7.5) * math.sin(x / 41.0 + 0.7)
    pts.append(f'{x},{y:.1f}')
onda = ' '.join(pts)
a(f'<polyline points="{onda}" fill="none" stroke="#05d9e8" stroke-width="2.4" opacity="0.95" filter="url(#brillo)"/>')
a(f'<line x1="0" y1="{HY}" x2="{W}" y2="{HY}" stroke="#7ff9ff" stroke-width="1" opacity="0.35"/>')

# ----------------------------------------------------------------- texto
FUENTE = "Arial Black, Arial Bold, Helvetica Neue, Helvetica, DejaVu Sans, sans-serif"
a(f'<g font-family="{FUENTE}" text-anchor="middle">')
a(f'  <text x="{SUN_CX}" y="210" font-size="82" font-weight="900" letter-spacing="7" '
  f'fill="#ff1f6d" opacity="0.85" filter="url(#brillo)">VIBEVOICE NIX</text>')
a(f'  <text x="{SUN_CX}" y="206" font-size="82" font-weight="900" letter-spacing="7" '
  f'fill="url(#cromo)" stroke="#0b0118" stroke-width="1.2" paint-order="stroke">VIBEVOICE NIX</text>')
a(f'  <text x="{SUN_CX}" y="252" font-size="19" font-weight="700" letter-spacing="6.5" '
  f'fill="#7ff9ff" stroke="#0b0118" stroke-width="3.5" paint-order="stroke" '
  f'filter="url(#brilloSuave)">VOZ LOCAL PARA EL HOMELAB</text>')
a('</g>')

# Chips de los tres motores
chips = [("PIPER  ·  TTS", "#05d9e8"), ("WHISPER.CPP  ·  STT", "#05d9e8"), ("VIBEVOICE  ·  LAB", "#ff2a6d")]
ancho_chip, alto_chip, sep = 250, 36, 22
total = len(chips) * ancho_chip + (len(chips) - 1) * sep
x0 = (W - total) / 2
cy = 462
a(f'<g font-family="{FUENTE}" text-anchor="middle">')
for i, (etq, color) in enumerate(chips):
    cx = x0 + i * (ancho_chip + sep)
    a(f'  <rect x="{cx:.0f}" y="{cy}" width="{ancho_chip}" height="{alto_chip}" rx="18" '
      f'fill="#0b0118" opacity="0.82"/>')
    a(f'  <rect x="{cx:.0f}" y="{cy}" width="{ancho_chip}" height="{alto_chip}" rx="18" '
      f'fill="none" stroke="{color}" stroke-width="1.4" opacity="0.9"/>')
    a(f'  <text x="{cx + ancho_chip / 2:.0f}" y="{cy + 24}" font-size="14" font-weight="700" '
      f'letter-spacing="2.2" fill="{color}">{etq}</text>')
a('</g>')

# ----------------------------------------------------------------- CRT
a(f'<rect width="{W}" height="{H}" fill="url(#scan)"/>')
a(f'<rect x="6" y="6" width="{W - 12}" height="{H - 12}" rx="10" fill="none" '
  f'stroke="#05d9e8" stroke-width="1.6" opacity="0.35"/>')
a('</svg>')

destino = os.path.join(os.path.dirname(os.path.abspath(__file__)), "banner.svg")
with open(destino, "w") as f:
    f.write("\n".join(out) + "\n")
print("escrito", destino)
