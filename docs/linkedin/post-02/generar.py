"""Genera las imágenes del post 2 de la serie de LinkedIn: la tabla de optimización.

Salida en este mismo directorio:
    slide-1.png, slide-2.png, slide-3.png   1080x1350 (4:5, el formato vertical del feed)
    carrusel.pdf                            las tres juntas, para subir como documento

Todo con herramientas libres: HTML escrito aquí, renderizado con Chromium headless,
tipografías Inter y JetBrains Mono (licencia OFL) y el PDF montado con Pillow.

    python3 docs/linkedin/post-02/generar.py

Las cifras salen de docs/optimizacion.md y docs/plan-rendimiento.md; si cambian allí,
se cambian en PASOS y NO_ENTRO y se vuelve a generar.
"""
import glob
import os
import shutil
import subprocess
import tempfile

from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
W, H = 1080, 1350

# (cambio, detalle, RTF, ganancia) — mismo i7-8700T, medianas de banco
PASOS = [
    ("Punto de partida", "fp32 · 20 pasos", 5.39, None),
    ("Cuantización int8 dinámica", "PyTorch · pesos a ¼", 2.75, "1,96×"),
    ("6 pasos de difusión, no 20", "DPM-Solver multipaso", 2.18, "1,26×"),
    ("Motor OpenVINO", "grafos compilados", 1.09, "2,00×"),
    ("Decodificador sin conv. traspuesta", "k = 2s → producto de matrices", 0.98, "1,11×"),
    ("Bucle de difusión en un solo grafo", "6 pasos + guía + solver, 1 llamada", 0.885, "1,06×"),
]

NO_ENTRO = [
    ("La iGPU del propio i7", "−6,1 %", "la meta pedía −7 %"),
    ("LM en int8 en vez de int4", "+9 %", "más lento y no suena mejor"),
    ("Dos pasadas del LM en paralelo", "−7 %", "la meta pedía −20 %"),
    ("Enseñar a la VM su topología SMT", "+0,6 %", "ruido"),
    ("torch.compile", "1,00×", "nada"),
]

CSS = """
@font-face { font-family: I; src: local('Inter'); }
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { width: %(W)dpx; height: %(H)dpx; }
body {
  font-family: 'Inter', sans-serif; color: #f4ecff; overflow: hidden;
  background:
    radial-gradient(900px 600px at 85%% -10%%, rgba(255,31,109,.28), transparent 60%%),
    radial-gradient(800px 700px at -10%% 110%%, rgba(127,212,255,.16), transparent 60%%),
    linear-gradient(180deg, #0c0218 0%%, #150428 55%%, #0a0114 100%%);
  position: relative; padding: 84px 80px;
}
body::after {  /* rejilla synthwave muy tenue, como el banner del repo */
  content: ''; position: absolute; inset: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(255,79,154,.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,79,154,.05) 1px, transparent 1px);
  background-size: 54px 54px;
  mask-image: linear-gradient(180deg, transparent 0%%, #000 100%%);
}
.mono { font-family: 'JetBrains Mono', monospace; }
.top { display: flex; justify-content: space-between; align-items: center;
       font-family: 'JetBrains Mono', monospace; font-size: 22px; letter-spacing: 3px;
       color: #b79bd6; text-transform: uppercase; }
.top b { color: #ff4f9a; font-weight: 800; }
.foot { position: absolute; left: 80px; right: 80px; bottom: 70px;
        display: flex; justify-content: space-between; align-items: center;
        font-family: 'JetBrains Mono', monospace; font-size: 21px; color: #9c86b8; }
.foot .gh { color: #f4ecff; }
.chip { display: inline-block; border: 2px solid rgba(255,79,154,.55); color: #ffb3dd;
        border-radius: 999px; padding: 8px 20px; font-size: 22px; font-weight: 600;
        font-family: 'JetBrains Mono', monospace; margin-right: 12px; }
.grad { background: linear-gradient(180deg, #ffffff 0%%, #cdefff 40%%, #7fd4ff 60%%, #ff4f9a 100%%);
        -webkit-background-clip: text; background-clip: text; color: transparent; }
.hot { background: linear-gradient(90deg, #ffcf4d, #ff8330 45%%, #ff1f6d);
       -webkit-background-clip: text; background-clip: text; color: transparent; }
""" % {"W": W, "H": H}


def pagina(cuerpo, n):
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
<div class="top"><span><b>VibeVoiceNix</b> · 02 / 08</span><span>{n} / 3</span></div>
{cuerpo}
<div class="foot"><span>i7-8700T · 35 W · sin GPU</span><span class="gh">github.com/juan52878911/VibeVoiceNix</span></div>
</body></html>"""


def slide1():
    total = PASOS[0][2] / PASOS[-1][2]
    return pagina(f"""
<div style="margin-top:150px">
  <div class="mono" style="font-size:30px;color:#b79bd6;letter-spacing:4px">RTF · VIBEVOICE 0.5B</div>
  <div class="mono" style="font-size:176px;font-weight:800;line-height:1;margin-top:26px;letter-spacing:-6px">
    <span style="color:#6f5a8a;text-decoration:line-through;text-decoration-thickness:8px;text-decoration-color:#ff1f6d">5,39</span>
  </div>
  <div class="mono grad" style="font-size:250px;font-weight:800;line-height:1.02;letter-spacing:-10px">0,885</div>
</div>
<div style="margin-top:56px;font-size:88px;font-weight:900;letter-spacing:-2px;line-height:1">
  <span class="hot">{total:.1f}×</span>&nbsp;más rápido.
</div>
<div style="margin-top:26px;font-size:40px;font-weight:500;color:#d9c9f0;line-height:1.3">
  La misma CPU. Sin GPU. Sin nube.
</div>
<div style="margin-top:54px">
  <span class="chip">RTF &lt; 1 = más rápido que el tiempo real</span>
</div>
""".replace(f"{total:.1f}", f"{total:.1f}".replace(".", ",")), 1)


def fmt(x):
    return (f"{x:.3f}".rstrip("0").rstrip(".") if x < 1 else f"{x:.2f}").replace(".", ",")


def slide2():
    maxv = PASOS[0][2]
    ancho = 700  # px de barra para el RTF máximo
    filas = []
    for i, (cambio, detalle, rtf, gan) in enumerate(PASOS):
        ultimo = i == len(PASOS) - 1
        w = max(10, ancho * rtf / maxv)
        color = ("linear-gradient(90deg,#7fd4ff,#b8f3ff)" if rtf < 1
                 else "linear-gradient(90deg,#ff1f6d,#ff8330)")
        glow = "box-shadow:0 0 30px rgba(127,212,255,.8);" if ultimo else ""
        borde = "none" if ultimo else "1px solid rgba(183,155,214,.18)"
        filas.append(f"""
<div style="padding:20px 0 22px;border-bottom:{borde}">
  <div style="display:flex;justify-content:space-between;align-items:baseline">
    <div style="font-size:{33 if ultimo else 30}px;font-weight:{800 if ultimo else 700};color:{'#ffffff' if ultimo else '#eadcff'}">{cambio}</div>
    <div class="mono" style="font-size:28px;font-weight:700;color:{'#ffcf4d' if gan else '#6f5a8a'}">{gan or '—'}</div>
  </div>
  <div style="display:flex;align-items:center;gap:18px;margin-top:12px">
    <div style="height:18px;width:{w:.0f}px;border-radius:9px;background:{color};{glow}"></div>
    <div class="mono" style="font-size:{40 if ultimo else 28}px;font-weight:800;color:{'#b8f3ff' if rtf < 1 else '#ffb38a'}">{fmt(rtf)}</div>
    <div class="mono" style="font-size:18px;color:#8a74a6;margin-left:auto;white-space:nowrap">{detalle}</div>
  </div>
</div>""")
    return pagina(f"""
<div style="margin-top:44px;font-size:62px;font-weight:900;letter-spacing:-1.5px;line-height:1.05">
  Cinco cambios.<br><span class="hot">La misma CPU.</span>
</div>
<div class="mono" style="display:flex;justify-content:space-between;margin-top:36px;font-size:19px;color:#9c86b8;letter-spacing:2px">
  <span>CAMBIO · RTF</span><span>GANANCIA</span>
</div>
<div style="margin-top:4px">{''.join(filas)}</div>
<div class="mono" style="margin-top:18px;font-size:20px;color:#9c86b8">
  <span style="color:#ff8330">■</span> más lento que el tiempo real &nbsp; <span style="color:#7fd4ff">■</span> más rápido
</div>
""", 2)


def slide3():
    filas = "".join(f"""
<div style="display:grid;grid-template-columns:1fr 190px;align-items:center;padding:26px 0;border-bottom:1px solid rgba(183,155,214,.18)">
  <div>
    <div style="font-size:34px;font-weight:700;color:#eadcff">{idea}</div>
    <div class="mono" style="font-size:20px;color:#9c86b8;margin-top:6px">{porque}</div>
  </div>
  <div class="mono" style="text-align:right;font-size:36px;font-weight:800;color:#ff4f9a">{cifra}</div>
</div>""" for idea, cifra, porque in NO_ENTRO)
    return pagina(f"""
<div style="margin-top:60px;font-size:66px;font-weight:900;letter-spacing:-1.5px;line-height:1.05">
  Lo que <span class="hot">no</span> entró<br>en la tabla.
</div>
<div style="margin-top:26px;font-size:32px;color:#d9c9f0;line-height:1.35;font-weight:500">
  Cada idea, contra una meta escrita <b style="color:#fff">antes</b> de medir.
</div>
<div style="margin-top:34px">{filas}</div>
<div style="margin-top:44px;font-size:40px;font-weight:800;line-height:1.2">
  Todo medido. <span class="grad">Todo documentado.</span>
</div>
""", 3)


def chromium():
    for c in ("chromium", "chromium-browser", "google-chrome"):
        if shutil.which(c):
            return shutil.which(c)
    candidatos = glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")
    if candidatos:
        return sorted(candidatos)[-1]
    raise SystemExit("No encuentro Chromium")


def render(html, png):
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "s.html")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(html)
        subprocess.run([chromium(), "--headless=new", "--no-sandbox", "--disable-gpu",
                        "--hide-scrollbars", "--force-device-scale-factor=1",
                        f"--window-size={W},{H + 200}", f"--screenshot={png}", f"file://{f}"],
                       check=True, capture_output=True)
    Image.open(png).crop((0, 0, W, H)).save(png)


def main():
    pngs = []
    for i, gen in enumerate((slide1, slide2, slide3), 1):
        png = os.path.join(AQUI, f"slide-{i}.png")
        render(gen(), png)
        pngs.append(png)
        print("ok", png)
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    pdf = os.path.join(AQUI, "carrusel.pdf")
    imgs[0].save(pdf, save_all=True, append_images=imgs[1:], resolution=144)
    print("ok", pdf)


if __name__ == "__main__":
    main()
