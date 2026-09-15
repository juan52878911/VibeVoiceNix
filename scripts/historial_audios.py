#!/usr/bin/env python3
"""Genera una web estática para escuchar el historial de audios archivados en el NAS.

    python3 historial_audios.py /srv/archivo/vibevoice-historial

Recorre el directorio (una carpeta por sesión de trabajo, copiada tal cual desde el Mac con
scripts/subir_historial.sh) y escribe:
  - index.html: las sesiones, con cuántos audios tiene cada una y cuánto ocupan;
  - _historial/<sesión>.html: una tabla por carpeta de experimento con un reproductor por audio
    (preload=none: la página no descarga nada hasta darle a play), filtro de texto y, cuando la
    carpeta trae medidas (calidad.json, medidas.json, resultado.json), música AST, WER, ECAPA y UTMOS
    junto a cada clip.

Solo usa la biblioteca estándar: corre en el CT del NAS sin instalar nada. Es idempotente; se vuelve a
lanzar después de cada subida. Los audios son datos biométricos de identidades con consentimiento:
la web es solo para la LAN.
"""
import html
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

AUDIO = {".wav", ".mp3", ".ogg", ".opus", ".flac", ".m4a"}
TIPO = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".opus": "audio/ogg",
        ".flac": "audio/flac", ".m4a": "audio/mp4"}
METRICAS = ("musica", "wer", "ecapa", "utmos", "dur", "dicho", "oido", "cond", "voz", "texto", "semilla")

ESTILO = """
:root{--f:#fbfaf7;--t:#1d1d1b;--s:#6b6a65;--l:#e4e1d8;--a:#2f5d8a;--m:#b3411f;--ok:#2e7d4f}
@media (prefers-color-scheme:dark){:root{--f:#161614;--t:#ecebe6;--s:#9a988f;--l:#2e2d29;--a:#8fb8e3;--m:#f08a64;--ok:#7cc79a}}
*{box-sizing:border-box}body{margin:0;padding:24px 16px;background:var(--f);color:var(--t);
font:14px/1.45 -apple-system,system-ui,Segoe UI,sans-serif}main{max-width:1200px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 6px}p.s,span.s{color:var(--s)}
a{color:var(--a)}table{border-collapse:collapse;width:100%}th,td{padding:5px 8px;border-bottom:1px solid var(--l);
text-align:left;vertical-align:middle}th{font-weight:600;color:var(--s);font-size:12px}
td.n{font-variant-numeric:tabular-nums;white-space:nowrap}audio{height:32px;width:240px;max-width:100%}
.tabla{overflow-x:auto}.mus{color:var(--m);font-weight:600}.ok{color:var(--ok)}
input{width:100%;max-width:420px;padding:8px 10px;border:1px solid var(--l);border-radius:6px;background:transparent;color:var(--t)}
details{margin:8px 0}summary{cursor:pointer;font-weight:600}code{font-size:12px}
"""


def tam(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def medidas_de(carpeta: Path) -> dict:
    """{nombre de fichero: {métrica: valor}} con lo que traiga la carpeta."""
    m = {}
    # También la carpeta de arriba: elegir_arranque.py deja los WAV en <salida>/wav y medidas.json en <salida>.
    for ruta in [c / n for c in (carpeta.parent, carpeta) for n in ("calidad.json", "resultado.json", "medidas.json")]:
        if not ruta.exists():
            continue
        try:
            datos = json.loads(ruta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        filas = datos.items() if isinstance(datos, dict) else ((None, f) for f in datos)
        for clave, fila in filas:
            if not isinstance(fila, dict):
                continue
            fichero = fila.get("fichero") or (clave if clave and str(clave).endswith(tuple(AUDIO)) else None)
            if fichero is None and {"voz", "texto", "semilla", "cond"} <= fila.keys():
                fichero = f"{fila['voz']}__{fila['texto']}__s{fila['semilla']}__{fila['cond']}.wav"
            if fichero:
                m.setdefault(Path(str(fichero)).name, {}).update(
                    {k: fila[k] for k in METRICAS if k in fila and not isinstance(fila[k], (list, dict))})
    return m


def celda(k, v):
    if v is None:
        return "<td></td>"
    if isinstance(v, float):
        clase = ' class="n mus"' if k == "musica" and v > 0.2 else ' class="n"'
        return f"<td{clase}>{v:.3f}</td>"
    return f"<td>{html.escape(str(v))[:160]}</td>"


def pagina_sesion(raiz: Path, sesion: Path, salida: Path):
    grupos = {}
    for dirpath, _, ficheros in os.walk(sesion):
        audios = sorted(f for f in ficheros if Path(f).suffix.lower() in AUDIO)
        if audios:
            grupos[Path(dirpath)] = audios
    total, bytes_ = 0, 0
    partes = [f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
              f"<title>{html.escape(sesion.name)} · historial de audios</title><style>{ESTILO}</style><main>"
              f"<p><a href='../index.html'>← todas las sesiones</a></p><h1>{html.escape(sesion.name)}</h1>"
              f"<p><input id=f placeholder='Filtrar por nombre, voz, condición…' oninput='filtrar(this.value)'></p>"]
    for carpeta in sorted(grupos):
        audios = grupos[carpeta]
        med = medidas_de(carpeta)
        cols = [k for k in ("cond", "musica", "wer", "ecapa", "utmos", "dur") if any(k in med.get(a, {}) for a in audios)]
        rel = carpeta.relative_to(sesion).as_posix() or "."
        n_mus = sum(1 for a in audios if (med.get(a, {}).get("musica") or 0) > 0.2)
        resumen = f"{len(audios)} audios" + (f" · <span class=mus>{n_mus} con música &gt; 0,2</span>" if "musica" in cols else "")
        partes.append(f"<details open><summary>{html.escape(rel)} <span class=s>({resumen})</span></summary><div class=tabla><table>"
                      "<tr><th>audio</th><th>fichero</th>" + "".join(f"<th>{c}</th>" for c in cols) + "<th>transcripción</th></tr>")
        for a in audios:
            ruta = carpeta / a
            try:
                bytes_ += ruta.stat().st_size
            except OSError:
                pass
            total += 1
            url = "../" + quote(ruta.relative_to(raiz).as_posix())
            fila = med.get(a, {})
            dicho = fila.get("dicho") or fila.get("oido") or ""
            partes.append(f"<tr data-t='{html.escape((rel + ' ' + a + ' ' + dicho).lower(), quote=True)}'>"
                          f"<td><audio controls preload=none src='{url}' type='{TIPO.get(ruta.suffix.lower(), '')}'></audio></td>"
                          f"<td><a href='{url}' download>{html.escape(a)}</a></td>"
                          + "".join(celda(c, fila.get(c)) for c in cols)
                          + f"<td class=s>{html.escape(dicho)[:160]}</td></tr>")
        partes.append("</table></div></details>")
    partes.append("<script>function filtrar(q){q=q.toLowerCase();for(const r of document.querySelectorAll('tr[data-t]'))"
                  "r.hidden=q&&!r.dataset.t.includes(q);}</script></main>")
    salida.write_text("".join(partes), encoding="utf-8")
    return total, bytes_, len(grupos)


def main():
    raiz = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    destino = raiz / "_historial"
    destino.mkdir(exist_ok=True)
    filas = []
    for sesion in sorted((p for p in raiz.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))), reverse=True):
        n, b, g = pagina_sesion(raiz, sesion, destino / f"{sesion.name}.html")
        filas.append((sesion.name, n, b, g))
    cuerpo = "".join(f"<tr><td><a href='_historial/{quote(s)}.html'>{html.escape(s)}</a></td><td class=n>{n}</td>"
                     f"<td class=n>{g}</td><td class=n>{tam(b)}</td></tr>" for s, n, b, g in filas)
    (raiz / "index.html").write_text(
        f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>Historial de audios VibeVoiceNix</title><style>{ESTILO}</style><main><h1>Historial de audios VibeVoiceNix</h1>"
        f"<p class=s>{sum(f[1] for f in filas)} audios en {len(filas)} sesiones · generado {time.strftime('%Y-%m-%d %H:%M')} · "
        f"solo LAN; datos de voz de identidades con consentimiento</p>"
        f"<div class=tabla><table><tr><th>sesión</th><th>audios</th><th>carpetas</th><th>tamaño de audio</th></tr>{cuerpo}</table></div>"
        f"<p class=s>Los ficheros originales (scripts, JSON, clones .pt) están junto a los audios: "
        f"<code>\\\\192.168.2.65\\archivo\\vibevoice-historial</code></p></main>", encoding="utf-8")
    print(f"{sum(f[1] for f in filas)} audios en {len(filas)} sesiones -> {raiz / 'index.html'}")


if __name__ == "__main__":
    main()
