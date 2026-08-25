#!/usr/bin/env python
"""PRUEBA DE PUNTA A PUNTA DEL ASISTENTE CON HERRAMIENTAS, POR CODIGO.

    pkgs/vibevoice/.venv/bin/python scripts/herramientas_extremo.py
    pkgs/vibevoice/.venv/bin/python scripts/herramientas_extremo.py --puerto 8099
    pkgs/vibevoice/.venv/bin/python scripts/herramientas_extremo.py --solo confirmacion

QUE PRUEBA, Y POR QUE ASI
Levanta un puente de verdad en un puerto aparte, le manda preguntas por
POST /preguntar y lee los MISMOS marcos binarios que leeria el navegador.
No hay navegador, no hay microfono y NO SE REPRODUCE NADA: el PCM se cuenta y
se tira. Asi se puede afirmar con numeros que:

  - el modelo llama a la herramienta que toca en cada pregunta
  - lo que se dice en voz alta tiene sentido y NO LLEVA JSON DENTRO
  - las acciones que escriben preguntan antes, y no escriben si no hay un si
  - un "sí" suelto, sin nada pendiente, no dispara nada

Y de paso mide lo unico que se nota al usarlo: cuanto tarda en sonar el primer
byte, con herramienta y sin ella.

EN OTRO PUERTO, Y ESO NO ES UN DETALLE. El puente del usuario escucha en el
8090; levantar otro encima lo mataria o fallaria. Aqui se arranca uno propio
en el 8099 con su propio fichero de datos -- una COPIA del de verdad -- para
que las escrituras de la prueba no le toquen el calendario a nadie.
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def leer_respuesta(url, cuerpo, plazo=180):
    """Manda una pregunta y devuelve (texto dicho, sucesos, tiempos).

    Lee el flujo de marcos [tipo:1][largo:4 BE][carga] igual que la pagina. El
    PCM (tipo 0) se cuenta y se tira: aqui no suena nada."""
    pet = urllib.request.Request(f"{url}/preguntar", method="POST",
                                 data=json.dumps(cuerpo).encode(),
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    r = urllib.request.urlopen(pet, timeout=plazo)
    resto, dicho, sucesos = b"", [], []
    # `primer_sonido` es lo que OYE el usuario, venga de donde venga: una
    # coletilla pregenerada cuenta igual que la voz sintetizada, porque lo que
    # se mide es el silencio, no la procedencia. `primer_pcm` es aparte para
    # poder separar las dos cosas y que no parezca que todo va a 1,2 s por
    # arte de magia.
    marcas = {"primer_sonido": None, "primer_texto": None,
              "primer_relleno": None, "primer_pcm": None}
    muestras = 0
    while True:
        # read1 y NO read: read(65536) espera a tener los 65536 bytes o a que
        # cierren, asi que el primer marco no aparece hasta que hay 64 KB de
        # PCM detras -- unos segundos. Con eso, medir "primer sonido" aqui
        # daba 6,8 s donde el puente decia 1,1 s, y la culpa era del medidor.
        # read1 devuelve lo que haya llegado.
        trozo = r.read1(65536)
        if not trozo:
            break
        resto += trozo
        while len(resto) >= 5:
            tipo, largo = struct.unpack(">BI", resto[:5])
            if len(resto) - 5 < largo:
                break
            carga, resto = resto[5:5 + largo], resto[5 + largo:]
            if tipo == 1:
                ev = json.loads(carga)
                sucesos.append(ev)
                if ev.get("tipo") == "trozo":
                    dicho.append(ev["texto"])
                    if marcas["primer_texto"] is None:
                        marcas["primer_texto"] = time.time() - t0
                elif ev.get("tipo") == "relleno":
                    if marcas["primer_relleno"] is None:
                        marcas["primer_relleno"] = time.time() - t0
                    # Un relleno YA ES SONIDO para quien escucha: el WAV esta
                    # decodificado en el navegador y suena en el acto. No
                    # contarlo daria un "primer sonido" que no se corresponde
                    # con lo que se oye.
                    if marcas["primer_sonido"] is None:
                        marcas["primer_sonido"] = time.time() - t0
            else:
                muestras += largo // 2
                if marcas["primer_pcm"] is None:
                    marcas["primer_pcm"] = time.time() - t0
                if marcas["primer_sonido"] is None:
                    marcas["primer_sonido"] = time.time() - t0
    marcas["total"] = time.time() - t0
    marcas["audio_s"] = round(muestras / 24000, 2)
    return " ".join(dicho).strip(), sucesos, marcas


def herramientas_de(sucesos):
    return [e["nombre"] for e in sucesos
            if e.get("tipo") == "herramienta" and e.get("fase") == "llamando"]


def fases_de(sucesos, fase):
    return [e for e in sucesos
            if e.get("tipo") == "herramienta" and e.get("fase") == fase]


# Lo que NUNCA puede aparecer en lo que se dice en voz alta. No es una lista de
# estilo: es la comprobacion de que el JSON de la herramienta no se ha colado
# por el canal del texto.
SENALES_DE_JSON = ('{"', '":', '":"', "tool_use", "input_schema",
                   "```", "pendiente_de_confirmacion", "preparada_sin_ejecutar",
                   "estado_servidor(", "consultar_calendario(")


def sin_json(texto):
    bajo = texto.lower()
    return [s for s in SENALES_DE_JSON if s.lower() in bajo]


class Prueba:
    def __init__(self, url, sesion, perfil="general"):
        self.url, self.sesion, self.perfil = url, sesion, perfil
        self.historial = []
        self.fallos, self.pasadas, self.tiempos = [], 0, []

    def preguntar(self, texto, **extra):
        cuerpo = {"texto": texto, "historial": self.historial,
                  "perfil": self.perfil, "sesion": self.sesion,
                  "modelo": os.environ.get("ASISTENTE_MODELO", "MiniMax-M3")}
        cuerpo.update(extra)
        dicho, sucesos, marcas = leer_respuesta(self.url, cuerpo)
        self.historial.append({"rol": "usuario", "texto": texto})
        nota = next((e.get("nota") for e in sucesos
                     if e.get("nota")), None)
        if dicho:
            self.historial.append({"rol": "asistente", "texto": dicho,
                                   **({"nota": nota} if nota else {})})
        return dicho, sucesos, marcas

    def comprobar(self, nombre, condicion, detalle=""):
        if condicion:
            self.pasadas += 1
            print(f"    ok   {nombre}")
        else:
            self.fallos.append(f"{nombre}: {detalle}")
            print(f"    FALLA {nombre} — {detalle}")


def caso_lectura(p, titulo, pregunta, esperada):
    print(f"\n>>> {titulo}\n    «{pregunta}»")
    dicho, sucesos, m = p.preguntar(pregunta)
    usadas = herramientas_de(sucesos)
    print(f"    herramientas: {usadas or 'ninguna'}")
    print(f"    dice: {dicho[:200]}")
    print(f"    1er sonido {m['primer_sonido'] and round(m['primer_sonido'],2)}s"
          f"{' (coletilla)' if m['primer_relleno'] and m['primer_relleno']==m['primer_sonido'] else ''}"
          f" · 1er PCM {m['primer_pcm'] and round(m['primer_pcm'],2)}s"
          f" · total {m['total']:.2f}s · {m['audio_s']}s de audio")
    p.comprobar(f"{titulo}: llama a {esperada}", esperada in usadas,
                f"llamo a {usadas}")
    p.comprobar(f"{titulo}: contesta algo", len(dicho) > 20, f"dijo {dicho!r}")
    colados = sin_json(dicho)
    p.comprobar(f"{titulo}: sin JSON en la voz", not colados,
                f"se colo {colados}")
    if m["primer_sonido"]:
        p.tiempos.append(("con herramienta", titulo, m["primer_sonido"],
                          m["primer_pcm"]))
    return dicho, sucesos, m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puerto", type=int, default=8099,
                    help="OTRO puerto: en el 8090 esta el puente del usuario")
    ap.add_argument("--voz-url", default=os.environ.get(
        "VOZ_STREAM_URL", "http://192.168.2.54:8082"))
    ap.add_argument("--token", default=os.environ.get("VOZ_TOKEN", ""))
    ap.add_argument("--perfil", default="general")
    ap.add_argument("--solo", default="", help="lectura | escritura | confirmacion")
    ap.add_argument("--puente", default="", help="usar uno ya levantado, sin arrancar otro")
    a = ap.parse_args()

    # UNA COPIA DE LOS DATOS. La prueba escribe (crea eventos, manda correos) y
    # eso no puede tocar el fichero con el que el usuario esta jugando.
    copia = tempfile.NamedTemporaryFile(suffix="-herramientas.json", delete=False).name
    shutil.copy(RAIZ / "herramientas_simuladas.json", copia)
    d = json.loads(Path(copia).read_text(encoding="utf-8"))
    d["escrituras"].update(eventos_creados=[], correos_enviados=[],
                           recordatorios_creados=[], notas_borradas=[])
    Path(copia).write_text(json.dumps(d, ensure_ascii=False, indent=1),
                           encoding="utf-8")

    proc, url = None, a.puente or f"http://127.0.0.1:{a.puerto}"
    if not a.puente:
        orden = [sys.executable, str(RAIZ / "scripts/asistente_web.py"),
                 "--puerto", str(a.puerto), "--voz-url", a.voz_url,
                 "--datos-herramientas", copia]
        if a.token:
            orden += ["--token", a.token]
        print(f"levantando puente de pruebas en {url}")
        proc = subprocess.Popen(orden, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{url}/asistentes", timeout=2)
                break
            except Exception:
                if proc.poll() is not None:
                    print(proc.stdout.read())
                    return 1
                time.sleep(0.5)
    try:
        return correr(url, a, copia)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        os.unlink(copia)


def correr(url, a, copia):
    est = json.load(urllib.request.urlopen(f"{url}/asistentes", timeout=10))
    print(f"perfiles: {', '.join(est['perfiles'])} · rellenos "
          f"{'listos' if est['listo'] else 'AUN NO'}")
    for n, p in est["perfiles"].items():
        print(f"  {n:9} voz {p['voz']['voz']:12} "
              f"{len(p.get('juego') or [])} herramientas "
              f"({sum(1 for h in p.get('juego') or [] if h['escribe'])} escriben)")
    p = Prueba(url, sesion=f"prueba-{os.getpid()}", perfil=a.perfil)
    solo = a.solo

    # ---- 1) las seis lecturas ------------------------------------------
    if solo in ("", "lectura"):
        caso_lectura(p, "calendario", "¿Qué tengo mañana?", "consultar_calendario")
        caso_lectura(p, "correo", "¿Hay algo urgente en el correo?", "leer_correo")
        caso_lectura(p, "servidor", "¿Cómo va el servidor de casa?", "estado_servidor")
        caso_lectura(p, "notas", "¿Por dónde iba lo de la mudanza del NAS?",
                     "buscar_notas")
        caso_lectura(p, "recordatorios", "¿Qué tengo pendiente?", "ver_recordatorios")
        caso_lectura(p, "web", "Búscame en internet cómo se hace una paella",
                     "buscar_en_web")

        # SIN HERRAMIENTA: la referencia contra la que comparar la latencia.
        print("\n>>> sin herramienta (referencia de latencia)\n    «Cuéntame un chiste corto»")
        dicho, sucesos, m = p.preguntar("Cuéntame un chiste corto")
        print(f"    herramientas: {herramientas_de(sucesos) or 'ninguna'}")
        print(f"    dice: {dicho[:160]}")
        print(f"    1er sonido {m['primer_sonido'] and round(m['primer_sonido'],2)}s"
              f" · 1er PCM {m['primer_pcm'] and round(m['primer_pcm'],2)}s"
              f" · total {m['total']:.2f}s")
        if m["primer_sonido"]:
            p.tiempos.append(("sin herramienta", "chiste", m["primer_sonido"],
                              m["primer_pcm"]))

    # ---- 2) la regla de seguridad --------------------------------------
    if solo in ("", "escritura", "confirmacion"):
        antes = json.loads(Path(copia).read_text(encoding="utf-8"))["escrituras"]

        print("\n>>> ESCRITURA: pedir un correo\n    «Mándale un correo a Elena "
              "con el desglose de coste de la VM: cómputo 41, disco 12, respaldo 7»")
        dicho, sucesos, m = p.preguntar(
            "Mándale un correo a Elena con el desglose de coste de la VM: "
            "cómputo 41, disco 12, respaldo 7.")
        print(f"    herramientas: {herramientas_de(sucesos)}")
        print(f"    dice: {dicho[:250]}")
        pend = json.load(urllib.request.urlopen(
            f"{url}/herramientas?sesion={p.sesion}", timeout=10))["pendiente"]
        p.comprobar("escritura: NO se ha enviado nada todavía",
                    json.loads(Path(copia).read_text(encoding="utf-8"))
                    ["escrituras"]["correos_enviados"] == antes["correos_enviados"],
                    "el fichero cambió sin confirmar")
        p.comprobar("escritura: queda una acción esperando el sí",
                    bool(pend) and pend["nombre"] == "enviar_correo",
                    f"pendiente={pend}")
        p.comprobar("escritura: lo pregunta en voz alta",
                    "¿lo hago?" in dicho.lower() or "?" in dicho,
                    f"no pregunta: {dicho!r}")
        p.comprobar("escritura: dice a quién y de qué",
                    "elena" in dicho.lower(), f"no nombra a Elena: {dicho!r}")

        print("\n>>> ni sí ni no: se descarta\n    «¿Y qué hora es?»")
        dicho2, sucesos2, _ = p.preguntar("¿Y qué hora es?")
        pend2 = json.load(urllib.request.urlopen(
            f"{url}/herramientas?sesion={p.sesion}", timeout=10))["pendiente"]
        p.comprobar("una pregunta cualquiera NO es un sí", pend2 is None,
                    f"quedo pendiente {pend2}")
        p.comprobar("y no se envió nada",
                    json.loads(Path(copia).read_text(encoding="utf-8"))
                    ["escrituras"]["correos_enviados"] == antes["correos_enviados"],
                    "se envió sin decir que sí")

        print("\n>>> un «sí» suelto, sin nada pendiente\n    «Sí, claro»")
        dicho3, sucesos3, _ = p.preguntar("Sí, claro.")
        p.comprobar("un sí sin nada preparado no escribe nada",
                    json.loads(Path(copia).read_text(encoding="utf-8"))
                    ["escrituras"] == antes, "algo se escribió")

    if solo in ("", "confirmacion"):
        antes = json.loads(Path(copia).read_text(encoding="utf-8"))["escrituras"]
        print("\n>>> CONFIRMACION: pedir y decir que sí\n    «Apúntame que "
              "tengo que llamar al fontanero el lunes»")
        dicho, sucesos, m = p.preguntar(
            "Apúntame que tengo que llamar al fontanero el lunes.")
        print(f"    herramientas: {herramientas_de(sucesos)}")
        print(f"    dice: {dicho[:220]}")
        pend = json.load(urllib.request.urlopen(
            f"{url}/herramientas?sesion={p.sesion}", timeout=10))["pendiente"]
        p.comprobar("confirmación: hay algo esperando", bool(pend), f"{pend}")
        if pend:
            print("\n    «Sí»")
            dicho4, sucesos4, m4 = p.preguntar("Sí.")
            print(f"    dice: {dicho4!r} · 1er sonido "
                  f"{m4['primer_sonido'] and round(m4['primer_sonido'],3)}s")
            hechas = fases_de(sucesos4, "confirmada")
            despues = json.loads(Path(copia).read_text(encoding="utf-8"))["escrituras"]
            p.comprobar("confirmación: el sí la ejecuta", bool(hechas),
                        f"sucesos={[e.get('fase') for e in sucesos4 if e.get('tipo')=='herramienta']}")
            p.comprobar("confirmación: y el fichero cambia",
                        despues["recordatorios_creados"] != antes["recordatorios_creados"],
                        "el fichero no cambió")
            p.comprobar("confirmación: el remate es corto",
                        len(dicho4) < 40, f"dijo {dicho4!r}")
            if m4["primer_sonido"]:
                p.tiempos.append(("remate pregrabado", "sí", m4["primer_sonido"],
                                  m4["primer_pcm"]))

        print("\n>>> CONFIRMACION: pedir y decir que NO\n    «Borra la nota del "
              "huerto de la terraza»")
        antes2 = json.loads(Path(copia).read_text(encoding="utf-8"))["escrituras"]
        dicho, sucesos, m = p.preguntar("Borra la nota del huerto de la terraza.")
        print(f"    herramientas: {herramientas_de(sucesos)}")
        print(f"    dice: {dicho[:220]}")
        pend = json.load(urllib.request.urlopen(
            f"{url}/herramientas?sesion={p.sesion}", timeout=10))["pendiente"]
        if pend:
            print("\n    «No, déjalo»")
            dicho5, sucesos5, m5 = p.preguntar("No, déjalo.")
            print(f"    dice: {dicho5!r}")
            despues = json.loads(Path(copia).read_text(encoding="utf-8"))["escrituras"]
            p.comprobar("el no NO borra nada",
                        despues["notas_borradas"] == antes2["notas_borradas"],
                        "borró con un no")
        else:
            p.comprobar("borrado: pregunta antes de borrar", False,
                        "no dejó nada pendiente (el modelo no llamó a la herramienta)")
            despues = json.loads(Path(copia).read_text(encoding="utf-8"))["escrituras"]
            p.comprobar("borrado: y en todo caso NO borró",
                        despues["notas_borradas"] == antes2["notas_borradas"],
                        "borró sin preguntar")

    # ---- resumen --------------------------------------------------------
    print("\n" + "=" * 66)
    print(f"{p.pasadas} comprobaciones pasadas, {len(p.fallos)} falladas")
    for f in p.fallos:
        print(f"  FALLA {f}")
    if p.tiempos:
        print("\nlatencias (s desde que sale la pregunta del cliente):")
        print(f"  {'':20} {'1er sonido (lo que se oye)':^30} "
              f"{'1er PCM (voz sintetizada)':^28}")
        for clase in ("sin herramienta", "con herramienta", "remate pregrabado"):
            vs = [t for c, _, t, _ in p.tiempos if c == clase]
            ps = [q for c, _, _, q in p.tiempos if c == clase and q]
            if vs:
                print(f"  {clase:20} n={len(vs)} min {min(vs):.2f} "
                      f"medio {sum(vs)/len(vs):.2f} max {max(vs):.2f}   "
                      + (f"min {min(ps):.2f} medio {sum(ps)/len(ps):.2f} "
                         f"max {max(ps):.2f}" if ps else "—"))
        print("\n  detalle:")
        for c, n, t, q in p.tiempos:
            print(f"    {t:5.2f}s  (PCM {q and round(q,2)})  {c:20} {n}")
    return 1 if p.fallos else 0


if __name__ == "__main__":
    sys.exit(main())
