"""Genera el corpus del banco A/B contra un voz-stream: voces x frases x semillas.

Solo biblioteca estandar (corre en la VM). Un WAV por clip, con la cabecera bien
escrita (el flujo de /tts/stream la manda con tamanos de relleno), y un CSV con
los tiempos de cada peticion.

    python banco_ab_generar.py --url http://127.0.0.1:8082 --etiqueta nuevo --salida /root/banco-ab/nuevo
"""
import argparse
import csv
import json
import os
import time
import urllib.request
import wave

VOCES = ["andres", "isis", "juan", "santiago", "sp-Spk1_man", "sp-Spk3_man", "sp-Spk0_woman"]
SEMILLAS = [101, 17]

# Clave corta -> texto. Variado a proposito: numeros en letra, preguntas,
# exclamaciones, pausas marcadas, frases cortas del asistente, largas, y un
# parrafo de ~25 s para ver si algo deriva con la longitud.
FRASES = {
    "backup": "El backup de anoche terminó sin errores.",
    "servicios": "Los tres servicios responden con normalidad.",
    "memoria": "El uso de memoria bajó un veinticuatro por ciento.",
    "pendientes": "Tienes tres cosas pendientes para hoy.",
    "reunion": "La reunión de las once se confirmó esta mañana.",
    "incidencias": "No hay incidencias que reportar en las últimas horas.",
    "expresivo": "¿En serio? ¡No me lo puedo creer! Eso sí que no me lo esperaba para nada, de verdad.",
    "pregunta": "¿Quieres que te lo mande ahora o prefieres que espere a mañana por la tarde?",
    "pedido": "El pedido llega el martes catorce de octubre, entre las nueve y las doce de la mañana.",
    "temperatura": "La temperatura del servidor principal se mantiene estable desde la madrugada, y el disco de respaldo ya terminó de sincronizarse.",
    "certificados": "Mañana a primera hora hay que revisar los certificados, porque dos de ellos caducan antes del fin de semana.",
    "corto": "Vale, ahora mismo lo miro.",
    "saludo": "Hola, buenos días. ¿Qué tal has dormido?",
    "pasos": "Primero apaga el router, espera treinta segundos, y luego vuelve a encenderlo con calma.",
    "disculpa": "Lo siento, no he podido encontrar ese correo; ¿puedes darme algún detalle más?",
    "alegria": "¡Qué bien! Por fin ha salido todo como queríamos.",
    "parrafo": ("Te cuento cómo ha ido la semana. El lunes terminamos la migración de la base de datos, "
                "el miércoles hubo un corte de luz que duró casi una hora, y el viernes, por fin, "
                "conseguimos que las copias de seguridad se hicieran solas todas las noches sin que nadie "
                "tuviera que tocarlas."),
}


def pedir(url, token, texto, voz, semilla, cfg):
    cuerpo = {"texto": texto, "voz": voz, "semilla": semilla}
    if cfg is not None:
        cuerpo["cfg_scale"] = cfg
    pet = urllib.request.Request(
        f"{url}/tts/stream", method="POST", data=json.dumps(cuerpo).encode(),
        headers={"content-type": "application/json",
                 **({"authorization": f"Bearer {token}"} if token else {})})
    t0 = time.perf_counter()
    r = urllib.request.urlopen(pet, timeout=900)
    trozos, primero = [], None
    while True:
        b = r.read1(65536)
        if not b:
            break
        trozos.append(b)
        if primero is None and sum(len(t) for t in trozos) > 44:
            primero = time.perf_counter() - t0
    total = time.perf_counter() - t0
    ritmo = int(r.headers.get("X-Ritmo-Hz", "24000"))
    pcm = b"".join(trozos)[44:]
    return pcm[: len(pcm) // 2 * 2], ritmo, total, primero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--token", default=os.environ.get("VOZ_TOKEN", ""))
    ap.add_argument("--etiqueta", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--voces", nargs="+", default=VOCES)
    ap.add_argument("--semillas", type=int, nargs="+", default=SEMILLAS)
    ap.add_argument("--frases", nargs="+", default=list(FRASES))
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    with open(os.path.join(a.salida, "frases.json"), "w") as f:
        json.dump(FRASES, f, ensure_ascii=False, indent=1)
    for voz in a.voces:                       # calentar cada voz fuera de la cuenta
        pedir(a.url, a.token, "Calentando.", voz, 1, a.cfg)
    ruta_csv = os.path.join(a.salida, "clips.csv")
    campos = ["fichero", "voz", "frase", "semilla", "cfg", "audio_s", "reloj_s", "rtf", "primero_s"]
    n = len(a.voces) * len(a.frases) * len(a.semillas)
    hecho = 0
    t_ini = time.time()
    with open(ruta_csv, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=campos)
        w.writeheader()
        for voz in a.voces:
            for semilla in a.semillas:
                for clave in a.frases:
                    pcm, ritmo, total, primero = pedir(a.url, a.token, FRASES[clave], voz, semilla, a.cfg)
                    fichero = f"{voz}__{clave}__s{semilla}.wav"
                    with wave.open(os.path.join(a.salida, fichero), "wb") as ww:
                        ww.setnchannels(1)
                        ww.setsampwidth(2)
                        ww.setframerate(ritmo)
                        ww.writeframes(pcm)
                    seg = len(pcm) / 2 / ritmo
                    w.writerow({"fichero": fichero, "voz": voz, "frase": clave, "semilla": semilla,
                                "cfg": a.cfg, "audio_s": round(seg, 3), "reloj_s": round(total, 3),
                                "rtf": round(total / seg, 4) if seg else "", "primero_s": round(primero or 0, 3)})
                    fcsv.flush()
                    hecho += 1
                    if hecho % 17 == 0 or hecho == n:
                        print(f"[{a.etiqueta}] {hecho}/{n} · {voz} s{semilla} · {(time.time() - t_ini) / 60:.1f} min", flush=True)
    print(f"[{a.etiqueta}] listo: {n} clips en {(time.time() - t_ini) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
