#!/usr/bin/env python
"""Narra texto que llega a trozos, sin cortes entre frases.

Pensado para poner voz a un LLM mientras escribe: le vas dando texto segun
sale y esto lo va diciendo, sin silencios raros en medio.

    # narrar algo de una vez
    python scripts/narrador.py "Hola. Tienes tres cosas pendientes hoy."

    # narrar lo que salga por la tuberia, segun vaya saliendo
    mi-llm --stream | python scripts/narrador.py

    # dejarlo en un fichero en vez de reproducirlo
    python scripts/narrador.py --salida respuesta.wav "..."

DE DONDE SALEN LOS HUECOS, Y COMO SE TAPAN
El primer fotograma de CUALQUIER locucion cuesta ~355 ms dentro de generate().
No es sobrecarga de la peticion -- se midio: el deepcopy del prefijo son 1,3 ms
y procesar la entrada 1,6 ms. Y no se puede paralelizar: una sola generacion ya
satura la CPU, y el servicio tiene un candado por eso mismo. (Se intento con un
pool de dos hilos: salio peor, ver el comentario de productor().)

Lo que si funciona es acumular audio antes de empezar a sonar, para que cada
frase nueva arranque mientras aun queda cola de la anterior. Medido con la
maquina CARGADA, que es el caso dificil:

    bufer 0,6 s   1 de 2 pasadas con cortes
    bufer 1,5 s   limpio          <- por defecto
    bufer 2,5 s   limpio, pero 1 s mas de espera para nada

Con la maquina en reposo sobra con 0. El defecto de 1,5 s esta elegido para
que aguante cuando algo mas compite por la CPU.

POR QUE NO BASTA CON PEDIR FRASE A FRASE
El servicio genera a RTF ~0,90: un segundo de audio cuesta 0,9 s de computo.
Sobra un 10 %, pero cada peticion arranca con ~0,25 s de latencia. Encadenando
frases a lo tonto, ese arranque se oye como un silencio entre una y otra.

La solucion es ir por delante: mientras suena la frase N se esta generando la
N+1. Como generar va mas rapido que hablar, el bufer crece y los arranques
quedan tapados. Solo el primero se nota, y es el unico inevitable.

REGLA DE ORO: trocear por FRASES, no por tokens ni por palabras. El modelo
necesita la frase entera para entonar bien -- la prosodia de "no" cambia
segun lo que venga detras. Trocear mas fino da voz robotica.
"""
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
import wave

RITMO = 24_000

# Fin de frase: puntuacion fuerte, admitiendo comillas o parentesis detras, y
# seguida de espacio o del final. Ese "seguida de" evita partir "3.14" o "S.L."
FIN_FRASE = re.compile(r"[.!?\u2026]+[\"'\u201d\u2019)\]]*(?=\s|$)")
# Fin de clausula: sirve para empezar a hablar antes sin destrozar la prosodia.
FIN_CLAUSULA = re.compile(r"[,;:\u2014\u2013][\"'\u201d\u2019)\]]*(?=\s)")

# Debajo de ~2 s de audio el RTF se dispara (de 0,8 a 1,7): el coste fijo de
# cada peticion domina. Por eso se JUNTAN frases cortas hasta llegar aqui.
MINIMO_CARACTERES = 60
# La primera se acepta mas corta: es lo unico que separa al usuario del
# silencio. Sale cara, pero solo se paga una vez.
MINIMO_PRIMERA = 25
# Valvula de seguridad: si un LLM suelta un parrafo entero sin puntuacion
# fuerte, no se puede esperar indefinidamente. Pasado esto se corta por
# clausula, y si tampoco hay, por el ultimo espacio -- NUNCA a mitad de
# palabra.
MAXIMO_SIN_CORTE = 320


def _punto_de_corte(texto: str, minimo: int, con_clausula: bool):
    """Indice donde cortar respetando el lenguaje, o None si aun no toca.

    El orden de preferencia importa y es lo que arregla el fallo que tenia
    esto antes: cortaba al llegar a N caracteres, sin mirar si era una frase.
    Producia trozos como 'El despliegue se' y 'realiza mediante ... depend',
    partiendo palabras por la mitad. El sintetizador recibia texto sin sentido
    y no podia entonar.

      1. fin de FRASE  -> lo ideal: el modelo ve la frase entera y entona bien
      2. fin de CLAUSULA -> aceptable, solo para arrancar antes o si la frase
                            se hace larguisima
      3. ultimo ESPACIO -> ultimo recurso, y aun asi respeta las palabras
    """
    for m in FIN_FRASE.finditer(texto):
        if len(texto[:m.end()].strip()) >= minimo:
            return m.end()
    if con_clausula or len(texto) >= MAXIMO_SIN_CORTE:
        for m in FIN_CLAUSULA.finditer(texto):
            if len(texto[:m.end()].strip()) >= minimo:
                return m.end()
    if len(texto) >= MAXIMO_SIN_CORTE:
        hueco = texto.rfind(" ", minimo)
        if hueco > 0:
            return hueco
    return None


def trocear(texto: str, forzar_final: bool = False, primera: bool = False,
            minimo_primera: int = MINIMO_PRIMERA):
    """Devuelve (trozos_listos, resto_pendiente).

    Solo saca trozos que terminan donde el lenguaje permite. Lo que no llega a
    un limite valido se queda en el resto, esperando mas texto -- salvo
    forzar_final, que es cuando el LLM ya termino y no va a llegar mas.
    """
    trozos, resto = [], texto
    while True:
        minimo = minimo_primera if (primera and not trozos) else MINIMO_CARACTERES
        corte = _punto_de_corte(resto, minimo, con_clausula=(primera and not trozos))
        if corte is None:
            break
        trozo = resto[:corte].strip()
        resto = resto[corte:].lstrip()
        if trozo:
            trozos.append(trozo)
    if forzar_final and resto.strip():
        trozos.append(resto.strip())
        resto = ""
    return trozos, resto


def sintetizar(frase, url, token, voz, cfg, ajustes=None):
    """Pide una frase y devuelve sus trozos de PCM segun llegan.

    `ajustes` es un dict con lo opcional (pasos, velocidad, semilla, sesion);
    lo que no venga, lo decide el servicio.
    """
    cuerpo = {"texto": frase, "voz": voz, "cfg_scale": cfg}
    cuerpo.update({k: v for k, v in (ajustes or {}).items() if v is not None})
    pet = urllib.request.Request(
        f"{url}/tts/stream",
        method="POST",
        data=json.dumps(cuerpo).encode(),
        headers={"content-type": "application/json",
                 **({"authorization": f"Bearer {token}"} if token else {})},
    )
    r = urllib.request.urlopen(pet, timeout=600)
    primero = True
    while True:
        trozo = r.read(8192)
        if not trozo:
            break
        if primero:                 # fuera la cabecera WAV de cada respuesta
            trozo, primero = trozo[44:], False
        if trozo:
            yield trozo


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("texto", nargs="*", help="texto a narrar; si falta, lee de la entrada")
    ap.add_argument("--url", default=os.environ.get("VOZ_STREAM_URL", "http://127.0.0.1:8082"))
    ap.add_argument("--token", default=os.environ.get("VOZ_TOKEN", ""))
    ap.add_argument("--voz", default=os.environ.get("VIBEVOICE_VOZ", "sp-Spk1_man"))
    ap.add_argument("--cfg", type=float, default=3.0,
                    help="guia CFG; 3.0 medido como el mas fiel")
    ap.add_argument("--salida", help="escribir a un WAV en vez de reproducir")
    ap.add_argument("--bufer", type=float, default=1.5,
                    help="segundos de audio a acumular antes de empezar a sonar "
                         "(0 = empezar cuanto antes, a riesgo de huecos)")
    ap.add_argument("--silencio", action="store_true", help="sin informe de tiempos")
    a = ap.parse_args()

    audio = queue.Queue(maxsize=64)
    informe = {"frases": 0, "esperas": 0, "espera_total": 0.0, "muestras": 0,
               "inicio_audio": 0.0}
    t_inicio = time.time()

    def productor():
        """Trocea y sintetiza, en orden.

        NO se intenta solapar frases: el servicio tiene un candado y solo hace
        UNA generacion a la vez, asi que lanzar la siguiente antes no adelanta
        nada -- se queda encolada en el servidor. Se probo con un pool de dos
        hilos y salio peor: para tener algo con que tapar la frontera hay que
        acumular la frase entera antes de emitirla, y eso retrasa el primer
        sonido de 0,2 s a 20 s. El remedio era peor.

        Lo que SI tapa las fronteras es el margen: generar va a RTF ~0,8, o sea
        que cada frase deja ~20 % de bufer acumulado. Con frases de mas de dos
        segundos eso cubre de sobra el arranque de la siguiente.
        """
        pendiente = ""
        fuente = [" ".join(a.texto)] if a.texto else sys.stdin
        for pedazo in fuente:
            pendiente += pedazo
            frases, pendiente = trocear(pendiente, primera=informe["frases"] == 0)
            for f in frases:
                informe["frases"] += 1
                for t in sintetizar(f, a.url, a.token, a.voz, a.cfg):
                    audio.put(t)
        frases, _ = trocear(pendiente, forzar_final=True,
                            primera=informe["frases"] == 0)
        for f in frases:
            informe["frases"] += 1
            for t in sintetizar(f, a.url, a.token, a.voz, a.cfg):
                audio.put(t)
        audio.put(None)

    hilo = threading.Thread(target=productor, daemon=True)
    hilo.start()

    if a.salida:
        w = wave.open(a.salida, "wb")
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RITMO)
        escribir, cerrar = w.writeframes, w.close
    else:
        # ffplay reproduce PCM crudo desde la tuberia sin bufer propio, que es
        # justo lo que hace falta para no anadir latencia encima.
        pr = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
             "-f", "s16le", "-ar", str(RITMO), "-ac", "1", "-"],
            stdin=subprocess.PIPE)
        escribir = pr.stdin.write
        def cerrar():
            pr.stdin.close(); pr.wait()

    # Simulacion honesta del reloj de reproduccion. Un corte es que un trozo
    # llegue cuando el altavoz ya se quedo sin nada que sonar. Se mide el
    # DEFICIT, no el retraso de cada trozo por separado: si un trozo llega
    # tarde y los siguientes vienen detras, el silencio se oye UNA vez, no una
    # por trozo. (Sumar por trozo daba mas silencio que audio total, que era la
    # pista de que la cuenta estaba mal.)
    # El primer fotograma de CUALQUIER locucion cuesta ~355 ms dentro de
    # generate() -- medido, y no es sobrecarga de la peticion: el deepcopy del
    # prefijo son 1,3 ms y procesar la entrada 1,6 ms. No se puede paralelizar
    # porque una sola generacion ya satura la CPU.
    #
    # Lo que SI se puede es taparlo: acumulando un poco de audio antes de
    # empezar a sonar, cada frase nueva arranca mientras aun queda cola de la
    # anterior. Se paga una vez, al principio.
    reloj = None
    espera_inicial = a.bufer
    while True:
        trozo = audio.get()
        if trozo is None:
            break
        llegada = time.time() - t_inicio
        if reloj is None:
            informe["inicio_audio"] = llegada + espera_inicial
            reloj = informe["inicio_audio"]
        elif llegada > reloj + 0.02:      # el bufer se vacio: eso es un corte
            informe["esperas"] += 1
            informe["espera_total"] += llegada - reloj
            reloj = llegada
        reloj += (len(trozo) // 2) / RITMO
        informe["muestras"] += len(trozo) // 2
        escribir(trozo)
    cerrar()

    if not a.silencio:
        seg = informe["muestras"] / RITMO
        total = time.time() - t_inicio
        print(f"\n{informe['frases']} frases · {seg:.1f}s de audio en {total:.1f}s"
              f" · RTF {total/seg:.2f}" if seg else "\nsin audio", file=sys.stderr)
        if informe["esperas"]:
            print(f"CORTES: {informe['esperas']} trozos llegaron tarde,"
                  f" {informe['espera_total']:.2f}s de silencio acumulado",
                  file=sys.stderr)
        else:
            print(f"SIN CORTES · primer sonido a los {informe['inicio_audio']:.2f}s",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
