#!/usr/bin/env python
"""Prueba el websocket de sesiones contra la via HTTP, que es la referencia.

    pkgs/vibevoice/.venv/bin/python scripts/ws_fidelidad.py \
        --url http://127.0.0.1:8082 --token "$VOZ_TOKEN"

Cuatro cosas, y las cuatro con numeros:

  1. FIDELIDAD. El PCM que baja por el websocket tiene que ser IDENTICO -- md5,
     no "parecido" -- al de la sesion HTTP con las mismas frases, la misma voz
     y la misma semilla, y tambien al de /tts/stream con las frases unidas por
     ESPACIO. Se pide respiro=False a proposito: el respiro (el aire de cada
     final de frase; bloque RESPIRO de voz_stream.py) no toca el texto pero si
     el audio -- alarga la pausa y recorta el sobrante --, asi que con el
     puesto el md5 ya no puede coincidir. Sin respiro, la sesion cose los
     trozos con espacio y pone un unico "\\n" al final de la locucion (igual
     que hace el procesador con una peticion suelta). Comparar contra la
     referencia equivocada es el error clasico aqui. El respiro tiene su
     propia prueba (punto 5).

     El websocket ademas se alimenta en el CASO DIFICIL: cada frase se manda
     solo cuando llega el evento esperando=true, es decir cuando el modelo ya
     se quedo parado sin texto por delante. Si la pausa cambiara un solo
     calculo, el md5 se rompia.

  2. EVENTOS. Que lleguen, y en el orden que promete el protocolo.

  3. CONCURRENCIA. Dos sesiones A LA VEZ con la misma semilla tienen que dar el
     mismo md5, y ademas el mismo que una sesion a solas. Es la parte que el
     candado del modelo no cubre por si sola -- cada sesion lo suelta en sus
     pausas y la otra se cuela en mitad de su locucion --, y lo que la hace
     cumplirse es que cada sesion se lleve su RNG puesto (SesionViva._pausar).
     Se comprueba tambien que de verdad coexistieron: si no, la prueba pasaria
     por no haber probado nada.

  4. LIMPIEZA. Al cortar el websocket a mitad de la locucion, la sesion tiene
     que desaparecer: GET /tts/sesion/{id} da 404 y el candado del modelo queda
     libre para la peticion siguiente.

  6. PAUSA. La concurrencia de arriba prueba dos sesiones IGUALES. Esta
     prueba mete una sesion DISTINTA -- otros pasos, otra semilla -- en la
     pausa de la primera, y ademas en la pausa peor: la sesion A abre con una
     frase corta y se queda parada ANTES de generar su primer fotograma,
     esperando la ventana de adelanto. Cuando A reanuda, el proceso esta como
     lo dejo la intrusa: sus pasos de difusion, su contador de la rampa de
     arranque, su remate. Si la sesion no se lleva y trae TODO su estado
     (foto_generacion/reponer_generacion en voz_stream.py), A sale con los
     pasos de B y sin su rampa, y B remata con los pasos de A. Los dos md5
     tienen que ser los de cada sesion a solas. La variante pausa-stream usa
     /tts/stream de intrusa (pasos+4, neg_cada 2).

  5. RESPIRO. Con respiro (el defecto de las sesiones) el audio tiene que
     respetar el contrato de la pausa: ninguna racha de fotogramas de 133 ms
     por debajo del umbral puede pasar de fotogramas+alarga+2 (los dos de
     margen son los bordes, que quedan justo bajo el umbral de deteccion pero
     encima del de recorte), y tiene que haber pausas de final de frase
     (rachas >= 2). Ademas el HABLA tiene que seguir siendo la misma que sin
     respiro: quitando de la version con respiro los fotogramas que son suelo
     de sala, lo que queda tiene que coincidir muestra a muestra con lo mismo
     hecho sobre la version sin respiro -- es lo que prueba que el aire se
     INSERTA y no se sintetiza (bloque RESPIRO de voz_stream.py). Los
     parametros exactos se leen de /health, no se suponen.

El marco binario es el de scripts/asistente_web.py: [tipo:1][longitud:4 BE]
[carga], tipo 0 = PCM y tipo 1 = evento JSON.
"""
import argparse
import asyncio
import hashlib
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import websockets

RITMO = 24_000
FRASES = [
    "El tren llega a las siete de la tarde.",
    "Manana por la mañana vamos al parque.",
    "No olvides comprar pan y leche.",
]


# --------------------------------------------------------------------- HTTP
def pedir(url, token, cuerpo=None, metodo=None, tiempo=600):
    pet = urllib.request.Request(
        url, method=metodo or ("POST" if cuerpo is not None else "GET"),
        data=json.dumps(cuerpo).encode() if cuerpo is not None else None,
        headers={"content-type": "application/json",
                 **({"authorization": f"Bearer {token}"} if token else {})})
    return urllib.request.urlopen(pet, timeout=tiempo)


def http_stream(url, token, texto, voz, cfg, semilla, pasos, neg_cada=None):
    """/tts/stream de una vez. Se le quitan los 44 bytes de cabecera WAV."""
    cuerpo = {"texto": texto, "voz": voz, "cfg_scale": cfg, "semilla": semilla}
    if pasos is not None:
        cuerpo["pasos"] = pasos
    if neg_cada is not None:
        cuerpo["neg_cada"] = neg_cada
    return pedir(f"{url}/tts/stream", token, cuerpo).read()[44:]


def http_sesion(url, token, nombre, frases, voz, cfg, semilla, pasos,
                respiro=False):
    """Sesion HTTP: se meten todas las frases y se escucha el WAV continuo."""
    import threading
    base = {"voz": voz, "cfg_scale": cfg, "semilla": semilla,
            "respiro": respiro}
    if pasos is not None:
        base["pasos"] = pasos
    pcm = bytearray()

    json.load(pedir(f"{url}/tts/sesion/{nombre}", token,
                    {**base, "texto": frases[0]}))

    def leer():
        r = pedir(f"{url}/tts/sesion/{nombre}/audio", token)
        r.read(44)
        while True:
            t = r.read(8192)
            if not t:
                break
            pcm.extend(t)

    hilo = threading.Thread(target=leer, daemon=True)
    hilo.start()
    for f in frases[1:]:
        json.load(pedir(f"{url}/tts/sesion/{nombre}", token,
                        {**base, "texto": f}))
    pedir(f"{url}/tts/sesion/{nombre}/fin", token, {})
    hilo.join(timeout=600)
    return bytes(pcm)


def estado_sesion(url, token, nombre):
    """(codigo, cuerpo). 404 significa que la sesion ya no existe."""
    try:
        r = pedir(f"{url}/tts/sesion/{nombre}", token)
        return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


# ---------------------------------------------------------------- websocket
def desmarcar(buf, salida_pcm, eventos):
    """Saca marcos completos de `buf`. Devuelve lo que sobra.

    Un mensaje de websocket ya llega entero, asi que en la practica cada uno
    trae un marco justo; se acumula igualmente para no depender de eso.
    """
    i = 0
    while len(buf) - i >= 5:
        tipo, largo = struct.unpack(">BI", buf[i:i + 5])
        if len(buf) - i - 5 < largo:
            break
        carga = buf[i + 5:i + 5 + largo]
        i += 5 + largo
        if tipo == 0:
            salida_pcm.extend(carga)
        elif tipo == 1:
            eventos.append(json.loads(carga.decode()))
        else:
            raise AssertionError(f"tipo de marco desconocido: {tipo}")
    return buf[i:]


async def ws_sesion(url, token, frases, voz, cfg, semilla, pasos,
                    por_cabecera=True, cortar_en=None, traza=None,
                    antes_de_cortar=None, respiro=False,
                    antes_de_seguir=None, al_primer_audio=None):
    """Habla por el websocket y devuelve (pcm, eventos, nombre).

    cortar_en: si viene, se cierra el socket a lo bruto en cuanto hayan bajado
    esos bytes de PCM -- el caso "el cliente se fue a mitad".
    antes_de_cortar: se llama con el nombre de la sesion justo antes de cortar,
    que es el unico momento en que se la puede ver viva desde fuera.
    antes_de_seguir: corrutina que se espera ANTES de mandar cada frase
    posterior a la primera, o sea con el modelo parado sin texto por delante.
    Es el hueco en el que otra generate() se puede colar (prueba `pausa`).
    al_primer_audio: se llama una vez, con el primer PCM que baja.
    """
    ws_url = url.replace("http://", "ws://").replace("https://", "wss://")
    destino = f"{ws_url}/tts/sesion/ws"
    cabeceras = {}
    if token and por_cabecera:
        cabeceras["authorization"] = f"Bearer {token}"
    elif token:
        # Repliegue del navegador: no puede poner cabeceras en new WebSocket().
        # Deja el token en los registros del servidor, ver _autorizado_ws().
        destino += "?" + urllib.parse.urlencode({"token": token})

    pcm, eventos, buf = bytearray(), [], b""
    nombre = None
    # velocidad: 1.0 explicita a proposito: es el unico valor que admite el
    # websocket y conviene que la prueba pase por esa comprobacion.
    abrir = {"accion": "abrir", "voz": voz, "cfg_scale": cfg,
             "velocidad": 1.0, "semilla": semilla, "respiro": respiro}
    if pasos is not None:
        abrir["pasos"] = pasos

    async with websockets.connect(destino, additional_headers=cabeceras,
                                  max_size=None, ping_interval=None) as ws:
        await ws.send(json.dumps(abrir))
        pendientes = list(frases)
        t0 = time.perf_counter()

        async def bombear_texto():
            """Manda la frase siguiente solo cuando el modelo se queda parado.

            Es el caso dificil: si la pausa alterara algun calculo, el md5
            dejaria de coincidir con el de mandarlo todo de golpe.
            """
            nonlocal pendientes
            await ws.send(json.dumps({"accion": "texto",
                                      "texto": pendientes.pop(0)}))

        await bombear_texto()
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=120)
            except asyncio.TimeoutError:
                raise AssertionError("120 s sin recibir nada por el websocket")
            except websockets.exceptions.ConnectionClosed:
                break
            if isinstance(msg, str):
                raise AssertionError(f"llego texto y se esperaba binario: {msg[:80]}")
            antes = len(eventos)
            habia_pcm = len(pcm)
            buf = desmarcar(buf + msg, pcm, eventos)
            if al_primer_audio is not None and not habia_pcm and pcm:
                al_primer_audio()
            for ev in eventos[antes:]:
                if traza is not None:
                    traza.append((round(time.perf_counter() - t0, 2), ev))
                if ev["tipo"] == "abierta":
                    nombre = ev["sesion"]
                if ev["tipo"] == "error":
                    raise AssertionError(f"error del servidor: {ev['texto']}")
                if ev["tipo"] == "esperando" and ev["esperando"]:
                    if pendientes:
                        if antes_de_seguir is not None:
                            await antes_de_seguir()
                        await bombear_texto()
                    else:
                        await ws.send(json.dumps({"accion": "fin"}))
                if ev["tipo"] == "hecho":
                    return bytes(pcm), eventos, nombre
            if cortar_en is not None and len(pcm) >= cortar_en:
                if antes_de_cortar is not None:
                    antes_de_cortar(nombre)
                # A lo bruto y a proposito: sin "fin", sin close educado. Es lo
                # que ve el servidor cuando al navegador le cierran la pestaña.
                await ws.close(code=1001)
                return bytes(pcm), eventos, nombre
    return bytes(pcm), eventos, nombre


# --------------------------------------------------------------------- main
def md5(b):
    return hashlib.md5(b).hexdigest()


def dur(b):
    return len(b) / 2 / RITMO


FOT = 3200          # muestras de un fotograma acustico (133 ms a 24 kHz)


def _fotogramas(pcm):
    """El PCM s16le partido en fotogramas de 133 ms, como arrays de enteros."""
    import array
    m = array.array("h")
    m.frombytes(pcm[:len(pcm) // 2 * 2])
    if sys.byteorder == "big":
        m.byteswap()
    return [m[i:i + FOT] for i in range(0, len(m) - FOT + 1, FOT)]


def _rms(f):
    return (sum(v * v for v in f) / len(f)) ** 0.5 / 32768.0


def rachas_calladas(pcm, umbral):
    """Rachas de fotogramas seguidos por debajo del umbral."""
    out, r = [], 0
    for f in _fotogramas(pcm):
        if _rms(f) < umbral:
            r += 1
        else:
            if r:
                out.append(r)
            r = 0
    if r:
        out.append(r)
    return out


def aplicar_respiro(pcm, fot, alarga, tope, umbral, pico, prerrollo, pie):
    """El respiro del servidor, reimplementado aqui (ColaAudioSesion.put).

    Es DELIBERADAMENTE una segunda implementacion y no una importacion: lo que
    se quiere comprobar es que el servidor hace lo que su documentacion dice,
    y compartir el codigo no probaria nada.
    """
    import array
    salida = array.array("h")
    umbral_pico = int(pico * 32768)
    seguidos = 0
    sonado = False
    suelo = None      # ultimo fotograma callado ENTERO, material del aire
    for f in _fotogramas(pcm):
        if _rms(f) >= umbral:
            seguidos = 0
            sonado = True
            salida.extend(f)
            continue
        seguidos += 1
        if seguidos > tope:
            # Pasado el tope se recorta, pero solo la CABEZA callada: si el
            # fotograma lleva dentro el ataque de la palabra siguiente se emite
            # desde justo antes de el.
            primera = next((k for k, v in enumerate(f)
                            if abs(v) >= umbral_pico), None)
            if primera is not None:
                salida.extend(f[max(0, primera - prerrollo):])
            continue
        if seguidos != fot or not sonado or alarga <= 0:
            salida.extend(f)
            if max(abs(v) for v in f) < umbral_pico:
                suelo = f
            continue
        # EL AIRE, y va DENTRO del fotograma. Este es el de la costura: sus
        # primeros 110 ms son suelo de sala y los ultimos 20 el arranque de la
        # palabra siguiente. Se parte por el PIE de ese ataque y se emite
        # cabeza, aire, ataque -- el aire hecho SOLO con la cabeza, que es
        # suelo de verdad, en espejo alternado y recortado a `alarga`
        # fotogramas exactos.
        primera = next((k for k, v in enumerate(f)
                        if abs(v) >= umbral_pico), None)
        corte = len(f) if primera is None else max(0, primera - pie)
        cabeza, ataque = f[:corte], f[corte:]
        fuente = cabeza if len(cabeza) >= len(f) // 4 else (suelo or f)
        salida.extend(cabeza)
        puestas, k = 0, 0
        while puestas < alarga * len(f):
            pieza = fuente[::-1] if k % 2 == 0 else fuente
            pieza = pieza[:alarga * len(f) - puestas]
            salida.extend(pieza)
            puestas += len(pieza)
            k += 1
        salida.extend(ataque)
    if sys.byteorder == "big":
        salida.byteswap()
    return salida.tobytes()


def comprobar_orden(eventos):
    """Los eventos tienen que llegar en el orden que promete el protocolo."""
    tipos = [e["tipo"] for e in eventos]
    fallos = []
    if not tipos or tipos[0] != "abierta":
        fallos.append(f"el primer evento no es 'abierta' sino {tipos[:1]}")
    if tipos[-1] != "hecho":
        fallos.append(f"el ultimo evento no es 'hecho' sino {tipos[-1:]}")
    for obligatorio in ("abierta", "texto", "sonando", "esperando",
                        "fin_texto", "hecho"):
        if obligatorio not in tipos:
            fallos.append(f"falta el evento '{obligatorio}'")
    if "sonando" in tipos and "abierta" in tipos:
        if tipos.index("sonando") < tipos.index("texto"):
            fallos.append("'sonando' llego antes que el primer 'texto'")
    if "fin_texto" in tipos and tipos.index("fin_texto") > tipos.index("hecho"):
        fallos.append("'fin_texto' despues de 'hecho'")
    # esperando: tiene que alternar true/false y empezar en true
    esperas = [e["esperando"] for e in eventos if e["tipo"] == "esperando"]
    if esperas and esperas[0] is not True:
        fallos.append("el primer 'esperando' no es true")
    for a, b in zip(esperas, esperas[1:]):
        if a == b:
            fallos.append("dos 'esperando' seguidos con el mismo valor")
            break
    return fallos


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("VOZ_STREAM_URL",
                                                    "http://127.0.0.1:8082"))
    ap.add_argument("--token", default=os.environ.get("VOZ_TOKEN", ""))
    ap.add_argument("--voz", default="sp-Spk3_man")
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--semilla", type=int, default=11)
    ap.add_argument("--pasos", type=int, default=6)
    ap.add_argument("--pruebas",
                    default="fidelidad,eventos,concurrencia,pausa,pausa-stream,"
                            "respiro,corte,errores,auth")
    a = ap.parse_args()
    pruebas = a.pruebas.split(",")
    fallos = []

    # ------------------------------------------------------- 1) fidelidad --
    if "fidelidad" in pruebas or "eventos" in pruebas:
        traza = []
        t = time.time()
        pcm_ws, eventos, nombre = asyncio.run(ws_sesion(
            a.url, a.token, FRASES, a.voz, a.cfg, a.semilla, a.pasos,
            traza=traza))
        print(f"[ws]    {dur(pcm_ws):5.2f} s de audio en {time.time()-t:5.1f} s "
              f"· {len(pcm_ws)} bytes · md5 {md5(pcm_ws)} · sesion {nombre}")

    if "fidelidad" in pruebas:
        t = time.time()
        pcm_ses = http_sesion(a.url, a.token, f"ref-{a.semilla}", FRASES,
                              a.voz, a.cfg, a.semilla, a.pasos)
        print(f"[http sesion] {dur(pcm_ses):5.2f} s de audio en "
              f"{time.time()-t:5.1f} s · {len(pcm_ses)} bytes · md5 {md5(pcm_ses)}")

        t = time.time()
        # Con ESPACIOS: la sesion cose los trozos con espacio y solo pone el
        # "\n" del final de la locucion (SesionViva.alimentar/cerrar), que
        # /tts/stream anade igual por su cuenta (text.strip() + "\n").
        pcm_str = http_stream(a.url, a.token, " ".join(FRASES), a.voz, a.cfg,
                              a.semilla, a.pasos)
        print(f"[http stream] {dur(pcm_str):5.2f} s de audio en "
              f"{time.time()-t:5.1f} s · {len(pcm_str)} bytes · md5 {md5(pcm_str)}")

        for etiqueta, otro in (("sesion HTTP", pcm_ses), ("/tts/stream", pcm_str)):
            if md5(pcm_ws) == md5(otro):
                print(f"  OK  websocket == {etiqueta}: identico bit a bit")
            else:
                n = min(len(pcm_ws), len(otro))
                iguales = next((i for i in range(n)
                                if pcm_ws[i] != otro[i]), n)
                fallos.append(f"websocket != {etiqueta}")
                print(f"  FALLO websocket != {etiqueta}: {len(pcm_ws)} vs "
                      f"{len(otro)} bytes, primer byte distinto en {iguales}")

    # --------------------------------------------------------- 2) eventos --
    if "eventos" in pruebas:
        print("\n[eventos]")
        for s, ev in traza:
            resto = {k: v for k, v in ev.items() if k not in ("tipo", "s")}
            print(f"  {s:6.2f}s  {ev['tipo']:<10s} {resto if resto else ''}")
        malos = comprobar_orden(eventos)
        if malos:
            fallos.extend(malos)
            for m in malos:
                print(f"  FALLO {m}")
        else:
            print(f"  OK  {len(eventos)} eventos en el orden esperado")

    # ------------------------------------------ 3) dos sesiones a la vez --
    # LA GARANTIA QUE SE PRUEBA AQUI es "misma semilla = mismo audio" mientras
    # OTRA sesion habla encima. Es justo lo que el candado del modelo no cubre
    # por si solo: una sesion lo SUELTA en cada pausa -- callada no debe
    # secuestrar la CPU de nadie -- y la otra se cuela en mitad de su locucion.
    #
    # Las dos se alimentan por el camino dificil, frase a frase esperando el
    # evento esperando=true, que es precisamente el que provoca las pausas.
    # Mandando el texto de golpe esto no probaria nada: sin pausa nadie suelta
    # el candado a mitad y las dos saldrian identicas aunque el fallo siguiera.
    #
    # Se comprueba ADEMAS que de verdad coexistieron, mirando /health mientras
    # corren. Sin ese dato una pasada verde no distinguiria "no interfieren" de
    # "se ejecutaron una detras de otra", que es un aprobado por la puerta de
    # atras.
    if "concurrencia" in pruebas:
        print("\n[dos sesiones a la vez]")

        t = time.time()
        pcm_sola, _, n_sola = asyncio.run(ws_sesion(
            a.url, a.token, FRASES, a.voz, a.cfg, a.semilla, a.pasos))
        print(f"  sola  {dur(pcm_sola):5.2f} s · {len(pcm_sola)} bytes · "
              f"md5 {md5(pcm_sola)} · en {time.time()-t:.1f} s")

        async def dos_a_la_vez():
            """Las dos sesiones, y un vigilante que cuenta cuantas hay vivas."""
            maximo = [0]
            parar = asyncio.Event()

            async def vigilar():
                while not parar.is_set():
                    try:
                        salud = json.load(pedir(f"{a.url}/health", a.token))
                        maximo[0] = max(maximo[0],
                                        len(salud["sesiones"]["abiertas"]))
                    except Exception:
                        pass
                    await asyncio.sleep(0.2)

            ojo = asyncio.create_task(vigilar())
            try:
                res = await asyncio.gather(*[
                    ws_sesion(a.url, a.token, FRASES, a.voz, a.cfg, a.semilla,
                              a.pasos)
                    for _ in range(2)])
            finally:
                parar.set()
                await ojo
            return res, maximo[0]

        t = time.time()
        (uno, dos), a_la_vez = asyncio.run(dos_a_la_vez())
        print(f"  las dos en {time.time()-t:.1f} s · maximo de sesiones vivas "
              f"a la vez segun /health: {a_la_vez}")
        for etiqueta, (pcm, _, nombre) in (("1a", uno), ("2a", dos)):
            marca = "==" if md5(pcm) == md5(pcm_sola) else "!="
            print(f"  {etiqueta}    {dur(pcm):5.2f} s · {len(pcm)} bytes · "
                  f"md5 {md5(pcm)} {marca} sola · sesion {nombre}")

        if a_la_vez < 2:
            fallos.append("las dos sesiones no llegaron a coexistir: la prueba "
                          "de concurrencia no probo nada")
            print("  FALLO /health nunca vio dos sesiones vivas a la vez")
        else:
            print("  OK  coexistieron de verdad")

        for etiqueta, (pcm, _, _) in (("1a", uno), ("2a", dos)):
            if md5(pcm) == md5(pcm_sola):
                print(f"  OK  la {etiqueta} == sesion a solas: identica bit a bit")
            else:
                n = min(len(pcm), len(pcm_sola))
                iguales = next((i for i in range(n) if pcm[i] != pcm_sola[i]), n)
                fallos.append(f"concurrencia: la {etiqueta} != sesion a solas")
                print(f"  FALLO la {etiqueta} != sesion a solas: {len(pcm)} vs "
                      f"{len(pcm_sola)} bytes, primer byte distinto en {iguales}")

    # ------------------------------ 3a) una sesion DISTINTA en la pausa --
    # La concurrencia de arriba mete dos sesiones IGUALES: si la intrusa deja
    # el proceso con los mismos pasos y el mismo neg_cada, esos dos estados no
    # se notan aunque nadie los reponga. Aqui la intrusa es DISTINTA (pasos+4,
    # otra semilla) y entra en la pausa peor: A abre con una frase corta --
    # menos de una ventana de 5 tokens -- y se queda parada ANTES de generar
    # su primer fotograma, esperando la ventana de adelanto. Al reanudar, sin
    # foto_generacion/reponer_generacion, A sale con los pasos de B y sin su
    # rampa de arranque (B ya la consumio), y B remata con los pasos de A.
    #
    # El orden lo fija el cliente: A avisa de que esta parada (antes_de_seguir),
    # entonces arranca B, y A no manda su segunda frase hasta que B ha SONADO.
    # A solo recupera el candado cuando B lo suelta en su propia pausa, asi que
    # para entonces B lleva decenas de fotogramas hechos.
    if "pausa" in pruebas or "pausa-stream" in pruebas:
        print("\n[una sesion distinta en la pausa]")
        FRASES_PAUSA = ["Sí, claro."] + FRASES
        pasos_b, semilla_b = a.pasos + 4, a.semilla + 1

        t = time.time()
        pcm_sola_a, _, _ = asyncio.run(ws_sesion(
            a.url, a.token, FRASES_PAUSA, a.voz, a.cfg, a.semilla, a.pasos))
        print(f"  A sola {dur(pcm_sola_a):5.2f} s · md5 {md5(pcm_sola_a)} · "
              f"pasos {a.pasos}, semilla {a.semilla} · en {time.time()-t:.1f} s")

        def comparar(etiqueta, pcm, ref):
            if md5(pcm) == md5(ref):
                print(f"  OK  {etiqueta} == a solas: identica bit a bit")
                return
            n = min(len(pcm), len(ref))
            i = next((k for k in range(n) if pcm[k] != ref[k]), n)
            fallos.append(f"pausa: {etiqueta} != a solas")
            print(f"  FALLO {etiqueta} != a solas: {len(pcm)} vs {len(ref)} "
                  f"bytes, primer byte distinto en {i}")

    if "pausa" in pruebas:
        t = time.time()
        pcm_sola_b, _, _ = asyncio.run(ws_sesion(
            a.url, a.token, [" ".join(FRASES)], a.voz, a.cfg, semilla_b, pasos_b))
        print(f"  B sola {dur(pcm_sola_b):5.2f} s · md5 {md5(pcm_sola_b)} · "
              f"pasos {pasos_b}, semilla {semilla_b} · en {time.time()-t:.1f} s")

        async def con_intrusa():
            a_parada, b_sono = asyncio.Event(), asyncio.Event()
            veces = [0]

            async def antes_de_seguir():
                # Solo la primera pausa espera a B; las siguientes siguen el
                # camino normal (B ya esta dentro o ha terminado).
                veces[0] += 1
                if veces[0] == 1:
                    a_parada.set()
                    await asyncio.wait_for(b_sono.wait(), 60)

            async def B():
                await asyncio.wait_for(a_parada.wait(), 60)
                return await ws_sesion(a.url, a.token, [" ".join(FRASES)],
                                       a.voz, a.cfg, semilla_b, pasos_b,
                                       al_primer_audio=b_sono.set)

            return await asyncio.gather(
                ws_sesion(a.url, a.token, FRASES_PAUSA, a.voz, a.cfg,
                          a.semilla, a.pasos, antes_de_seguir=antes_de_seguir),
                B())

        t = time.time()
        (pcm_a, _, n_a), (pcm_b, _, n_b) = asyncio.run(con_intrusa())
        print(f"  con intrusa: A {dur(pcm_a):5.2f} s md5 {md5(pcm_a)} ({n_a}) · "
              f"B {dur(pcm_b):5.2f} s md5 {md5(pcm_b)} ({n_b}) · "
              f"en {time.time()-t:.1f} s")
        comparar("A (pausada antes de su primer fotograma)", pcm_a, pcm_sola_a)
        comparar("B (la intrusa, pasos+4)", pcm_b, pcm_sola_b)

    # La misma pausa, pero la intrusa es /tts/stream con otros pasos y con
    # neg_cada 2: _sintetizar fija los dos en el modelo y no los devuelve, asi
    # que sin reposicion A reanudaria con ellos. neg_cada solo existe en el
    # motor openvino; en torch la peticion lo acepta y no cambia nada, y la
    # prueba sigue valiendo por los pasos.
    if "pausa-stream" in pruebas:
        async def con_stream():
            veces = [0]

            async def antes_de_seguir():
                veces[0] += 1
                if veces[0] != 1:
                    return
                tarea = asyncio.create_task(asyncio.to_thread(
                    http_stream, a.url, a.token, " ".join(FRASES), a.voz,
                    a.cfg, semilla_b, pasos_b, 2))
                # Esperar a que la intrusa TENGA el modelo: A ha soltado el
                # candado, asi que "ocupado" solo puede ser ella.
                limite = time.time() + 60
                while time.time() < limite:
                    salud = json.load(pedir(f"{a.url}/health", a.token))
                    if salud["ocupado"]:
                        break
                    await asyncio.sleep(0.05)
                else:
                    raise AssertionError("/tts/stream nunca llego a ocupar el modelo")
                antes_de_seguir.tarea = tarea

            res = await ws_sesion(a.url, a.token, FRASES_PAUSA, a.voz, a.cfg,
                                  a.semilla, a.pasos,
                                  antes_de_seguir=antes_de_seguir)
            intrusa = await getattr(antes_de_seguir, "tarea")
            return res, intrusa

        t = time.time()
        (pcm_a2, _, n_a2), pcm_intrusa = asyncio.run(con_stream())
        print(f"  con /tts/stream de intrusa (pasos {pasos_b}, neg_cada 2, "
              f"{dur(pcm_intrusa):.2f} s): A {dur(pcm_a2):5.2f} s md5 "
              f"{md5(pcm_a2)} ({n_a2}) · en {time.time()-t:.1f} s")
        comparar("A (con /tts/stream en su pausa)", pcm_a2, pcm_sola_a)

    # ------------------------------------------------------- 3b) respiro --
    # Con respiro (el defecto real de las sesiones) el audio no puede coincidir
    # por md5 con " ".join, porque el aire se anade despues. Lo que si se puede
    # es REPRODUCIRLO: el respiro es un post-proceso puro sobre el audio sin
    # respiro, y aqui se aplica en local y se compara md5. Si sale igual, el
    # aire esta INSERTADO y el habla es la misma; si no, el respiro esta
    # cambiando algo que no deberia tocar.
    if "respiro" in pruebas:
        print("\n[respiro]")
        salud = json.load(pedir(f"{a.url}/health", a.token))
        resp = salud["sesiones"].get("respiro", {})
        fot = resp.get("fotogramas", 2)
        alarga = resp.get("alarga", 0)
        tope = resp.get("tope", 8)
        umbral = resp.get("umbral_rms", 0.006)
        pico = resp.get("umbral_pico", 0.03)
        prerrollo = resp.get("prerrollo", 240)
        pie = resp.get("pie", 960)
        pcm_sin, _, _ = asyncio.run(ws_sesion(
            a.url, a.token, FRASES, a.voz, a.cfg, a.semilla, a.pasos,
            respiro=False))
        pcm_resp, _, _ = asyncio.run(ws_sesion(
            a.url, a.token, FRASES, a.voz, a.cfg, a.semilla, a.pasos,
            respiro=True))
        print(f"  sin respiro {dur(pcm_sin):5.2f} s · con respiro "
              f"{dur(pcm_resp):5.2f} s (pausa desde {fot} fotogramas, "
              f"alarga {alarga}, tope {tope})")

        esperado = aplicar_respiro(pcm_sin, fot, alarga, tope, umbral, pico,
                                   prerrollo, pie)
        if md5(esperado) == md5(pcm_resp):
            print("  OK  el respiro es exactamente el post-proceso documentado: "
                  "mismo md5 aplicandolo en local sobre el audio sin respiro")
        else:
            fallos.append("respiro: el audio no es el post-proceso del audio "
                          "sin respiro")
            n = min(len(esperado), len(pcm_resp))
            i = next((k for k in range(n) if esperado[k] != pcm_resp[k]), n)
            print(f"  FALLO simulado {len(esperado)} bytes vs servidor "
                  f"{len(pcm_resp)}, primer byte distinto en {i}")

        rachas = rachas_calladas(pcm_resp, umbral)
        pausas = [r for r in rachas if r >= 2]
        larga = max(rachas, default=0)
        print(f"  rachas de silencio {sorted(rachas, reverse=True)[:8]} "
              f"(fotogramas de 133 ms)")
        # Ninguna pausa puede pasar de tope+alarga: llegado al tope se recorta
        # lo que siga. Los dos de margen son los fotogramas de borde, que
        # quedan bajo el umbral de deteccion pero por encima del de recorte.
        if larga <= tope + alarga + 2:
            print(f"  OK  ninguna racha pasa del tope "
                  f"({larga} <= {tope}+{alarga}+2)")
        else:
            fallos.append(f"respiro: racha de {larga} fotogramas con tope "
                          f"{tope}+{alarga}")
            print(f"  FALLO racha de {larga} fotogramas; el tope es "
                  f"{tope}+{alarga}")
        # CUANTAS pausas hay lo decide el modelo, no este codigo: el respiro ya
        # no mete marcadores en el texto, solo alarga las que el modelo hace
        # (que estan siempre en un final de frase; ver el bloque RESPIRO). Lo
        # que si tiene que cumplirse es que el aire ESTE, y que sea el que se
        # anuncia: la pausa mas larga tiene que crecer exactamente `alarga`.
        sin_respiro = rachas_calladas(pcm_sin, umbral)
        larga_sin = max(sin_respiro, default=0)
        if larga_sin >= fot and larga == min(larga_sin, tope) + alarga:
            print(f"  OK  el aire esta puesto: la pausa mas larga pasa de "
                  f"{larga_sin} a {larga} fotogramas "
                  f"({larga*FOT/RITMO:.2f} s) · pausas {sorted(pausas, reverse=True)[:6]}")
        else:
            fallos.append(f"respiro: la pausa mas larga es {larga} fotogramas "
                          f"y sin respiro era {larga_sin} (alarga={alarga})")
            print(f"  FALLO pausa mas larga {larga}, sin respiro {larga_sin}, "
                  f"alarga {alarga}")

    # ----------------------------------- 4) corte a mitad y limpieza --
    if "corte" in pruebas:
        print("\n[corte a mitad]")
        antes = json.load(pedir(f"{a.url}/health", a.token))["sesiones"]["abiertas"]
        visto = {}

        def mirar(nombre):
            """Con el websocket todavia abierto: aqui la sesion TIENE que estar.

            Sin esta foto, un 404 despues no probaria nada -- podria ser que la
            sesion no se hubiera registrado nunca.
            """
            visto["codigo"], visto["cuerpo"] = estado_sesion(a.url, a.token, nombre)
            visto["salud"] = json.load(
                pedir(f"{a.url}/health", a.token))["sesiones"]["abiertas"]

        t = time.time()
        pcm_corto, _, nombre = asyncio.run(ws_sesion(
            a.url, a.token, FRASES, a.voz, a.cfg, a.semilla, a.pasos,
            cortar_en=48000,          # ~1 s de audio: bien dentro de la locucion
            antes_de_cortar=mirar))
        print(f"  cortado tras {dur(pcm_corto):.2f} s de audio "
              f"({len(pcm_corto)} bytes) en {time.time()-t:.1f} s")
        if visto.get("codigo") == 200 and nombre in visto.get("salud", []):
            print(f"  OK  antes de cortar: GET /tts/sesion/{nombre} -> 200, "
                  f"viva={visto['cuerpo']['viva']}, "
                  f"generaciones={visto['cuerpo']['generaciones']}")
        else:
            fallos.append("la sesion no estaba viva ANTES de cortar")
            print(f"  FALLO antes de cortar: {visto}")

        # El servidor tiene que enterarse y desmontar. Se mide cuanto tarda.
        t = time.time()
        codigo, cuerpo = None, None
        while time.time() - t < 30:
            codigo, cuerpo = estado_sesion(a.url, a.token, nombre)
            if codigo == 404:
                break
            time.sleep(0.05)
        if codigo == 404:
            print(f"  OK  tras cortar: GET /tts/sesion/{nombre} -> 404 en "
                  f"{time.time()-t:.2f} s")
        else:
            fallos.append("la sesion sigue viva tras cortar el websocket")
            print(f"  FALLO GET /tts/sesion/{nombre} -> {codigo} {cuerpo}")

        # El candado del modelo se suelta cuando MUERE el hilo, no cuando cae el
        # socket: generate() solo se puede abortar desde su siguiente put(), asi
        # que hay un retardo de como mucho un trozo (~133 ms de audio). Se mide
        # en vez de suponerlo.
        t = time.time()
        salud = json.load(pedir(f"{a.url}/health", a.token))
        while salud["ocupado"] and time.time() - t < 30:
            time.sleep(0.05)
            salud = json.load(pedir(f"{a.url}/health", a.token))
        libre = time.time() - t
        vivas = salud["sesiones"]["abiertas"]
        if not salud["ocupado"] and vivas == antes:
            print(f"  OK  candado del modelo libre {libre:.2f} s despues; "
                  f"sesiones abiertas {vivas}")
        else:
            fallos.append(f"tras 30 s sigue ocupado={salud['ocupado']}, "
                          f"sesiones={vivas}")
            print(f"  FALLO /health: sesiones {vivas} (antes {antes}), "
                  f"ocupado {salud['ocupado']}")

        # Y que el modelo funcione de verdad, no solo que el flag este a False.
        t = time.time()
        prueba = http_stream(a.url, a.token, "Vale.", a.voz, a.cfg, a.semilla,
                             a.pasos)
        print(f"  OK  el modelo vuelve a generar: {dur(prueba):.2f} s de audio "
              f"en {time.time()-t:.1f} s")

    # ---------------------------------------------- 5) errores del protocolo --
    if "errores" in pruebas:
        print("\n[errores del protocolo]")

        async def primer_error(mensajes):
            """Manda lo que sea y devuelve el primer evento de error.

            Se salta los eventos previos ('abierta' y demas): en los casos que
            fallan a mitad de conversacion el error no es el primero que baja.
            """
            ws_url = a.url.replace("http://", "ws://")
            cab = {"authorization": f"Bearer {a.token}"} if a.token else {}
            async with websockets.connect(f"{ws_url}/tts/sesion/ws",
                                          additional_headers=cab,
                                          ping_interval=None) as ws:
                for m in mensajes:
                    await ws.send(m if isinstance(m, str) else json.dumps(m))
                buf, pcm, eventos, vistos = b"", bytearray(), [], 0
                while True:
                    buf = desmarcar(buf + await asyncio.wait_for(ws.recv(), 20),
                                    pcm, eventos)
                    for ev in eventos[vistos:]:
                        if ev["tipo"] == "error":
                            return ev
                    vistos = len(eventos)

        casos = [
            ("velocidad 1.15", [{"accion": "abrir", "velocidad": 1.15}]),
            ("velocidad fuera de rango", [{"accion": "abrir", "velocidad": 3.0}]),
            ("no empieza por abrir", [{"accion": "texto", "texto": "hola"}]),
            ("no es JSON", ["{roto"]),
            ("voz inexistente", [{"accion": "abrir", "voz": "no-existe"}]),
            ("accion desconocida", [{"accion": "abrir"}, {"accion": "bailar"}]),
        ]
        for etiqueta, mensajes in casos:
            try:
                ev = asyncio.run(primer_error(mensajes))
            except Exception as e:
                ev = {"tipo": type(e).__name__, "texto": str(e)[:60]}
            ok = ev.get("tipo") == "error"
            if not ok:
                fallos.append(f"errores: {etiqueta} no dio evento de error")
            print(f"  {'OK ' if ok else 'FALLO'} {etiqueta:26s} -> "
                  f"{str(ev.get('texto', ev))[:90]}")

    # ------------------------------------------------------------- 6) auth --
    if "auth" in pruebas and a.token:
        print("\n[autenticacion]")
        for etiqueta, tok, cab in (("sin token", "", True),
                                   ("token malo", "x" * 8, True),
                                   ("por query", a.token, False)):
            try:
                asyncio.run(ws_sesion(a.url, tok, ["Hola."], a.voz, a.cfg,
                                      a.semilla, a.pasos, por_cabecera=cab,
                                      cortar_en=1))
                salida = "conecto"
            except Exception as e:
                salida = f"{type(e).__name__}: {str(e)[:60]}"
            espera_ok = etiqueta == "por query"
            ok = (salida == "conecto") == espera_ok
            print(f"  {'OK ' if ok else 'FALLO'} {etiqueta:12s} -> {salida}")
            if not ok:
                fallos.append(f"auth {etiqueta}")

    print("\n" + "=" * 70)
    if fallos:
        print(f"FALLOS ({len(fallos)}):")
        for f in fallos:
            print(f"  - {f}")
        return 1
    print("todo correcto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
