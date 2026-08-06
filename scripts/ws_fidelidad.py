#!/usr/bin/env python
"""Prueba el websocket de sesiones contra la via HTTP, que es la referencia.

    pkgs/vibevoice/.venv/bin/python scripts/ws_fidelidad.py \
        --url http://127.0.0.1:8082 --token "$VOZ_TOKEN"

Cuatro cosas, y las cuatro con numeros:

  1. FIDELIDAD. El PCM que baja por el websocket tiene que ser IDENTICO -- md5,
     no "parecido" -- al de la sesion HTTP con las mismas frases, la misma voz
     y la misma semilla, y tambien al de /tts/stream con las frases unidas por
     SALTO DE LINEA. Lo de los saltos de linea no es un detalle: la sesion
     tokeniza cada frase como texto.strip() + "\\n", asi que unir con espacios
     da OTRO texto y por tanto otro audio. Comparar contra la referencia
     equivocada es el error clasico aqui.

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


def http_stream(url, token, texto, voz, cfg, semilla, pasos):
    """/tts/stream de una vez. Se le quitan los 44 bytes de cabecera WAV."""
    cuerpo = {"texto": texto, "voz": voz, "cfg_scale": cfg, "semilla": semilla}
    if pasos is not None:
        cuerpo["pasos"] = pasos
    return pedir(f"{url}/tts/stream", token, cuerpo).read()[44:]


def http_sesion(url, token, nombre, frases, voz, cfg, semilla, pasos):
    """Sesion HTTP: se meten todas las frases y se escucha el WAV continuo."""
    import threading
    base = {"voz": voz, "cfg_scale": cfg, "semilla": semilla}
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
                    antes_de_cortar=None):
    """Habla por el websocket y devuelve (pcm, eventos, nombre).

    cortar_en: si viene, se cierra el socket a lo bruto en cuanto hayan bajado
    esos bytes de PCM -- el caso "el cliente se fue a mitad".
    antes_de_cortar: se llama con el nombre de la sesion justo antes de cortar,
    que es el unico momento en que se la puede ver viva desde fuera.
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
             "velocidad": 1.0, "semilla": semilla}
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
            buf = desmarcar(buf + msg, pcm, eventos)
            for ev in eventos[antes:]:
                if traza is not None:
                    traza.append((round(time.perf_counter() - t0, 2), ev))
                if ev["tipo"] == "abierta":
                    nombre = ev["sesion"]
                if ev["tipo"] == "error":
                    raise AssertionError(f"error del servidor: {ev['texto']}")
                if ev["tipo"] == "esperando" and ev["esperando"]:
                    if pendientes:
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
                    default="fidelidad,eventos,concurrencia,corte,errores,auth")
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
        pcm_str = http_stream(a.url, a.token, "\n".join(FRASES), a.voz, a.cfg,
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
