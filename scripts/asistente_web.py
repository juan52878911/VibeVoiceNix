#!/usr/bin/env python
"""Asistente de voz en el navegador: escribes, responde hablando.

EL "CUELGUE CON RESPUESTAS LARGAS", RESUELTO -- Y NO ERA LO QUE PARECIA
La peticion larga que "no terminaba en 9 minutos" no era este bucle ni la
sonda, y el servicio de voz SI era el culpable: el registro de la VM lo dejo
escrito (2026-08-05, sesion ws-8247c2d0). El modelo agoto el texto de la
respuesta y su EOS nunca llego: siguio generando audio sin texto detras
-- llevaba 766 posiciones de mas cuando salto el tope de cache ("-766 tokens
pendientes" en el journal) y seis minutos despues seguia, hasta que la
desconexion del cliente lo aborto. Sellar el texto (lo unico que hacia el
tope) no frena eso: las lecturas ya devolvian vacio. Sin 'hecho' del
servidor, este puente retransmitia el chorro para siempre y la peticion no
acababa nunca. Es un descarrile del modelo y no pasa siempre; por eso las
mismas ~15 frases otras veces terminan en ~95 s con su 'hecho'.

La VM quedaba "limpia" tras cada intento (`ocupado: false`, `abiertas: []`)
no porque la sesion terminara bien, sino porque al morir el CLIENTE de prueba
el websocket se caia y abortar() la desmontaba. Y el "proceso muerto": el
puente de la primera reproduccion ya estaba muerto ANTES de la peticion larga
(el cliente recibio 'Connection refused' al instante; el arranque en segundo
plano de esa prueba murio sin dejar traza), y el de la segunda seguia VIVO
tras el cuelgue. No hay ningun cuelgue del proceso reproducible.

El arreglo esta en las dos puntas:
  - voz_stream.py (LocucionDescarrilada): si generate() sigue pidiendo
    ventanas MARGEN_EOS posiciones mas alla del texto sellado, se corta la
    locucion con 'error' + 'hecho' y la sesion muere limpia.
  - aqui (TOPE_AUDIO_BASE/TOPE_AUDIO_POR_TOKEN): si baja bastante mas audio
    del que el texto entregado puede justificar, se corta con error en vez de
    retransmitir parloteo. Cinturon por si el servidor desplegado no lleva aun
    el freno, o por si descarrila de otra forma.

    pkgs/vibevoice/.venv/bin/python scripts/asistente_web.py
    # y abre http://127.0.0.1:8090

Hace de puente entre tres cosas que ya funcionan por separado:

    navegador  ->  este puente  ->  MiniMax u Ollama  (el texto)
                                ->  voz-stream        (la voz)

El proveedor se elige por el nombre del modelo: los 'MiniMax-*' van a la
suscripcion (mas rapidos y mas capaces), el resto a Ollama en local. La
credencial se reutiliza de opencode; no hay copia en el repo.

POR QUE UN PUENTE Y NO LLAMAR DESDE EL NAVEGADOR
Porque el navegador tendria que hacer dos cosas que no sabe hacer bien: trocear
el flujo del LLM por frases segun llega, y encadenar peticiones de voz sin que
se oigan los cortes. Aqui eso ya esta resuelto y probado; el navegador solo
recibe un flujo continuo de PCM y lo reproduce.

UNA SESION, UNA LOCUCION -- Y POR QUE SE CAMBIO
Antes esto pedia POST /tts/stream UNA VEZ POR FRASE. Sonaba a lista de frases y
no a alguien hablando: cada peticion arranca del prefijo pristino, sin nada de
lo anterior, asi que la entonacion se reiniciaba en cada punto y las junturas se
oian. Ahora se abre UNA sesion por respuesta contra

    WS /tts/sesion/ws

y cada frase que sale del troceador se le mete a esa MISMA generate() viva con
{"accion":"texto"}. El servicio garantiza -- verificado en scripts/ws_fidelidad.py,
md5 a md5 -- que alimentar frase a frase da el mismo audio que mandar todo el
texto de una vez. O sea: la prosodia es la de un texto continuo, pero se empieza
a oir en cuanto hay una frase.

Lo que se pierde con el cambio es la VELOCIDAD del servidor: el websocket la
rechaza a proposito (estirar() es WSOLA y necesita la locucion entera, que aqui
no tiene fin conocido). La pagina la aplica al reproducir, y lo dice.

LO QUE SE VE EN LA PAGINA
Los tres hitos que importan, en vivo: cuanto tarda el LLM en soltar el primer
token, cuanto en tener la primera frase, y cuanto hasta que suena. Medido contra
la VM con qwen3:1.7b, del primer token del LLM al primer sonido van 0,53 s de
media (6 pasadas), y de esos solo ~0,3 s son la voz: el resto es el LLM
terminando la frase. Por eso la pagina los separa: para que se vea donde esta el
tiempo de verdad.

ADEMAS, DESDE LA PAGINA
  - La instruccion de sistema del LLM se edita en un panel plegable y viaja con
    cada pregunta; persiste en localStorage del navegador y, vacia, vuelve a la
    de serie (--sistema). El sufijo /no_think sigue siendo cosa SOLO de Ollama.
  - Un boton de microfono graba (getUserMedia + MediaRecorder), manda el audio
    a POST /stt de este puente, que lo reenvia al /stt de la API de voz
    (whisper; --api-url o VOZ_API_URL, el 8080 local por defecto) y deja la
    transcripcion en el cuadro de la pregunta PARA REVISARLA, no la pregunta
    sola. getUserMedia exige contexto seguro: HTTPS, salvo en localhost. En
    http://127.0.0.1:8090 funciona; desde otra maquina de la red, no, y el
    boton sale deshabilitado explicando por que.

ESCUCHA CONTINUA (nueva, y arranca APAGADA)
Un interruptor en la pagina abre el microfono EN CONTINUO, sin palabra de
activacion: la pagina segmenta por energia (VAD), cada intervencion pasa por
POST /escuchar -- huella de voz para saber QUIEN habla, whisper para el texto,
y una compuerta (un LLM pequeño y rapido) que decide si iba dirigida al
asistente -- y solo entonces se responde, con el historial de la conversacion
detras. Quien habla se etiqueta contra perfiles de voz (scripts/oido.py) que
se crean solos para desconocidos y se matriculan o renombran desde la pagina.

La realimentacion acustica (el asistente oyendose a si mismo y contestandose)
se corta por HUELLA DE VOZ: el puente aprende la voz que el mismo emite --
guarda unos segundos del PCM de cada respuesta y refresca el perfil
'asistente' -- y descarta todo lo que case con ella. Por eso se puede
interrumpir mientras habla. Si no hay modelo de huellas, la pagina se repliega
a medio duplex: ignora el microfono mientras suena la voz. Los porques y las
medidas, en las cabeceras de scripts/oido.py y scripts/conversacion.py.

INTERRUMPIRLE HABLANDO: CALLAR Y ENTENDER SON DOS COSAS DISTINTAS
El ciclo de antes decidia TODO despues de whisper, y whisper costaba 2,1-2,9 s:
si le hablabas encima seguia hablando dos segundos largos. Pero para callarse
no hace falta saber QUE le has dicho, solo que hay una persona hablando, y eso
lo sabe la huella de voz en 10 ms. Asi que la interrupcion va por tres
escalones, cada uno mas lento y mas listo que el anterior:

  1. BAJAR LA VOZ, ~0,15 s. En cuanto el VAD dice "hay voz" -- 96 ms de
     energia seguida -- la pagina baja el volumen al 18 % (duckear()). No
     decide nada todavia: es reversible, y si resulta ser el propio asistente
     colandose por el microfono, el volumen vuelve y solo se ha oido un bache.
  2. CALLAR DE VERDAD, 0,6-1,7 s segun como arranque la frase. Con 0,3 s de
     voz grabada, POST /barrera pregunta a la huella si es una persona
     matriculada. Si lo es, se aborta la locucion, se cierra la sesion de voz
     y se apunta POR DONDE IBA. Se prueba en escalones (0,3 · 0,4 · 0,5 · 0,7
     · 1,0 · 1,4 s de voz) porque el primero cuesta 10 ms y acierta la mitad
     de las veces; la tabla, en oido.py. Si la intervencion es demasiado corta
     para decidirse -- «¿Cómo?» son 192 ms de voz --, recoge la huella del
     trozo entero al llegar a /escuchar, sin esperar a la transcripcion.
  3. ENTENDER, despues. Cuando el VAD cierra la frase, el trozo entero pasa
     por el camino de siempre y ademas por interrupcion.clasificar(): «espera»
     y «para» y «sigue» son frases hechas y se resuelven SIN LLM y SIN
     compuerta, que es lo que hace que «espera» -> «¿qué pasa?» suene en 1,07 s
     desde que dejas de hablar en vez de en cinco segundos. Lo que no es una
     frase hecha va al LLM con el contexto del corte detras.
  4. Y SI RESULTA QUE NO ERA PARA EL, VUELVE. La barrera calla sin saber que
     le han dicho, asi que a veces callara porque hablabas con otra persona.
     Sin marcha atras eso le dejaria mudo a mitad de frase: reanudar() retoma
     la locucion por donde iba en cuanto la compuerta dice que no era para el.

POR DONDE IBA CUANDO LE CORTASTE
Al abortar, la pagina parte la respuesta en dos: lo que LLEGO A SONAR y lo que
se quedo dentro. Las dos mitades van al historial y las dos se le enseñan al
modelo (interrupcion.texto_para_el_modelo). Es lo que permite «detalla eso
ultimo» -- que habla de lo dicho -- y «sigue» -- que habla de lo que faltaba.

LA COMPUERTA YA NO ESPERA SU TURNO
Antes el orden era whisper -> compuerta -> LLM, en fila, y la compuerta ponia
0,47-0,70 s en el camino critico. Ahora /preguntar la lanza EN PARALELO con el
LLM y solo retiene la entrega de la primera frase a la sesion de voz hasta
tener veredicto. Como el LLM tarda 1,15-6,5 s en tener la primera frase, la
compuerta termina antes y no se nota. Lo que cuesta: si la respuesta era NO,
se han gastado ~0,7 s de LLM para nada. Se apaga con --sin-solapar.

WHISPER, EL SUMANDO GORDO, EN NATIVO
whisper.cpp corria en Docker, donde no hay Metal, y costaba 2,1-2,9 s por
frase FUERA DEL LARGO del audio -- es arranque del codificador, no proceso.
Compilado nativo en el Mac cuesta 0,28 s con el MISMO modelo y la MISMA
transcripcion (medido, tabla en scripts/escucha_fidelidad.py). Se levanta con
scripts/whisper-mac.sh y se le apunta con --whisper-url; sin esa opcion todo
sigue yendo por voz-api como siempre.

DEPENDENCIA: el cliente de websocket (`websockets`, el mismo que usa
scripts/ws_fidelidad.py). Esta en pkgs/vibevoice/.venv, que es con lo que hay
que arrancar esto:

    pkgs/vibevoice/.venv/bin/python scripts/asistente_web.py

Para las huellas de voz hacen falta ademas speechbrain y torchaudio en ese
mismo venv (cinco paquetes; torch ya estaba). Sin ellos todo lo demas
funciona y la escucha continua pasa a medio duplex.
"""
import argparse
import base64
import json
import os
import queue
import re
import struct
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from narrador import trocear  # noqa: E402
from asistente import ABRE_PENSAMIENTO, CIERRA_PENSAMIENTO, limpiar, preguntar  # noqa: E402
from conversacion import RECUERDO_COMPUERTA, decidir, preguntar_con_historial  # noqa: E402
from interrupcion import clasificar, que_decir  # noqa: E402
from oido import Oido  # noqa: E402
import perfiles  # noqa: E402
import herramientas  # noqa: E402

try:
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect as ws_conectar
except ImportError:  # pragma: no cover - solo para poder dar un error legible
    ws_conectar = None

    class ConnectionClosed(Exception):
        """Marcador para que el modulo importe aunque falte `websockets`."""

CFG = {}
# Los perfiles de voz (scripts/oido.py). None si main() no llego a crearlos.
OIDO = None
# El ejecutor de herramientas (scripts/herramientas.py). Uno solo para todo el
# proceso: es donde viven las confirmaciones pendientes, y una accion
# preparada tiene que sobrevivir de una peticion HTTP a la siguiente -- cada
# /preguntar es una conexion nueva, asi que si el estado viviera en el handler
# no habria confirmacion que valiera. Se reparte por 'sesion', que manda la
# pagina: dos pestañas no se pisan la una a la otra.
HERRAMIENTAS = None

# Sin ruido del socket durante esto, se da por rota la sesion. El plazo largo
# es a proposito: el servicio deja de emitir mientras espera texto -- hasta
# VIBEVOICE_ESPERA_TEXTO, 20 s por defecto -- y eso es normal, no una averia.
SILENCIO_MAXIMO = 90.0

# Tope de audio por texto entregado: el freno contra una locucion descarrilada
# (EOS que no llega y el modelo parloteando sin texto detras; ver la cabecera).
# El silencio lo cubre SILENCIO_MAXIMO, pero un descarrile no calla: emite.
# Medido en castellano: ~0,27 s de audio por token del sintetizador (334
# tokens ~ 90 s). El doble de eso mas una base holgada nunca corta una
# locucion legitima, y a un descarrile le deja como mucho medio minuto largo.
TOPE_AUDIO_BASE = 30.0        # segundos de gracia, cubre arranques y colas
TOPE_AUDIO_POR_TOKEN = 0.6    # segundos de audio admitidos por token acusado

# CUANTO VA EL TEXTO POR DELANTE DEL AUDIO, EN TOKENS.
# generate() lee la ventana en curso Y LA SIGUIENTE -- el lookahead con el que
# alarga por adelantado la mascara de atencion -- antes de emitir los latentes
# de la primera. Asi que cuando la sesion dice que lleva N tokens consumidos, lo
# que ha salido por el socket corresponde a N - 2*ventana.
#
# Medido contra la VM en una locucion de 44 tokens: con 15, 25, 35 y 44 tokens
# consumidos, el audio emitido correspondia a 8,8 · 16,2 · 24,4 y 32,5 tokens.
# Adelanto de 10 ±1,5 en todo el recorrido. Sin restarlo, una primera frase
# corta se marcaba como dicha ANTES de que sonara un solo byte de ella.
VENTANA_TEXTO = 5             # TTS_TEXT_WINDOW_SIZE, igual que en voz_stream.py
ADELANTO_TEXTO = 2 * VENTANA_TEXTO

# Los que sirve la suscripcion de MiniMax. M3 primero: es el mas capaz y el
# que se usa por defecto. Los "highspeed" responden antes a cambio de calidad.
MINIMAX_MODELOS = ["MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed",
                   "MiniMax-M2.5", "MiniMax-M2.5-highspeed"]

# r"""...""" y no """...""": el JavaScript de dentro lleva expresiones
# regulares con \s, y Python las lee como secuencias de escape suyas. Sin la r
# el aviso es solo un SyntaxWarning hoy, pero en una version futura es un error
# y ademas el navegador recibiria otra cosa de la que esta escrita aqui.
PAGINA = r"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Asistente de voz</title><style>
:root{--f:#0d0f13;--p:#161a21;--b:#242a35;--t:#e8eaed;--s:#98a2b3;--a:#d99a4e}
*{box-sizing:border-box}
body{margin:0;padding:2.5rem 1.25rem;background:var(--f);color:var(--t);
 font:16px/1.6 ui-sans-serif,system-ui,-apple-system,sans-serif}
main{max-width:44rem;margin:0 auto;display:flex;flex-direction:column;gap:1.5rem}
h1{margin:0;font-size:1.5rem;letter-spacing:-.02em}
.sub{margin:.25rem 0 0;color:var(--s);font-size:.92rem}
.caja{background:var(--p);border:1px solid var(--b);border-radius:10px;padding:1.25rem}
textarea{width:100%;min-height:5rem;background:var(--f);color:var(--t);
 border:1px solid var(--b);border-radius:7px;padding:.75rem;font:inherit;resize:vertical}
textarea:focus{outline:none;border-color:var(--a)}
.fila{display:flex;gap:.75rem;align-items:center;margin-top:.9rem;flex-wrap:wrap}
button{background:var(--a);color:#1a1206;border:0;border-radius:7px;
 padding:.6rem 1.3rem;font:600 .95rem/1 inherit;cursor:pointer}
button:disabled{opacity:.45;cursor:default}
button.sec{background:transparent;color:var(--s);border:1px solid var(--b)}
button.rec{background:#8f3227;color:#f6d9d3;border-color:#8f3227;
 animation:latir 1.1s ease-in-out infinite}
.nota{margin:.6rem 0 0;font-size:.75rem;color:#6b7280}
select{background:var(--f);color:var(--t);border:1px solid var(--b);
 border-radius:7px;padding:.5rem}
.hitos{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:.75rem;margin-top:1rem}
.h{background:var(--f);border:1px solid var(--b);border-radius:7px;padding:.7rem .85rem}
.h .n{font:600 1.35rem/1 ui-monospace,monospace;color:var(--a);font-variant-numeric:tabular-nums}
.h .e{font-size:.72rem;color:var(--s);text-transform:uppercase;letter-spacing:.08em;margin-top:.35rem}
.resp{margin-top:1rem;padding-top:1rem;border-top:1px solid var(--b);
 font-size:.98rem;line-height:1.9;min-height:2rem}
/* Los cuatro estados por los que pasa cada trozo. El color no decora: dice
   en que fase del proceso esta ese texto ahora mismo. Con la sesion abierta
   el audio es UNA locucion continua, asi que 'sonando' no es "empieza otra
   peticion" sino "el modelo va por aqui". */
.t{border-radius:4px;padding:.1rem .25rem;transition:background .25s,color .25s}
.t.pend{color:#6b7280}                                   /* el LLM aun escribe */
.t.seg{background:#2b3550;color:#a9c3f5}                  /* cortado, sin entregar */
.t.sint{background:#4a3a1a;color:#f0c274;
        animation:latir 1.1s ease-in-out infinite}        /* en la sesion */
.t.son{background:#1f4033;color:#7fd6a8}                  /* el modelo va por aqui */
.t.fin{color:#cfd4dc}                                     /* dicho */
@keyframes latir{0%,100%{opacity:1}50%{opacity:.55}}
@media (prefers-reduced-motion:reduce){.t.sint{animation:none}}
.ley{display:flex;gap:.9rem;flex-wrap:wrap;margin-top:.8rem;font-size:.74rem;color:var(--s)}
.ley i{font-style:normal;padding:.1rem .35rem;border-radius:3px}
.ajustes{margin-top:.9rem;border-top:1px solid var(--b);padding-top:.7rem}
.ajustes summary{cursor:pointer;color:var(--s);font-size:.88rem}
.rej{display:grid;gap:.9rem;margin-top:.9rem}
.rej label{display:grid;grid-template-columns:8rem 1fr auto;gap:.6rem;
 align-items:center;font-size:.88rem;color:var(--s)}
.rej b{font:600 .9rem ui-monospace,monospace;color:var(--a);
 font-variant-numeric:tabular-nums;min-width:2.6rem;text-align:right}
.rej i{grid-column:1/-1;font-style:normal;font-size:.75rem;color:#6b7280;margin-top:-.35rem}
.rej input[type=range]{width:100%;accent-color:var(--a)}
.rej select,.rej input[type=number]{background:var(--f);color:var(--t);
 border:1px solid var(--b);border-radius:6px;padding:.35rem}
.est{margin-top:.75rem;font-size:.88rem;color:var(--s)}
.est.err{color:#e0725f}
/* La escucha continua: el chip dice EN QUE ESTA el oido ahora mismo. */
.chip{padding:.25rem .7rem;border-radius:99px;font-size:.8rem;
 border:1px solid var(--b);color:var(--s);white-space:nowrap}
.chip[data-e=escuchando]{border-color:#2b4a6f;color:#7fb3e8}
.chip[data-e=voz]{border-color:#7a5b1e;color:#f0c274;animation:latir 1.1s ease-in-out infinite}
.chip[data-e=proc]{border-color:#4a3a6b;color:#b8a3e8;animation:latir 1.1s ease-in-out infinite}
.chip[data-e=pensando]{border-color:#4a3a6b;color:#b8a3e8;animation:latir 1.1s ease-in-out infinite}
.chip[data-e=hablando]{border-color:#1f4033;color:#7fd6a8}
.vu{width:7rem;height:.55rem;background:var(--f);border:1px solid var(--b);
 border-radius:99px;overflow:hidden;align-self:center}
.vu i{display:block;height:100%;width:0;background:var(--a);transition:width .06s linear}
.log{margin-top:1rem;padding-top:.9rem;border-top:1px solid var(--b);
 font-size:.82rem;color:var(--s);max-height:16rem;overflow:auto;
 display:flex;flex-direction:column;gap:.4rem}
.log .quien{color:var(--a);font-weight:600}
.log .dicho{color:var(--t)}
.log .meta{font-size:.72rem;color:#6b7280;font-variant-numeric:tabular-nums}
.log .fuera{opacity:.65}
.perfil{display:flex;gap:.6rem;align-items:center;font-size:.85rem}
.perfil input{flex:1;background:var(--f);color:var(--t);border:1px solid var(--b);
 border-radius:6px;padding:.4rem}
.perfil .tipo{font-size:.72rem;color:#6b7280;white-space:nowrap}
#perfNombre{background:var(--f);color:var(--t);border:1px solid var(--b);
 border-radius:6px;padding:.5rem}
/* HERRAMIENTAS. Dos filas de fichas: arriba lo que este perfil PUEDE hacer,
   abajo lo que ha hecho en la respuesta en curso. El color no decora: rojizo
   es «esta escribe», y por tanto «esta te va a preguntar antes». */
.herrs{display:flex;gap:.4rem;flex-wrap:wrap;margin-top:.7rem}
.herrs:empty{display:none}
.hf{font-size:.72rem;padding:.2rem .55rem;border-radius:99px;
 border:1px solid var(--b);color:var(--s);white-space:nowrap;
 font-variant-numeric:tabular-nums}
.hf.esc{border-color:#7a3b32;color:#e0a196}
.hf.usando{border-color:#7a5b1e;color:#f0c274;animation:latir 1.1s ease-in-out infinite}
.hf.ok{border-color:#2f6f4f;color:#7fd6a8}
.hf.mal{border-color:#8f3227;color:#e0725f}
.hf.conf{border-color:#7a3b32;color:#e0a196;animation:latir 1.1s ease-in-out infinite}
/* La acción a medias. Es lo único de la página que sale con marco: es lo
   único que puede cambiar algo fuera del asistente. */
.pend{margin-top:.9rem;padding:.8rem 1rem;border:1px solid #7a3b32;
 border-radius:8px;background:#26191680;font-size:.9rem}
.pend b{color:#e0a196;font-weight:600}
</style></head><body><main>
<header><h1>Asistente de voz</h1>
<p class="sub">Escribe y responde hablando. Los tres tiempos de abajo separan
lo que tarda el modelo de lenguaje de lo que tarda la voz. Cada respuesta se
dice en <b>una sola locución</b>: las frases se le van metiendo a la misma
sesión de voz según el modelo las escribe, así que la entonación sigue de una
a otra en vez de reiniciarse en cada punto.</p></header>

<div class="caja">
  <textarea id="q" placeholder="¿Cómo va el despliegue de anoche?">¿Cómo va el despliegue de anoche?</textarea>
  <div class="fila">
    <button id="ir">Preguntar</button>
    <button id="mic" class="sec">Hablar</button>
    <button id="parar" class="sec" hidden>Parar</button>
    <select id="perfil" title="quién contesta: cambia la voz, las coletillas y las herramientas"></select>
    <select id="modelo"></select>
    <label style="color:var(--s);font-size:.88rem">
      <input type="checkbox" id="pensar"> dejar que razone
    </label>
  </div>
  <p class="nota" id="perfilNota"></p>
  <div class="herrs" id="herrJuego"></div>
  <div class="pend" id="pend" hidden>
    <div><b id="pendTexto"></b></div>
    <div class="fila" style="margin-top:.5rem">
      <span class="meta">Contesta «sí» o «no» por voz, o usa el botón. Nada
      se ha hecho todavía.</span>
      <button id="pendNo" class="sec">Cancelar</button>
    </div>
  </div>
  <p class="nota">«Hablar» graba del micrófono, lo transcribe con whisper y deja
  el texto en el cuadro de arriba <b>para revisarlo</b> antes de preguntar.
  Ojo: solo funciona abriendo la página en esta máquina
  (<code>127.0.0.1</code>): el navegador exige HTTPS para dar micrófono, con la
  única excepción de localhost, y esta página se sirve por HTTP. Desde otra
  máquina de la red el botón saldrá deshabilitado.</p>
  <details class="ajustes"><summary>Instrucción para el modelo</summary>
    <textarea id="sistema" style="margin-top:.9rem;min-height:4.5rem"
      placeholder="vacía, vale la de serie"></textarea>
    <div class="fila">
      <button id="sisdef" class="sec">Volver a la de serie</button>
      <span style="font-size:.75rem;color:#6b7280">Se manda con cada pregunta y
      se guarda en este navegador (localStorage), así que sobrevive a recargas.
      Vacía, el puente usa la de serie.</span>
    </div>
  </details>
  <details class="ajustes"><summary>Ajustar la voz</summary>
    <div class="rej">
      <label>Voz <select id="voz"></select></label>
      <label>Expresividad <input type="range" id="cfg" min="1.5" max="4.5" step="0.1" value="3.5">
        <b id="vcfg">3.5</b>
        <i>Entre 3,0 y 3,5 la fidelidad es la misma; 3,5 entona más y habla algo más despacio. A 4,5 el peor caso se dobla y además aplana la melodía.</i></label>
      <label>Velocidad <input type="range" id="vel" min="0.85" max="1.20" step="0.01" value="1.00">
        <b id="vvel">1.00</b>
        <i>Se aplica AQUÍ, al reproducir, y <b>mueve el tono</b>: más rápido suena más agudo. El servicio se niega a hacerlo bien durante una sesión porque estirar el tiempo sin tocar el tono (WSOLA) necesita la locución entera, y una sesión no tiene final conocido. Déjala en 1,00 si quieres la voz tal cual.</i></label>
      <label>Detalle <input type="range" id="pasos" min="4" max="20" step="1" value="6">
        <b id="vpasos">6</b>
        <i>Pasos de difusión. Medido: 6 y 8 dan el mismo RTF; 20 cuesta un 26 % más sin ganar nada audible.</i></label>
      <label>Semilla <input type="number" id="semilla" placeholder="al azar" min="0" style="width:7rem">
        <i>Fija, el mismo texto da siempre el mismo audio. Vacía, cada vez sale distinto.</i></label>
    </div>
  </details>
  <div class="hitos">
    <div class="h"><div class="n" id="h1">—</div><div class="e">1er token</div></div>
    <div class="h"><div class="n" id="h2">—</div><div class="e">1ª frase</div></div>
    <div class="h"><div class="n" id="h3">—</div><div class="e">1er sonido</div></div>
    <div class="h" title="Del cierre de la primera frase al primer sonido. No es
todo voz: la sesión no emite nada hasta poder leer dos ventanas de texto (10
tokens), así que si la primera frase se queda corta, parte de este tiempo es el
modelo esperando a la segunda."><div class="n" id="h4">—</div><div class="e">solo la voz</div></div>
  </div>
  <div class="herrs" id="herrUso"></div>
  <div class="resp" id="texto"></div>
  <div class="ley">
    <span><i class="t pend">escribiendo</i> el LLM aún redacta</span>
    <span><i class="t seg">segmentado</i> frase cerrada, sin entregar</span>
    <span><i class="t sint">en la sesión</i> entregada, esperando su turno</span>
    <span><i class="t son">sonando</i> el modelo va por aquí</span>
  </div>
  <div class="est" id="est"></div>
</div>

<div class="caja">
  <div class="fila">
    <button id="esc">Activar escucha continua</button>
    <span class="chip" id="escEst" data-e="apagada">apagada</span>
    <span class="vu" title="nivel del micrófono contra el umbral de voz"><i id="vui"></i></span>
  </div>
  <p class="nota">Micrófono <b>siempre abierto</b>, sin palabra de activación:
  cada intervención se transcribe y un modelo decide si iba dirigida al
  asistente; solo entonces contesta, con la conversación entera detrás.
  Sabe <b>quién habla</b> por la huella de la voz (el timbre, no lo que se
  dice) y desecha la suya propia.
  Los desconocidos reciben un perfil automático; ponles nombre abajo.
  Arranca apagada y solo funciona en <code>127.0.0.1</code>, como el botón
  Hablar.</p>
  <p class="nota"><b>Puedes cortarle hablando.</b> Baja la voz en cuanto oye a
  alguien y calla del todo en medio segundo, cuando la huella confirma que
  eres una persona y no él mismo — sin esperar a saber qué has dicho. Después
  entiende: «espera» o «¿cómo?» te devuelven un «¿qué pasa?» sin pasar por el
  modelo grande, «para» le deja callado, «sigue» retoma por donde iba, y
  cualquier otra cosa la responde sabiendo <b>qué llevaba dicho y qué le
  quedaba</b>. Si resulta que no le hablabas a él, vuelve solo a su frase.</p>
  <details class="ajustes"><summary>Perfiles de voz (quién es quién)</summary>
    <div id="perfLista" style="display:flex;flex-direction:column;gap:.5rem;margin-top:.9rem"></div>
    <div class="fila">
      <input id="perfNombre" placeholder="nombre">
      <button id="perfAlta" class="sec">Grabar 5 s y matricular</button>
    </div>
    <p class="nota">Es un diferenciador, no una cerradura: separa personas para
    poder colgarles preferencias y memoria, no autentica a nadie. Renombra
    escribiendo en la casilla. El perfil «Asistente» se aprende solo, de su
    propia voz, cada vez que habla.</p>
  </details>
  <div class="log" id="log"></div>
</div>
</main><script>
const $=i=>document.getElementById(i);
let ctx,aborto,cabeza=0;

// ---- RELLENOS Y BUFER DE REPRODUCCION ------------------------------------
// Los WAV pregenerados (scripts/perfiles.py) se bajan y se DECODIFICAN al
// cargar la pagina, una vez. Pedirlos en el momento de usarlos metria en el
// camino critico la misma latencia que vienen a tapar: son ~30 KB cada uno,
// pero decodificar y viajar cuesta decenas de ms justo cuando no sobran.
//
// EL BUFER, QUE ES LA OTRA MITAD DEL ASUNTO -- Y SE MIDIO ANTES DE TOCARLO
// El reproductor encola con reloj propio (`cabeza`): cada trozo se programa
// donde termina el anterior, y si llega TARDE se oye un hueco. La sospecha
// era que 0,15 s de margen se quedaban cortos. MEDIDO no se quedan: en una
// locucion de 8,53 s por este mismo camino (puente -> HTTP -> marcos) el
// flujo NUNCA se retrasa -- retraso maximo acumulado 0,0 ms, cero huecos con
// prebufer 0,00 -- y por el websocket directo contra la VM el peor retraso
// son 18,7 ms. El motivo es que el respiro INSERTA aire que no cuesta
// generar, y eso deja el RTF que ve el reproductor en 0,88-0,98: el flujo
// gana terreno en vez de perderlo. Subir el prebufer "por si acaso" solo
// habria retrasado el primer sonido 0,45 s en cada respuesta.
//
// Asi que el margen de verdad no se compra con latencia: lo REGALAN los
// rellenos. Mientras suena una coletilla de 0,7-2,4 s, el audio real esta
// bajando; cuando arranca, lleva ese segundo largo acumulado. De ahi que
// `cabeza` sea max(ahora + prebufer, fin del relleno) y no una cosa u otra.
const rellenos={buffers:{}, cat:{}, prebufer:0.15, listo:false};
let finRelleno=0;          // instante (reloj de ctx) en que calla el relleno
let huecos=0, huecoMs=0;   // microcortes del reproductor, acumulados
// Los perfiles de asistente, tal y como los sirve /asistentes. `perfil` es
// el activo, y cambiarlo cambia TRES cosas de golpe -- voz, coletillas y
// juego de herramientas -- que es lo que hace que suene a otro asistente.
let asistentes=null, perfil=null;
// Una sesión por pestaña: es la clave con la que el puente guarda la acción
// que está esperando un sí. Dos pestañas abiertas no se pisan la confirmación.
const sesion=(crypto.randomUUID?crypto.randomUUID():String(Math.random())).slice(0,12);
async function cargarRellenos(cual){
  try{
    if(!asistentes){
      const r=await fetch("/asistentes"); if(!r.ok) return;
      asistentes=await r.json();
      rellenos.prebufer=asistentes.prebufer_s||rellenos.prebufer;
    }
    perfil=cual||perfil||asistentes.actual;
    const p=(asistentes.perfiles||{})[perfil]; if(!p) return;
    // Los buffers van por id (hash del texto+voz), así que los de un perfil no
    // chocan con los de otro y cambiar de perfil no obliga a volver a bajar
    // los que ya estaban. Lo que se rehace es `cat`: qué ids valen AHORA.
    rellenos.cat={};
    const ac=new (window.AudioContext||window.webkitAudioContext)();
    for(const [clase,lista] of Object.entries(p.rellenos||{})){
      rellenos.cat[clase]={};
      for(const x of lista){
        try{
          if(!rellenos.buffers[x.id]){
            const b=await (await fetch(x.url)).arrayBuffer();
            rellenos.buffers[x.id]=await ac.decodeAudioData(b);
          }
          rellenos.cat[clase][x.texto]=x.id;
        }catch(_){}
      }
    }
    await ac.close();
    rellenos.listo=Object.keys(rellenos.buffers).length>0;
    if(rellenos.listo) di("perfil «"+p.nombre+"»: "+
      Object.values(rellenos.cat).reduce((n,o)=>n+Object.keys(o).length,0)+" coletillas listas");
  }catch(_){}
}
// Suena YA, en el mismo nodo de ganancia y el mismo reloj que el habla: por
// eso no puede solaparse con ella ni con otro relleno. Devuelve cuando acaba.
function sonarRelleno(actx,gan,id){
  const buf=rellenos.buffers[id]; if(!buf||!actx||actx.state==="closed") return 0;
  const src=actx.createBufferSource(); src.buffer=buf; src.connect(gan);
  // `cabeza` entra en la cuenta desde que hay rellenos que suenan DESPUES del
  // habla y no antes: 'cerrando' remata una locución cortada y 'confirmando'
  // es la respuesta entera. Sin ella, un cierre se reproducía encima de lo
  // que todavía quedaba encolado. Antes no hacía falta porque todos los
  // rellenos eran previos al primer PCM y ahí cabeza vale 0.
  const cuando=Math.max(actx.currentTime+0.02, finRelleno, cabeza);
  src.start(cuando);
  finRelleno=cuando+buf.duration;
  cabeza=Math.max(cabeza,finRelleno);
  return finRelleno;
}
for(const [r,v] of [["cfg","vcfg"],["vel","vvel"],["pasos","vpasos"]]){
  const e=$(r), o=$(v);
  e.addEventListener("input",()=>o.textContent=
    r==="pasos"?e.value:(+e.value).toFixed(2));
}
fetch("/voces").then(r=>r.json()).then(v=>{
  $("voz").innerHTML=v.map(x=>`<option>${x}</option>`).join("");
  // sp-Spk1_man por defecto, y es eleccion MEDIDA, no gusto: con la misma
  // semilla las tres voces masculinas españolas transcriben limpio (WER 8,3 %)
  // y las femeninas fallan entre el 25 % y el 66,7 %. De las tres masculinas
  // se queda Spk1 porque es ademas la voz por defecto del resto del stack
  // (VIBEVOICE_VOZ en voz-stream, narrador y compose), asi que pagina y
  // servicio dicen lo mismo. Si faltara, otra masculina española; en ultimo
  // caso, la primera española: son las unicas que pronuncian bien castellano.
  const ops=[...$("voz").options];
  const el=ops.find(o=>o.value==="sp-Spk1_man")
    ||ops.find(o=>o.value.startsWith("sp-")&&o.value.endsWith("_man"))
    ||ops.find(o=>o.value.startsWith("sp-"));
  if(el) el.selected=true;
});
fetch("/modelos").then(r=>r.json()).then(m=>{
  $("modelo").innerHTML=m.map((x,i)=>`<option${i===0?" selected":""}>${x}</option>`).join("");
});
function di(t,e){$("est").className="est"+(e?" err":"");$("est").textContent=t}
// Los rellenos, en cuanto la pagina existe. Si el puente aun los esta
// generando (primer arranque, ~30 s de VM) esto vuelve vacio y se reintenta
// una vez: no hay nada que romper, solo una mejora que llega o no llega.
cargarRellenos().then(pintarPerfil).then(()=>{
  if(!rellenos.listo) setTimeout(()=>cargarRellenos(perfil).then(pintarPerfil),45000); });

// ---- el perfil activo -----------------------------------------------------
// Hasta ahora solo se podía elegir con --asistente al arrancar el puente, o
// sea reiniciándolo. Aquí se cambia en caliente y cambian a la vez la voz, las
// coletillas y las herramientas: son las tres cosas que hacen a un asistente
// distinto de otro, y separarlas no tendría sentido.
function pintarPerfil(){
  if(!asistentes) return;
  const sel=$("perfil");
  if(!sel.options.length){
    sel.innerHTML=Object.entries(asistentes.perfiles||{})
      .map(([k,p])=>`<option value="${k}"${k===perfil?" selected":""}>${escapar(p.nombre)}</option>`).join("");
    sel.addEventListener("change",async()=>{
      await cargarRellenos(sel.value); pintarPerfil(); refrescarPendiente();
    });
  }
  const p=(asistentes.perfiles||{})[perfil]; if(!p) return;
  $("perfilNota").innerHTML=escapar(p.descripcion||"")+
    ' <span class="meta">· voz '+escapar(p.voz.voz)+', semilla '+p.voz.semilla+'</span>';
  // La voz del perfil manda sobre los mandos de abajo, pero se DEJAN tocar:
  // el perfil es un punto de partida, no una jaula.
  for(const [id,v] of [["voz",p.voz.voz],["cfg",p.voz.cfg_scale],
                       ["pasos",p.voz.pasos],["semilla",p.voz.semilla]])
    if(v!==undefined&&v!==null&&$(id)) $(id).value=v;
  for(const [r,o] of [["cfg","vcfg"],["pasos","vpasos"]])
    $(o).textContent=r==="pasos"?$(r).value:(+$(r).value).toFixed(2);
  const c=$("herrJuego");
  c.innerHTML=(p.juego||[]).map(h=>
    `<span class="hf${h.escribe?" esc":""}" title="${escapar(h.descripcion)}">`+
    `${h.escribe?"✎ ":""}${escapar(h.nombre)}</span>`).join("")||
    '<span class="hf">sin herramientas: contesta solo con lo que sabe</span>';
}
// ---- la acción que espera un sí -------------------------------------------
async function refrescarPendiente(){
  try{
    const d=await fetch("/herramientas?sesion="+sesion).then(r=>r.json());
    pintarPendiente(d.pendiente);
  }catch(_){}
}
function pintarPendiente(p){
  $("pend").hidden=!p;
  if(p) $("pendTexto").textContent="Voy a "+p.resumen+". ¿Lo hago?";
}
$("pendNo").addEventListener("click",async()=>{
  await fetch("/herramientas/cancelar",{method:"POST",
    headers:{"content-type":"application/json"},body:JSON.stringify({sesion})});
  pintarPendiente(null); di("acción cancelada: no se ha hecho nada.");
});
refrescarPendiente();

// ---- instruccion de sistema -------------------------------------------
// Vive en el NAVEGADOR (localStorage), no en el puente: asi cada navegador
// conserva la suya entre preguntas y recargas sin reiniciar el servidor.
// El puente inyecta la de serie al servir la pagina; vacia o en blanco, se
// vuelve a ella. El sufijo /no_think para Ollama lo pone el puente aparte.
const SISTEMA_DEFECTO=__SISTEMA_DEFECTO__;
$("sistema").value=localStorage.getItem("asistente_sistema")??SISTEMA_DEFECTO;
$("sistema").addEventListener("input",()=>
  localStorage.setItem("asistente_sistema",$("sistema").value));
$("sisdef").addEventListener("click",()=>{
  $("sistema").value=SISTEMA_DEFECTO;
  localStorage.removeItem("asistente_sistema");
  di("instrucción de serie restaurada.");
});

// ---- microfono -> whisper ---------------------------------------------
// Graba con MediaRecorder y manda el blob TAL CUAL al puente: voz-api pasa
// lo que llegue por ffmpeg a WAV 16k mono, asi que da igual que Chrome
// grabe webm/opus y Safari mp4/aac. El texto transcrito NO se pregunta
// solo: cae en el cuadro para poder corregirlo si whisper oyo mal.
//
// getUserMedia solo existe en contextos seguros: HTTPS o localhost. Servida
// en http://127.0.0.1 funciona; desde otra maquina de la red, no -- y se
// deshabilita el boton con el porque, para que no parezca averia.
if(!window.isSecureContext||!navigator.mediaDevices){
  $("mic").disabled=true;
  $("mic").title="el navegador solo da micrófono en HTTPS o en localhost; "+
    "desde otra máquina esta página va por HTTP y no puede grabar";
}
let grab=null,tomas=[];
$("mic").addEventListener("click",async()=>{
  if(grab&&grab.state==="recording"){grab.stop();return}
  let flujo;
  try{flujo=await navigator.mediaDevices.getUserMedia({audio:true})}
  catch(e){return di(e.name==="NotAllowedError"||e.name==="SecurityError"
    ?"micrófono denegado: dale permiso a la página en el navegador"
    :"micrófono: "+e.message,true)}
  tomas=[];grab=new MediaRecorder(flujo);
  grab.ondataavailable=e=>{if(e.data.size)tomas.push(e.data)};
  grab.onstop=async()=>{
    flujo.getTracks().forEach(t=>t.stop());
    $("mic").textContent="Hablar";$("mic").classList.remove("rec");
    di("transcribiendo…");
    try{
      const blob=new Blob(tomas,{type:grab.mimeType||"audio/webm"});
      if(blob.size<200) throw new Error("no se grabó nada");
      const r=await fetch("/stt",{method:"POST",
        headers:{"content-type":blob.type||"application/octet-stream"},body:blob});
      const d=await r.json().catch(()=>({}));
      if(!r.ok) throw new Error(d.error||("HTTP "+r.status));
      const texto=(d.texto||"").trim();
      if(!texto) return di("whisper no entendió nada; prueba otra vez",true);
      $("q").value=texto;$("q").focus();
      di("transcrito. revísalo y pulsa Preguntar.");
    }catch(e){di("transcripción: "+e.message,true)}
  };
  grab.start();
  $("mic").textContent="Parar y transcribir";$("mic").classList.add("rec");
  di("grabando… pulsa otra vez para parar");
});

let trozos=[], pendiente="";
// Se repinta entero en vez de ir parcheando nodos: son unas pocas decenas de
// spans y asi el DOM no puede desincronizarse del estado real.
function pintar(){
  const c=$("texto"); c.textContent="";
  for(const t of trozos){
    const e=document.createElement("span");
    e.className="t "+t.estado; e.dataset.id=t.id; e.textContent=t.texto+" ";
    c.appendChild(e);
  }
  if(pendiente){
    const e=document.createElement("span");
    e.className="t pend"; e.textContent=pendiente;
    c.appendChild(e);
  }
}
function marca(id,estado){
  const t=trozos.find(x=>x.id===id);
  if(t){ t.estado=estado; pintar(); }
}

// ---- una pregunta, de punta a punta -----------------------------------
// Antes esto vivia dentro del boton Preguntar. Se saca a funcion porque la
// escucha continua lanza EXACTAMENTE el mismo camino (misma sesion de voz,
// mismos colores, mismos hitos) y necesita dos cosas mas: que la promesa se
// resuelva cuando el audio ha TERMINADO de sonar, y poder abortar la
// locucion en curso cuando alguien interrumpe. El contexto de audio y el
// abortador son LOCALES (actx/ab) ademas de globales: si una interrupcion
// arranca una pregunta nueva mientras la vieja aun limpia, cada una cierra
// SU contexto y no el de la otra.
const historial=[];        // la conversacion entera: {rol, texto, quien}
let enPregunta=false, enAudio=false, preguntaEnCurso=null;
// BAJAR LA VOZ Y CALLARLA SON DOS COSAS DISTINTAS, y por eso hay un nodo de
// ganancia en medio en vez de conectar cada trozo al destino. Bajarla es
// reversible y se hace a los ~150 ms, con la sola noticia de que hay voz;
// callarla es definitivo y espera a que la huella confirme que es una
// persona (~0,5 s). Sin el nodo, lo unico que se podia hacer era cerrar el
// contexto, que es irreversible: una falsa alarma habria partido la frase.
let ganancia=null, agachado=false, ultimoDicho="";
const NIVEL_AGACHADO=0.18;   // no cero: que se siga oyendo de fondo
const CAIDA_MS=70;           // fundido de salida. De golpe suena a corte seco
function duckear(){
  if(!ganancia||agachado) return;
  agachado=true;
  ganancia.gain.setTargetAtTime(NIVEL_AGACHADO,ganancia.context.currentTime,0.03);
}
function desduckear(){
  if(!ganancia||!agachado) return;
  agachado=false;
  ganancia.gain.setTargetAtTime(1,ganancia.context.currentTime,0.05);
}
// Callar del todo: fundido corto y abortar la peticion. El contexto se cierra
// solo, con retardo, para que el fundido llegue a oirse (ver el final de
// preguntarVoz).
function silenciar(){
  if(ganancia){
    agachado=false;
    ganancia.gain.cancelScheduledValues(ganancia.context.currentTime);
    ganancia.gain.setTargetAtTime(0,ganancia.context.currentTime,CAIDA_MS/3000);
  }
  if(aborto) aborto.abort();
}
// Lo que llego a SONAR y lo que se quedo dentro. Son cosas distintas para el
// modelo: «detalla eso ultimo» habla de lo primero y «sigue» de lo segundo.
// El estado de cada trozo ya lo sabe la pagina, que es quien lo pinta.
function loQueLlevaba(){
  const dicho=[], resto=[];
  for(const t of trozos)
    (t.estado==="fin"||t.estado==="son"?dicho:resto).push(t.texto);
  if(pendiente) resto.push(pendiente);
  return {dicho:dicho.join(" ").trim(), restante:resto.join(" ").trim()};
}
// LA BARRERA CALLA SIN SABER QUE LE HAN DICHO, y eso obliga a saber volver.
// Se para con la sola noticia de que hay una persona hablando -- ese es el
// truco entero, y por eso pasa en medio segundo en vez de en tres -- pero a
// veces resulta que esa persona le hablaba a OTRA. Sin marcha atras, hablar
// cerca del asistente le dejaba mudo a mitad de frase.
//
// `corte` guarda lo que quedo por decir hasta que se sabe si la interrupcion
// era para el. Si no lo era, se retoma la locucion por donde iba (reanudar);
// si lo era, se descarta y se contesta.
let corte=null;
function reanudar(motivo,c){
  c=c||corte; corte=null;
  if(!c||!c.restante) return false;
  // El apunte cortado se quita: al terminar la reanudacion se vuelve a
  // escribir entero, sin la marca, porque al final SI lo dijo todo.
  const i=historial.findIndex(h=>h.rol==="asistente"&&h.cortado&&h.texto===c.dicho);
  if(i>=0) historial.splice(i,1);
  apunta(`<span class="meta">${escapar(motivo)}: sigo por donde iba</span>`);
  lanzarPregunta("",{decir:c.restante,continuaDe:c.dicho,sinApunte:true});
  return true;
}
// ---- las fichas de herramienta -------------------------------------------
// Misma idea que los cuatro estados por trozo: el color dice EN QUÉ FASE está
// cada herramienta ahora mismo. Una ficha por llamada, que va cambiando de
// estado en vez de apilarse, para que se lea de un vistazo qué se consultó.
const fichas={};
function herramienta(ev){
  const c=$("herrUso");
  let f=fichas[ev.id];
  if(!f){ f=document.createElement("span"); fichas[ev.id]=f; c.appendChild(f); }
  const clases={llamando:"usando",hecho:"ok",confirmar:"conf",
                error:"mal",confirmada:"ok",descartada:"mal",tope:"mal"};
  f.className="hf "+(clases[ev.fase]||"");
  const marca={llamando:"⋯",hecho:"",confirmar:"⏸",error:"✕",
               confirmada:"✔",descartada:"✖"}[ev.fase]||"";
  f.textContent=(marca?marca+" ":"")+(ev.nombre||"")+
    (ev.fase==="llamando"?"":(ev.resumen?" · "+ev.resumen.replace(/^[⏸✔✖]\s*/,""):""));
  if(f.textContent.length>110) f.textContent=f.textContent.slice(0,107)+"…";
  f.title=(ev.args?JSON.stringify(ev.args):"")+(ev.error?"  "+ev.error:"");
  if(ev.fase==="confirmar"&&ev.resumen)
    pintarPendiente({resumen:ev.resumen.replace(/^⏸\s*/,"").replace(/ — esperando el sí$/,"")});
  if(ev.fase==="confirmada"||ev.fase==="descartada") pintarPendiente(null);
}
function lanzarPregunta(q,extra){
  preguntaEnCurso=preguntarVoz(q,extra||{}).finally(()=>{preguntaEnCurso=null;});
  return preguntaEnCurso;
}
async function preguntarVoz(q,extra){
  $("ir").disabled=true; $("parar").hidden=false; enPregunta=true;
  ["h1","h2","h3","h4"].forEach(i=>$(i).textContent="—");
  trozos=[]; pendiente=""; pintar(); di("preguntando…");
  $("herrUso").textContent=""; for(const k in fichas) delete fichas[k];
  const actx=new AudioContext(); const ab=new AbortController();
  const gan=actx.createGain(); gan.connect(actx.destination);
  // El arnes de pruebas inyecta audio por el camino del servidor y NO quiere
  // oirlo salir por los altavoces. Con el nodo de ganancia sale gratis.
  if(window.__escucha&&__escucha.mudo) gan.gain.value=0;
  ctx=actx; cabeza=0; aborto=ab; ganancia=gan; agachado=false; finRelleno=0;
  // La cuenta de falsas alarmas se lleva POR LOCUCION: si en la anterior el
  // microfono oyo al altavoz dos veces, esta empieza otra vez con margen.
  barrera.falsos=0;
  let cortada=false, descartada=false;
  // La velocidad NO viaja al servidor: el websocket de sesion la rechaza a
  // proposito (ver la nota del panel). Se aplica aqui con playbackRate, que
  // es gratis pero mueve el tono. Se congela al empezar para que moverla a
  // mitad no descuadre el reloj de encolado.
  const vel=+$("vel").value;
  // Lo que el modelo tiene que LEER de este turno cuando lo dijo el programa
  // y no él. Se apunta en el historial en vez del texto para que no lo imite:
  // ver el bloque LO QUE DICE EL PROGRAMA NO ENTRA EN EL HUECO DEL MODELO en
  // scripts/herramientas.py, con los números de por qué.
  let notaTurno=null;
  const t0=performance.now(); let resto=new Uint8Array(0), primero=0, hitos={};
  const marcas=extra.marcas||{}; marcas.t0=t0;
  let respuesta="";
  // Al reanudar no hay pregunta que apuntar: nadie ha dicho nada, es la misma
  // locucion de antes que sigue.
  if(!extra.sinApunte){
    historial.push({rol:"usuario",texto:q,quien:extra.hablante||undefined});
    while(historial.length>24) historial.shift();
  }
  try{
    const r=await fetch("/preguntar",{method:"POST",signal:ab.signal,
      headers:{"content-type":"application/json"},
      body:JSON.stringify({texto:q,modelo:$("modelo").value,pensar:$("pensar").checked,
        sistema:$("sistema").value,
        historial:historial.slice(0,-1),   // lo anterior a esta pregunta
        hablante:extra.hablante||null,
        // `decir` salta el LLM y manda el texto tal cual a la voz: es el
        // «¿qué pasa?» de una interrupcion. `compuerta` pide que el veredicto
        // se resuelva AQUI, en paralelo con el LLM, en vez de en /escuchar.
        decir:extra.decir||null,
        compuerta:extra.compuerta||null,
        // El perfil y la sesión: quién contesta, y con qué confirmación a
        // medias se está hablando.
        perfil:perfil, sesion:sesion,
        voz:$("voz").value, cfg:+$("cfg").value,
        pasos:+$("pasos").value,
        semilla:$("semilla").value===""?null:+$("semilla").value})});
    if(!r.ok) throw new Error("HTTP "+r.status);
    const lector=r.body.getReader();
    // Marcos de [tipo:1][longitud:4 BE][carga]. Se acumula hasta tener el
    // marco entero: un read() puede cortar por cualquier sitio.
    while(true){
      const {done,value}=await lector.read(); if(done) break;
      let d=new Uint8Array(resto.length+value.length); d.set(resto); d.set(value,resto.length);
      let i=0;
      while(d.length-i>=5){
        const v=new DataView(d.buffer,d.byteOffset+i,5);
        const tipo=v.getUint8(0), largo=v.getUint32(1);
        if(d.length-i-5<largo) break;              // marco incompleto
        const carga=d.subarray(i+5,i+5+largo); i+=5+largo;
        if(tipo===1){
          const ev=JSON.parse(new TextDecoder().decode(carga));
          switch(ev.tipo){
            case "hito":
              if(ev.hito==="token"){ marcas.token=ev.s; $("h1").textContent=ev.s.toFixed(2)+"s"; }
              if(ev.hito==="frase"){ hitos.frase=ev.s; marcas.frase=ev.s; $("h2").textContent=ev.s.toFixed(2)+"s"; }
              if(ev.hito==="compuerta") marcas.compuerta=ev.s;
              break;
            // La compuerta, corriendo en paralelo, dijo que no era para mi:
            // ni una palabra ha salido por la voz (el puente retiene la
            // entrega hasta el veredicto). Se deshace el apunte del historial.
            case "no_dirigida":
              descartada=true; marcas.compuerta=ev.s; marcas.no_dirigida=true;
              break;
            case "token":   // el LLM escribio: solo cambia lo pendiente
              pendiente=ev.pendiente; pintar(); break;
            case "trozo":   // se cerro un trozo: pasa a tener entidad propia
              trozos.push({id:ev.id,texto:ev.texto,estado:"seg"});
              respuesta+=ev.texto+" ";
              pendiente=ev.pendiente; pintar(); break;
            case "sintetizando": marca(ev.id,"sint"); break;  // entregado a la sesion
            case "sonando":     marca(ev.id,"son");  break;  // el modelo va por aqui
            case "hecho":       marca(ev.id,"fin");  break;  // ya dicho
            // El puente dice QUE relleno y CUANDO; aqui solo se encola.
            case "relleno":
              sonarRelleno(actx,gan,ev.id);
              marcas.relleno=marcas.relleno||ev.s;
              // 'confirmando' NO es una coletilla que tape un hueco: es la
              // respuesta entera, ya grabada. Cuenta como primer sonido, y es
              // el camino más rápido que tiene esto -- ni LLM ni sesión de voz.
              if(ev.clase==="confirmando"&&!primero){
                primero=ev.s; marcas.sonido=ev.s; marcas.pregrabado=true;
                $("h3").textContent=ev.s.toFixed(2)+"s (pregrabado)";
              }
              if(!primero) di("…"+ev.texto);
              break;
            // ---- herramientas: qué se está usando y con qué resultado -----
            case "herramienta":  herramienta(ev); break;
            case "pendiente":
              if(ev.nota) notaTurno=ev.nota;
              if(ev.estado==="resuelta"||ev.estado==="descartada") pintarPendiente(null);
              break;
            case "herramientas_resumen":
              if(ev.nota) notaTurno=ev.nota;
              pintarPendiente(ev.pendiente?{resumen:ev.pendiente}:null);
              break;
            case "error":       di(ev.texto,true);   break;
          }
          continue;
        }
        const pares=carga.length-(carga.length%2);
        if(!pares) continue;
        const pcm=new Int16Array(carga.slice(0,pares).buffer);
        const f32=new Float32Array(pcm.length);
        for(let k=0;k<pcm.length;k++) f32[k]=pcm[k]/32768;
        if(!primero){ primero=(performance.now()-t0)/1000; marcas.sonido=primero;
          $("h3").textContent=primero.toFixed(2)+"s";
          if(hitos.frase) $("h4").textContent=(primero-hitos.frase).toFixed(2)+"s";
          di("hablando…"); enAudio=true;
          if(escucha.activa) estEsc("hablando","hablando");
          // EL BUFER, Y POR QUE DETRAS DEL RELLENO. `cabeza` es donde se
          // programa el primer trozo real. Dos cosas mandan: que haya
          // acumulado bastante audio para aguantar un tropiezo de red
          // (prebufer) y que no se pise con la coletilla que este sonando.
          // Cuando ha habido relleno, el prebufer sale GRATIS: mientras se
          // oia, el audio real estaba bajando.
          cabeza=Math.max(actx.currentTime+rellenos.prebufer,finRelleno); }
        const buf=actx.createBuffer(1,f32.length,24000);
        buf.copyToChannel(f32,0);
        const src=actx.createBufferSource(); src.buffer=buf; src.connect(gan);
        src.playbackRate.value=vel;
        // AQUI SE OYE EL MICROCORTE, y hasta ahora no se contaba. Si el trozo
        // llega despues de que el anterior haya terminado de sonar, el bufer
        // se agoto y entre uno y otro queda un hueco de silencio. Se apunta
        // para poder MEDIRLO desde la consola (__escucha.estado().huecos) en
        // vez de discutir si se oye o no: en las medidas de esta red salen
        // cero, y si alguna vez salen no habra que adivinar de donde vienen.
        if(cabeza<actx.currentTime){
          huecos++; huecoMs+=(actx.currentTime-cabeza)*1000;
          cabeza=actx.currentTime;
        }
        // A otra velocidad el trozo dura otra cosa: si no se divide, el
        // siguiente se encola tarde y se oye un hueco en cada empalme.
        src.start(cabeza); cabeza+=buf.duration/vel;
      }
      resto=d.subarray(i);
    }
    // Los datos acaban ANTES que el sonido: se genera mas rapido de lo que
    // se escucha (RTF < 1), asi que al terminar la descarga aun queda cola
    // encolada en Web Audio. Se avisa cuando de verdad se calla.
    const restante=Math.max(0,(cabeza-actx.currentTime)*1000);
    di(restante>200?"terminando de hablar…":"listo.");
    await new Promise(rs=>setTimeout(rs,restante+250));
    di("listo.");
  }catch(e){
    cortada=e.name==="AbortError";
    di(cortada?"parado.":"error: "+e.message,!cortada);
  }
  $("ir").disabled=false; $("parar").hidden=true;
  enPregunta=false; enAudio=false;
  // El cierre va CON RETARDO para que el fundido de silenciar() llegue a
  // sonar: cerrar el contexto corta el audio en seco, y un corte seco a mitad
  // de palabra suena a averia, no a que te esta escuchando.
  const suyo=actx; setTimeout(()=>{ try{ suyo.close(); }catch(_){} },CAIDA_MS+30);
  if(ganancia===gan){ ganancia=null; agachado=false; }
  // AL HISTORIAL, PARTIDO EN DOS. Antes iba `respuesta` entera -- todo lo que
  // el LLM habia escrito -- y eso es mentira en cuanto hay una interrupcion:
  // la otra persona solo oyo una parte, y el modelo creia haber dicho el
  // resto. Ahora se guardan las dos mitades y el LLM las ve como lo que son.
  const donde=loQueLlevaba();
  // Al reanudar, lo dicho antes del corte y lo dicho ahora son UNA respuesta:
  // se pegan y el apunte queda sin marca de corte, porque al final se dijo
  // entera y el modelo no tiene por que creer que le cortaron.
  const antes=extra.continuaDe?extra.continuaDe+" ":"";
  if(descartada){
    // No era para mi y no salio ni un byte de voz: el apunte de la pregunta
    // que se metio al empezar sobra, o la proxima respuesta arrastraria una
    // conversacion ajena.
    const i=historial.findIndex(h=>h.rol==="usuario"&&h.texto===q);
    if(i>=0) historial.splice(i,1);
  }else if(cortada&&donde.dicho){
    historial.push({rol:"asistente",texto:antes+donde.dicho,cortado:true,
                    restante:donde.restante||undefined});
    ultimoDicho=antes+donde.dicho;
    corte={dicho:antes+donde.dicho,restante:donde.restante};
  }else if(respuesta.trim()){
    historial.push({rol:"asistente",texto:antes+respuesta.trim(),
                    nota:notaTurno||undefined});
    ultimoDicho=antes+respuesta.trim();
  }
  while(historial.length>24) historial.shift();
  if(escucha.activa&&!enPregunta) estEsc("escuchando");
  return marcas;
}
$("ir").addEventListener("click",()=>{
  const q=$("q").value.trim(); if(!q) return;
  lanzarPregunta(q);
});
$("parar").addEventListener("click",()=>silenciar());

// ================== escucha continua ===================================
// El microfono SIEMPRE abierto y sin palabra de activacion. El bucle:
//
//   VAD (energia, aqui) -> POST /escuchar (huella -> whisper -> compuerta)
//   -> si iba dirigida al asistente, lanzarPregunta() con el historial
//
// REALIMENTACION -- el asistente oyendose por el altavoz -- tres capas:
//   1. la huella de voz: el puente aprende la voz que EL MISMO emite y
//      descarta lo que case con ella; mientras habla, ademas, solo deja
//      pasar voz que gane CLARAMENTE a un perfil humano. Por eso se le
//      puede interrumpir.
//   2. echoCancellation en getUserMedia: el navegador resta del microfono
//      lo que sale por el altavoz. Ayuda; con altavoces no basta sola.
//   3. sin modelo de huellas: medio duplex -- el microfono se ignora
//      mientras el asistente habla o piensa. No hay interrupcion, pero
//      tampoco puede contestarse a si mismo.
//
// VAD: RMS por bloques de 32 ms con suelo de ruido adaptativo (EMA solo
// cuando NO hay voz, para no aprenderse a si mismo como ruido). Arranca con
// ~100 ms seguidos por encima del umbral alto y cierra tras 500 ms por
// debajo del bajo.
//
// EL CIERRE: 500 ms, MEDIDO. Estaba en 600 porque las pausas internas de una
// frase parten el segmento si se cierra demasiado pronto. Se paso el VAD --
// esta misma clase, portada a Python -- por doce locuciones generadas con
// VibeVoice, del largo de una orden real, contando cuantos trozos salian:
//
//     cierre  250  300  350  400  450  500  600 ms
//     frases partidas  2    2    2    2    1    1    1  de 12
//
// La unica que se parte por debajo de 450 es «No, espera, ¿qué has dicho?»,
// que lleva dentro una pausa de 400-450 ms. De 450 a 600 no cambia NADA: los
// 100 ms de mas eran gratis para el que espera y no compraban nada. Se deja
// en 500, un escalon por encima del ultimo que fallaba.
//
// minVozMs BAJA A 150 CUANDO HAY HUELLAS. Estaba en 250 para tirar toses y
// golpes, pero se comia «¿Cómo?» -- 192 ms de voz medidos, y justo una de las
// frases con las que se interrumpe. La huella filtra eso mucho mejor: tos,
// golpe y ruido de sala dan coseno entre -0,06 y +0,04 contra CUALQUIER
// perfil (medido, ver oido.py). Sin modelo de huellas se vuelve a 250,
// porque entonces el unico filtro es este.
//
// Se antepone ademas ~400 ms de antesala para no comerse el arranque de la
// primera palabra, que el umbral solo pilla ya empezada. Esa antesala es para
// WHISPER: la barrera de interrupcion la salta a proposito (preN), porque el
// silencio diluye la huella (tabla en oido.py).
class Vad{
  constructor(al){
    this.al=al; this.rate=48000;
    this.cierreMs=500; this.preMs=400; this.minVozMs=250; this.maxMs=15000;
    this.ruido=0.004; this.resto=new Float32Array(0);
    this.enVoz=false; this.pre=[]; this.seg=[]; this.silencio=0;
    this.conVoz=0; this.arranque=0; this.preN=0;
  }
  umbrales(){ return [Math.max(0.012,this.ruido*4), Math.max(0.006,this.ruido*2.5)]; }
  alimentar(f32,rate){
    if(rate!==this.rate){ this.rate=rate; this.resto=new Float32Array(0); }
    const tam=Math.round(rate*0.032);
    let d=new Float32Array(this.resto.length+f32.length);
    d.set(this.resto); d.set(f32,this.resto.length);
    let i=0;
    for(; i+tam<=d.length; i+=tam) this._bloque(d.subarray(i,i+tam));
    this.resto=d.slice(i);
  }
  _bloque(b){
    let s=0; for(let k=0;k<b.length;k++) s+=b[k]*b[k];
    const rms=Math.sqrt(s/b.length), [alto,bajo]=this.umbrales();
    const ms=b.length/this.rate*1000;
    this.al.nivel&&this.al.nivel(rms,alto);
    if(!this.enVoz){
      this.pre.push(b.slice());
      while((this.pre.length-1)*ms>this.preMs) this.pre.shift();
      if(rms>alto){
        if((this.arranque+=ms)>=90){
          this.enVoz=true; this.seg=this.pre; this.pre=[];
          // Cuantos bloques del segmento son antesala (silencio de antes de
          // la primera palabra). La barrera se los salta; whisper no.
          this.preN=Math.max(0,this.seg.length-Math.round(this.arranque/ms));
          this.silencio=0; this.conVoz=this.arranque; this.arranque=0;
          this.al.voz&&this.al.voz(true);
        }
      }else{
        this.arranque=0;
        this.ruido=this.ruido*0.95+rms*0.05;
      }
    }else{
      this.seg.push(b.slice());
      if(rms>bajo){ this.silencio=0; this.conVoz+=ms; }
      else this.silencio+=ms;
      // Cada bloque, mientras hay voz: es el gancho de la interrupcion. No
      // espera al cierre a proposito -- esperar al cierre es justo lo que
      // hacia que el asistente siguiera hablando dos segundos largos.
      this.al.creciendo&&this.al.creciendo(this);
      if(this.silencio>=this.cierreMs||this.seg.length*ms>=this.maxMs) this._cerrar();
    }
  }
  _cerrar(){
    const bloques=this.seg; this.seg=[]; this.enVoz=false;
    this.al.voz&&this.al.voz(false);
    if(this.conVoz<this.minVozMs) return;      // un golpe, una tos: fuera
    let n=0; for(const b of bloques) n+=b.length;
    const f32=new Float32Array(n); let o=0;
    for(const b of bloques){ f32.set(b,o); o+=b.length; }
    this.al.segmento&&this.al.segmento(f32,this.rate);
  }
}

const escucha={activa:false,huellas:false,ctx:null,flujo:null,nodo:null,
               vad:null,cola:[],procesando:false,ultimaRespuesta:0,
               vozConAudio:false,agacharse:true};

// ================== la barrera: callar antes de entender ================
// Tres escalones, del mas rapido al mas listo, medidos de punta a punta en la
// pagina real (muda, con audio inyectado; ver scripts/escucha_fidelidad.py):
//
//   ~0,15 s   el VAD dice "hay voz"         -> BAJAR el volumen (reversible)
//   0,6-1,7 s la huella dice "es una persona" -> CALLAR y apuntar por donde iba
//   +0,3 s    whisper e interrupcion.clasificar -> QUE hacer con el silencio
//
// El escalon de en medio es el que decide si esto se siente natural, y es el
// que no existia: antes habia que esperar a whisper (2,1-2,9 s) para saber
// siquiera que habias hablado. Lo que hace variar ese 0,6-1,7 s no es el
// coste de la huella (10 ms) sino COMO ARRANCA la frase: «para» empieza con
// una oclusiva y el VAD la pilla en 0,26 s; un «oye» flojo tarda 0,58 s en
// pasar el umbral, y encima acumula voz mas despacio.
//
// LOS ESCALONES DE LA HUELLA, MEDIDOS (tabla completa en scripts/oido.py):
// con 0,3 s de voz acierta la mitad de las veces, con 0,5 s acierta siempre,
// y la voz del PROPIO asistente no da un solo falso positivo en ningun largo
// (0 de 41). Por eso se pregunta varias veces en vez de esperar directamente
// al largo seguro: el primer escalon cuesta 10 ms y cuando acierta ahorra
// 200 ms. Y errar por corto no rompe nada, porque el trozo sigue su camino.
const ESCALONES_BARRERA=[0.3,0.4,0.5,0.7,1.0,1.4];
// Cuantas veces se puede bajar el volumen en falso dentro de una misma
// locucion antes de dejar de hacerlo. Sin este tope, un microfono que oiga
// bien al altavoz -- eco que el navegador no cancele del todo -- convertiria
// la locucion en un bache continuo. La barrera de la huella sigue viva: solo
// se pierde el aviso temprano.
const FALSOS_AGACHE=2;
const barrera={activa:false,escalon:0,enVuelo:false,inicio:0,falsos:0,
               callo:false,ms:null};
function reiniciarBarrera(){
  barrera.activa=false; barrera.escalon=0; barrera.enVuelo=false;
}
function abrirBarrera(){
  // Solo tiene sentido con el asistente hablando y con huellas: sin timbre no
  // hay forma de distinguirle a el de quien le interrumpe.
  if(!escucha.huellas||!enAudio||!enPregunta){ reiniciarBarrera(); return; }
  barrera.activa=true; barrera.escalon=0; barrera.enVuelo=false;
  barrera.inicio=performance.now();
  if(escucha.agacharse&&barrera.falsos<FALSOS_AGACHE) duckear();
}
async function tocarBarrera(vad){
  if(!barrera.activa||barrera.enVuelo) return;
  const meta=ESCALONES_BARRERA[barrera.escalon];
  if(meta===undefined||vad.conVoz/1000<meta) return;
  barrera.enVuelo=true; barrera.escalon++;
  // SIN LA ANTESALA (seg.slice(preN)): son 400 ms de silencio de sala que
  // diluyen el vector -- medido, con ellos hace falta 0,6 s de voz para
  // acertar siempre y sin ellos 0,5 s. Whisper si la quiere; la huella no.
  const bloques=vad.seg.slice(vad.preN);
  let n=0; for(const b of bloques) n+=b.length;
  const f32=new Float32Array(n); let o=0;
  for(const b of bloques){ f32.set(b,o); o+=b.length; }
  try{
    const d=await fetch("/barrera",{method:"POST",
      headers:{"content-type":"application/json"},
      body:JSON.stringify({wav:b64(codificarWav(f32,vad.rate))})}).then(r=>r.json());
    __escucha.traza.push({fase:"barrera",escalon:meta,humano:!!d.humano,
      cos:d.cos,cos_asistente:d.cos_asistente,
      ms:Math.round(performance.now()-barrera.inicio)});
    if(d.humano&&barrera.activa&&enPregunta){
      const ms=Math.round(performance.now()-barrera.inicio);
      barrera.activa=false; barrera.falsos=0;
      barrera.callo=true; barrera.ms=ms;
      apunta(`<span class="meta">te oí y me callé en ${(ms/1000).toFixed(2)}s `+
             `(${meta}s de voz, coseno ${d.cos})</span>`);
      estEsc("te escucho","voz");
      silenciar();            // aborta la locucion; el corte queda apuntado
    }else if(barrera.escalon>=ESCALONES_BARRERA.length){
      // Se agotaron los escalones sin ver una persona: lo mas probable es que
      // sea el propio asistente colandose por el microfono. Se devuelve el
      // volumen y se deja de preguntar hasta la siguiente entrada de voz.
      barrera.activa=false; barrera.falsos++;
      desduckear();
    }
  }catch(_){ barrera.activa=false; desduckear(); }
  barrera.enVuelo=false;
}
function estEsc(txt,clase){ $("escEst").textContent=txt; $("escEst").dataset.e=clase||txt; }
function escapar(t){return String(t).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function apunta(html,clase){
  const e=document.createElement("div"); if(clase) e.className=clase;
  e.innerHTML=html; $("log").prepend(e);
  while($("log").children.length>40) $("log").lastChild.remove();
}
function codificarWav(f32,rate){
  const n=f32.length, b=new ArrayBuffer(44+n*2), v=new DataView(b);
  const s=(o,t)=>{for(let i=0;i<t.length;i++)v.setUint8(o+i,t.charCodeAt(i))};
  s(0,"RIFF"); v.setUint32(4,36+n*2,true); s(8,"WAVEfmt ");
  v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,1,true);
  v.setUint32(24,rate,true); v.setUint32(28,rate*2,true);
  v.setUint16(32,2,true); v.setUint16(34,16,true);
  s(36,"data"); v.setUint32(40,n*2,true);
  for(let i=0;i<n;i++){const x=Math.max(-1,Math.min(1,f32[i]));
    v.setInt16(44+i*2,x<0?x*32768:x*32767,true);}
  return b;
}
function b64(buf){
  let s=""; const u=new Uint8Array(buf);
  for(let i=0;i<u.length;i+=32768) s+=String.fromCharCode.apply(null,u.subarray(i,i+32768));
  return btoa(s);
}
function prepararVad(){
  escucha.vad=new Vad({
    nivel:(rms,umbral)=>{$("vui").style.width=Math.min(100,rms/(umbral*3)*100)+"%";},
    voz:v=>{
      if(v){ escucha.vozConAudio=enAudio;
             abrirBarrera();          // el primer escalon: bajar la voz
             if(escucha.activa&&!enPregunta) estEsc("voz detectada","voz"); }
      else{
        // Se acabo la entrada de voz sin que la huella confirmara a nadie:
        // devolver el volumen. Si SI confirmo, la locucion ya esta abortada y
        // esto no toca nada.
        reiniciarBarrera(); desduckear();
        if(escucha.activa&&!enPregunta&&!escucha.cola.length&&!escucha.procesando)
          estEsc("escuchando");
      }
    },
    creciendo:vad=>tocarBarrera(vad),
    segmento:(f32,rate)=>alSegmento(f32,rate)});
  // Ver el bloque de la clase Vad: con huellas, la tos la filtra el timbre
  // mucho mejor que el largo, y 250 ms se comian «¿Cómo?» (192 ms medidos).
  escucha.vad.minVozMs=escucha.huellas?150:250;
}
function alSegmento(f32,rate){
  if(!escucha.activa) return;
  // "hablando": sono voz del asistente durante ALGUNA parte del trozo. El
  // servidor endurece el filtro con eso (ver /escuchar).
  const hablando=enAudio||escucha.vozConAudio||__escucha.hablando;
  // "interrumpe": la barrera ya callo al asistente por esta misma voz. El
  // trozo no viene a preguntar nada, viene a explicar la interrupcion.
  const interrumpe=barrera.callo; barrera.callo=false;
  escucha.vozConAudio=false;
  // Repliegue sin huellas: medio duplex. Sin timbre no hay forma fiable de
  // distinguir al asistente del que interrumpe, asi que mientras hay una
  // pregunta en marcha el microfono no cuenta.
  if(!escucha.huellas&&(enPregunta||__escucha.hablando)) return;
  if(escucha.cola.length>=2) escucha.cola.shift();   // no acumular retraso
  // `fin` es cuando DEJASTE DE HABLAR, no cuando el VAD lo dio por cerrado:
  // el cierre son cierreMs de silencio DESPUES de la ultima palabra, y
  // cargarselos al LLM en la cuenta escondia medio segundo del total.
  escucha.cola.push({f32,rate,hablando,interrumpe,
                     fin:performance.now()-escucha.vad.cierreMs});
  procesarCola();
}
async function procesarCola(){
  if(escucha.procesando) return;
  escucha.procesando=true;
  while(escucha.cola.length) await procesarSegmento(escucha.cola.shift());
  escucha.procesando=false;
  if(escucha.activa&&!enPregunta) estEsc("escuchando");
}
async function procesarSegmento(s){
  const traza={};
  estEsc("¿quién habla?","proc");
  let quien=null,texto="",decision=null,error=null;
  try{
    const r=await fetch("/escuchar",{method:"POST",
      headers:{"content-type":"application/json"},
      body:JSON.stringify({wav:b64(codificarWav(s.f32,s.rate)),
        hablando:!!s.hablando, interrumpe:!!s.interrumpe,
        historial:historial.slice(-6),
        respondio_hace_s:escucha.ultimaRespuesta?
          Math.round((performance.now()-escucha.ultimaRespuesta)/100)/10:null})});
    // NDJSON incremental: cada linea es una fase y se pinta SEGUN llega.
    const lector=r.body.getReader(); const dec=new TextDecoder(); let buf="";
    while(true){
      const {done,value}=await lector.read();
      if(value) buf+=dec.decode(value,{stream:true});
      let i;
      while((i=buf.indexOf("\\n"))>=0){
        const ln=buf.slice(0,i); buf=buf.slice(i+1);
        if(!ln.trim()) continue;
        const ev=JSON.parse(ln);
        __escucha.traza.push(ev);
        if(ev.fase==="huella"){ quien=ev;
          // CALLAR AQUI, NO AL FINAL. Esta huella es la buena -- la del
          // trozo entero -- y llega en ~30-90 ms; esperar a la decision
          // costaba ademas la transcripcion. Es la red de la barrera: recoge
          // las intervenciones demasiado cortas para decidirse a media frase
          // («¿Cómo?» son 192 ms de voz y no llega ni al primer escalon).
          if(!ev.descartada){ estEsc("transcribiendo","proc");
            if(enPregunta&&aborto) silenciar(); } }
        else if(ev.fase==="texto"){ texto=ev.texto||""; traza.stt=ev.s;
          if(texto) estEsc("¿me hablan a mí?","proc"); }
        else if(ev.fase==="decision"){ decision=ev; traza.compuerta=ev.s; }
        else if(ev.fase==="fin"&&ev.error) error=ev.error;
      }
      if(done) break;
    }
  }catch(e){ error=e.message; }
  if(error){ di("escucha: "+error,true); return; }
  const nombre=quien&&quien.nombre?quien.nombre:"¿?";
  if(quien&&quien.descartada){
    apunta(`<span class="meta">descartada: ${escapar(quien.motivo||"voz del asistente")}` +
           (quien.cos!==undefined?` (coseno ${quien.cos})`:"")+`</span>`,"fuera");
    reanudar("no era una persona");
    return;
  }
  if(!texto){ apunta(`<span class="meta">(voz sin palabras)</span>`,"fuera");
              reanudar("voz sin palabras"); return; }
  const cabecera=`<span class="quien">${escapar(nombre)}</span> <span class="dicho">«${escapar(texto)}»</span>`;
  const tiempos=`huella ${((quien&&quien.s)||0).toFixed(2)}s · stt ${(traza.stt||0).toFixed(2)}s · compuerta ${(traza.compuerta||0).toFixed(2)}s`;
  // dirigida === null quiere decir "sin resolver": la compuerta corre dentro
  // de /preguntar, en paralelo con el LLM. false sigue queriendo decir que no.
  if(decision&&decision.dirigida===false){
    apunta(`${cabecera} <span class="meta">no era para mí · ${tiempos}</span>`,"fuera");
    reanudar("no era para mí");
    return;
  }
  const intencion=decision&&decision.intencion||null;
  apunta(`${cabecera} <span class="meta">`+
         (intencion?`interrupción: ${intencion}`:"para mí")+` · ${tiempos}</span>`);
  if(__escucha.sinPreguntar){       // el arnes de pruebas corta aqui
    __escucha.traza.push({fase:"preguntaria",texto,intencion,
                          hablante:quien&&quien.perfil?nombre:null});
    if(escucha.activa) estEsc("escuchando");
    return;
  }
  // Ya se callo arriba, al llegar la huella. Aqui solo se espera a que la
  // pregunta vieja LIMPIE (cierra su contexto de audio) antes de lanzar la
  // nueva; son milisegundos y evita pisarse los globales.
  if(enPregunta&&aborto){ silenciar(); if(preguntaEnCurso) await preguntaEnCurso; }

  // ---- que hacer con el silencio que se acaba de abrir ------------------
  // Las ordenes hechas se responden SIN LLM y SIN compuerta. Es la diferencia
  // entre «espera» -> «¿qué pasa?» en menos de un segundo, y en cinco.
  const hablante=quien&&quien.perfil?nombre:null;
  // El corte se aparta AQUI: si se contesta, se descarta; si la compuerta
  // acaba diciendo que no era para el, se retoma con esta copia. Dejarlo en
  // la global no vale, porque la respuesta que se lanza ahora la pisaria.
  const cortePrevio=corte; corte=null;
  if(intencion==="parar"){
    apunta(`<span class="meta">callado. dime cuando quieras.</span>`);
    historial.push({rol:"usuario",texto,quien:hablante||undefined});
    while(historial.length>24) historial.shift();
    if(escucha.activa) estEsc("escuchando");
    return;
  }
  let extra={hablante,marcas:{}};
  // La frase fija viene del servidor (interrupcion.que_decir), no de aqui:
  // asi el texto que dice el asistente vive en un solo sitio.
  if(decision&&decision.decir) extra.decir=decision.decir;
  else if(intencion==="repetir"){
    if(!ultimoDicho){ apunta(`<span class="meta">no había dicho nada aún</span>`,"fuera");
      if(escucha.activa) estEsc("escuchando"); return; }
    extra.decir=ultimoDicho;
  }else if(intencion==="seguir"){
    // SI LO QUE QUEDABA DA PARA ALGO, se dice tal cual: es literalmente
    // seguir por donde iba, y suena al instante en vez de esperar 1-6 s a que
    // el LLM redacte una continuacion parecida. El tope de 60 letras es para
    // no rematar con un cabo suelto de tres palabras; por debajo de eso vale
    // mas que lo escriba el modelo, que ve el corte en el historial.
    if(cortePrevio&&cortePrevio.restante&&cortePrevio.restante.length>=60)
      extra.decir=cortePrevio.restante;
    else texto="Sigue por donde ibas.";
  }else{
    // Ni orden hecha ni nada: la compuerta decide, en paralelo con el LLM.
    extra.compuerta={historial:historial.slice(-6),hablante,
      respondio_hace_s:escucha.ultimaRespuesta?
        Math.round((performance.now()-escucha.ultimaRespuesta)/100)/10:null};
  }
  estEsc(extra.decir?"contestando":"pensando","pensando");
  const marcas=extra.marcas;
  // NO se espera al final del audio: los trozos que el VAD saque mientras
  // el asistente habla se procesan (asi es como se le puede interrumpir).
  lanzarPregunta(texto,extra).then(()=>{
    escucha.ultimaRespuesta=performance.now();
    if(marcas.no_dirigida){
      apunta(`<span class="meta">no era para mí (compuerta `+
             `${(marcas.compuerta||0).toFixed(2)}s, en paralelo: no sonó nada)</span>`,"fuera");
      reanudar("no era para mí",cortePrevio);
    }else if(marcas.sonido!==undefined){
      const total=(marcas.t0+marcas.sonido*1000-s.fin)/1000;
      apunta(`<span class="meta">dejas de hablar → primer sonido: ${total.toFixed(2)}s `+
             `(cierre del VAD ${(escucha.vad.cierreMs/1000).toFixed(2)} `+
             `+ stt ${(traza.stt||0).toFixed(2)}`+
             (marcas.compuerta!==undefined?` + compuerta ${marcas.compuerta.toFixed(2)} (solapada)`:"")+
             ` + LLM y voz ${(marcas.sonido||0).toFixed(2)})</span>`);
    }
    if(escucha.activa&&!enPregunta) estEsc("escuchando");
  });
}
async function comprobarHuellas(){
  try{
    const d=await fetch("/perfiles").then(r=>r.json());
    escucha.huellas=!!d.disponible;
    if(d.cargando) setTimeout(comprobarHuellas,4000);   // el modelo aun carga
    pintarPerfiles(d);
  }catch(_){ escucha.huellas=false; }
}
if(!window.isSecureContext||!navigator.mediaDevices){
  $("esc").disabled=true;
  $("esc").title="el navegador solo da micrófono en HTTPS o en localhost; "+
    "desde otra máquina esta página va por HTTP y no puede escuchar";
}
$("esc").addEventListener("click",async()=>{
  if(escucha.activa) return apagarEscucha();
  let flujo;
  try{
    // echoCancellation es la segunda capa contra la realimentacion: el
    // navegador resta del microfono lo que el mismo esta reproduciendo.
    flujo=await navigator.mediaDevices.getUserMedia({audio:{
      echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
  }catch(e){ return di(e.name==="NotAllowedError"||e.name==="SecurityError"
    ?"micrófono denegado: dale permiso a la página en el navegador"
    :"micrófono: "+e.message,true); }
  await comprobarHuellas();
  prepararVad();
  const c=new AudioContext();
  const src=c.createMediaStreamSource(flujo);
  // ScriptProcessor y no AudioWorklet: esta senalado como obsoleto pero
  // funciona en todos los navegadores sin servir un modulo aparte, y 2048
  // muestras (43 ms a 48 kHz) sobran para un VAD de energia. La salida del
  // nodo son ceros -- no se reproduce nada -- pero hay que conectarlo al
  // destino o el navegador no lo hace correr.
  const nodo=c.createScriptProcessor(2048,1,1);
  nodo.onaudioprocess=e=>{
    if(!escucha.activa) return;
    escucha.vad.alimentar(new Float32Array(e.inputBuffer.getChannelData(0)),c.sampleRate);
  };
  src.connect(nodo); nodo.connect(c.destination);
  Object.assign(escucha,{activa:true,ctx:c,flujo,nodo});
  $("esc").textContent="Apagar escucha continua";
  $("mic").disabled=true; $("mic").title="la escucha continua ya usa el micrófono";
  estEsc("escuchando");
  di(escucha.huellas?"escucha continua activa; háblame cuando quieras."
    :"escucha continua SIN huellas de voz (falta el modelo): no sabré quién "+
     "habla y me quedaré sordo mientras hablo yo, en medio dúplex.");
});
function apagarEscucha(){
  escucha.activa=false;
  try{escucha.nodo&&escucha.nodo.disconnect();}catch(_){}
  try{escucha.flujo&&escucha.flujo.getTracks().forEach(t=>t.stop());}catch(_){}
  try{escucha.ctx&&escucha.ctx.close();}catch(_){}
  escucha.cola.length=0;
  $("esc").textContent="Activar escucha continua";
  if(window.isSecureContext&&navigator.mediaDevices){
    $("mic").disabled=false; $("mic").title="";
  }
  estEsc("apagada"); $("vui").style.width="0";
  di("escucha continua apagada.");
}

// ---- perfiles: quien es quien -----------------------------------------
async function pintarPerfiles(d){
  if(!d){ try{ d=await fetch("/perfiles").then(r=>r.json()); }catch(_){ return; } }
  const c=$("perfLista"); c.textContent="";
  if(d.error&&!(d.perfiles||[]).length){
    c.innerHTML=`<span class="meta">huellas no disponibles: ${escapar(d.error)}</span>`;
    return;
  }
  for(const p of d.perfiles||[]){
    const fila=document.createElement("div"); fila.className="perfil";
    const inp=document.createElement("input"); inp.value=p.nombre;
    inp.addEventListener("change",async()=>{
      await fetch("/perfiles/renombrar",{method:"POST",
        headers:{"content-type":"application/json"},
        body:JSON.stringify({id:p.id,nombre:inp.value})});
      di("perfil renombrado.");
    });
    const t=document.createElement("span"); t.className="tipo";
    t.textContent=(p.tipo==="asistente"?"el asistente":
                   p.tipo==="desconocido"?"sin nombre aún":"persona")+
                  " · "+p.muestras+(p.muestras===1?" muestra":" muestras");
    fila.append(inp,t); c.appendChild(fila);
  }
  if(!(d.perfiles||[]).length)
    c.innerHTML='<span class="meta">sin perfiles todavía: graba el primero '+
                'abajo, o simplemente habla con la escucha activa</span>';
}
pintarPerfiles();
$("perfAlta").addEventListener("click",async()=>{
  const nombre=$("perfNombre").value.trim();
  if(!nombre) return di("ponle un nombre al perfil antes de grabar",true);
  let flujo;
  try{ flujo=await navigator.mediaDevices.getUserMedia({audio:true}); }
  catch(e){ return di("micrófono: "+e.message,true); }
  di("grabando 5 s para el perfil de "+nombre+"… habla con normalidad");
  const c=new AudioContext(); const src=c.createMediaStreamSource(flujo);
  const nodo=c.createScriptProcessor(2048,1,1); const tomas=[];
  nodo.onaudioprocess=e=>tomas.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(nodo); nodo.connect(c.destination);
  await new Promise(r=>setTimeout(r,5000));
  const rate=c.sampleRate;
  try{nodo.disconnect();src.disconnect();}catch(_){}
  flujo.getTracks().forEach(t=>t.stop()); c.close();
  const n=tomas.reduce((s,t)=>s+t.length,0);
  const f32=new Float32Array(n); let o=0;
  for(const t of tomas){ f32.set(t,o); o+=t.length; }
  di("guardando la huella…");
  const r=await fetch("/perfiles/matricular",{method:"POST",
    headers:{"content-type":"application/json"},
    body:JSON.stringify({nombre,wav:b64(codificarWav(f32,rate))})});
  const dd=await r.json().catch(()=>({}));
  if(dd.error) return di(dd.error,true);
  di("perfil de "+nombre+" guardado ("+dd.muestras+
     (dd.muestras===1?" muestra":" muestras")+").");
  pintarPerfiles();
});

// ---- ganchos de prueba -------------------------------------------------
// Para probar el bucle SIN microfono ni altavoces: inyectan PCM s16 por el
// MISMO camino que el microfono (VAD -> barrera -> /escuchar -> compuerta), y
// sinPreguntar corta justo antes del LLM grande y de la voz. Los usa el
// arnes de pruebas del repo; a la pagina no le estorban.
//
// `mudo` es lo que permite medir la INTERRUPCION de verdad -- pidiendo una
// respuesta larga y hablandole encima -- sin que salga un solo sonido por los
// altavoces: pone a cero el nodo de ganancia por el que ya pasa todo el
// audio. Nada mas cambia, asi que los tiempos son los mismos que con volumen.
// `inyectarEnVivo` alimenta el VAD en TIEMPO REAL (a ritmo de reloj, no de
// golpe) para que los milisegundos que mide la barrera signifiquen algo.
window.__escucha={traza:[],sinPreguntar:false,hablando:false,mudo:false,
  interno:escucha,barrera,
  armar(){ if(!escucha.vad) prepararVad();
           escucha.activa=true; escucha.huellas=true; estEsc("escuchando"); },
  _aF32(cad){
    const crudo=atob(cad), n=crudo.length>>1, f=new Float32Array(n);
    for(let i=0;i<n;i++){
      let v=crudo.charCodeAt(2*i)|(crudo.charCodeAt(2*i+1)<<8);
      if(v>=32768) v-=65536;
      f[i]=v/32768;
    }
    return f;
  },
  inyectarB64(cad,rate){ escucha.vad.alimentar(this._aF32(cad),rate||16000); },
  async inyectarEnVivo(cad,rate){
    rate=rate||16000;
    const f=this._aF32(cad), paso=Math.round(rate*0.032);
    barrera.ms=null;
    const t0=performance.now();
    for(let i=0;i<f.length;i+=paso){
      escucha.vad.alimentar(f.subarray(i,Math.min(i+paso,f.length)),rate);
      const debe=t0+(i+paso)/rate*1000;
      const falta=debe-performance.now();
      if(falta>0) await new Promise(r=>setTimeout(r,falta));
    }
    return {t0,callarMs:barrera.ms};
  },
  estado(){ return {chip:$("escEst").textContent,activa:escucha.activa,
    cola:escucha.cola.length,procesando:escucha.procesando,
    enVoz:escucha.vad?escucha.vad.enVoz:false,huellas:escucha.huellas,
    enPregunta,enAudio,agachado,callarMs:barrera.ms,
    ganancia:ganancia?ganancia.gain.value:null,
    // Los microcortes del reproductor y el catalogo cargado, para poder
    // medirlos desde la consola en vez de fiarse del oido.
    huecos,huecoMs:Math.round(huecoMs),
    rellenos:{listo:rellenos.listo,prebufer:rellenos.prebufer,
              n:Object.keys(rellenos.buffers).length},
    historial:historial.slice(-4)}; }};
</script></body></html>"""


# --------------------------------------------------------------------------
# PERFILES DE ASISTENTE Y RELLENOS
#
# El core esta en scripts/perfiles.py (formato, cache por hash, politica de
# cuando suena cada cosa). Aqui solo queda lo que es propio del puente:
# arrancar la generacion sin bloquear, servir los WAV y disparar los eventos.
#
# POR QUE LA GENERACION VA EN UN HILO Y NO EN main()
# Sintetizar 29 rellenos son ~30 s de VM la primera vez. Bloquear el arranque
# del puente por eso significaria que el asistente no responde durante medio
# minuto por una MEJORA, y ademas que si la VM esta apagada no arranca nunca.
# En un hilo: la pagina levanta al instante y los rellenos aparecen cuando
# aparecen. Los arranques siguientes no sintetizan nada (la cache va por hash
# del contenido), asi que el hilo termina en milisegundos.
#
# Y SI LA VOZ NO ESTA, NO PASA NADA: perfiles.generar() no lanza, apunta los
# fallos y devuelve lo que tenga. El asistente funciona sin rellenos.
_ASISTENTES = {"datos": None, "manifiesto": None, "catalogos": {}, "error": None}


def arrancar_perfiles(ruta, cache, url, token, generar=True) -> None:
    """Carga el fichero de perfiles y, en segundo plano, genera lo que falte."""
    try:
        datos = perfiles.cargar(ruta)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        _ASISTENTES["error"] = f"{type(e).__name__}: {e}"
        print(f"[perfiles] no se pudo leer el fichero: {e}", flush=True)
        return
    _ASISTENTES["datos"] = datos
    dir_cache = perfiles.directorio_cache(cache)
    CFG["rellenos_dir"] = str(dir_cache)

    def faena():
        try:
            if generar:
                m = perfiles.generar(datos, url, token, dir_cache)
            else:
                ruta_m = dir_cache / "catalogo.json"
                m = json.loads(ruta_m.read_text()) if ruta_m.exists() else \
                    {"perfiles": {}, "fallos": []}
        except Exception as e:                       # noqa: BLE001
            _ASISTENTES["error"] = f"{type(e).__name__}: {e}"
            print(f"[perfiles] generacion fallida: {e}", flush=True)
            return
        _ASISTENTES["manifiesto"] = m
        _ASISTENTES["catalogos"] = {
            n: perfiles.Catalogo(m, n, dir_cache) for n in datos["perfiles"]}

    threading.Thread(target=faena, daemon=True).start()


def estado_asistentes() -> dict:
    """Lo que la pagina necesita: quien es cada perfil y que WAV precargar."""
    datos = _ASISTENTES["datos"]
    if datos is None:
        return {"listo": False, "error": _ASISTENTES["error"], "perfiles": {}}
    m = _ASISTENTES["manifiesto"] or {"perfiles": {}}
    salida = {}
    for nombre, p in datos["perfiles"].items():
        cats = (m["perfiles"].get(nombre) or {}).get("categorias", {})
        # Las herramientas CONCRETAS que le tocan a este perfil, no solo los
        # bloques declarados: es lo que la pagina pinta para que se vea de un
        # vistazo que puede y que no puede hacer cada asistente, y sobre todo
        # cuales de ellas escriben.
        # Con --sin-herramientas la lista va VACIA aunque el perfil las
        # declare: la pagina tiene que enseñar lo que este asistente puede
        # hacer AHORA, no lo que podria si estuvieran encendidas.
        dominios = perfiles.dominios_de(p) if HERRAMIENTAS is not None else []
        juego = [{"nombre": h["nombre"], "dominio": h["dominio"],
                  "escribe": h["escribe"], "descripcion": h["descripcion"]}
                 for h in herramientas.CATALOGO
                 if h["nombre"] in herramientas.por_dominios(dominios)]
        salida[nombre] = {
            "nombre": p.get("nombre", nombre),
            "descripcion": p.get("descripcion", ""),
            "voz": p["voz"],
            "sistema": p.get("sistema", ""),
            "vocabulario": p.get("vocabulario", {}),
            "dominios": dominios,
            "juego": juego,
            "herramientas": [{"id": h["id"], "tipo": h["tipo"],
                              "habilitada": bool(h.get("habilitada")),
                              "descripcion": h.get("descripcion", "")}
                             for h in p.get("herramientas") or []],
            "rellenos": {c: [{"id": x["id"], "ms": x["ms"], "texto": x["texto"],
                              "url": f"/rellenos/{x['fichero']}"}
                             for x in v] for c, v in cats.items()},
        }
    return {"listo": _ASISTENTES["manifiesto"] is not None,
            "error": _ASISTENTES["error"],
            "actual": CFG.get("perfil") or datos.get("perfil_por_defecto"),
            "por_defecto": datos.get("perfil_por_defecto"),
            "umbrales": perfiles.UMBRALES,
            "prebufer_s": CFG.get("prebufer", 0.15),
            "fallos": (_ASISTENTES["manifiesto"] or {}).get("fallos", []),
            "perfiles": salida}


def _multipart(campos, nombre_fichero, datos, tipo="audio/wav"):
    lim = "----" + uuid.uuid4().hex
    cuerpo = b""
    for k, v in campos.items():
        cuerpo += (f'--{lim}\r\nContent-Disposition: form-data; name="{k}"'
                   f'\r\n\r\n{v}\r\n').encode()
    cuerpo += (f'--{lim}\r\nContent-Disposition: form-data; name="{nombre_fichero}"; '
               f'filename="voz.wav"\r\nContent-Type: {tipo}\r\n\r\n').encode()
    cuerpo += datos + f"\r\n--{lim}--\r\n".encode()
    return f"multipart/form-data; boundary={lim}", cuerpo


def stt_nativo(url, wav, prompt=""):
    """(texto, error) contra un whisper.cpp NATIVO, sin voz-api en medio.

    El camino de siempre es navegador -> puente -> voz-api -> ffmpeg ->
    whisper en Docker. Aqui se cortan los dos saltos de en medio: la pagina ya
    construye WAV s16 y whisper.cpp lo lee solo (miniaudio remuestrea; se
    comprobo con el mismo audio a 16, 22 y 48 kHz -- transcripcion identica),
    asi que ffmpeg no pinta nada.

    Lo que SI se conserva de voz-api es el sesgo de vocabulario: sin el,
    whisper transcribe "WireGuard" como "We The War". Viaja como `prompt`,
    igual que hace voz-api.
    """
    campos = {"language": "es", "temperature": "0.0", "response_format": "json"}
    if prompt:
        campos["prompt"] = prompt
    tipo, cuerpo = _multipart(campos, "file", wav)
    pet = urllib.request.Request(f"{url}/inference", method="POST", data=cuerpo,
                                 headers={"content-type": tipo})
    try:
        d = json.load(urllib.request.urlopen(pet, timeout=120))
    except Exception as e:
        return None, (f"whisper nativo no responde en {url}/inference "
                      f"({type(e).__name__}: {e}); levantalo con "
                      f"scripts/whisper-mac.sh")
    return (d.get("text") or "").strip(), None


def desmarcar(buf, al_pcm, al_evento):
    """Saca los marcos completos de `buf` y devuelve lo que sobra.

    El marco del servicio de voz es el MISMO que este puente le manda al
    navegador -- [tipo:1][longitud:4 BE][carga] --, asi que el PCM se reenvia
    tal cual, sin volver a empaquetarlo. Un mensaje de websocket ya llega
    entero y trae un marco justo; se acumula igualmente para no depender de eso.
    """
    i = 0
    while len(buf) - i >= 5:
        tipo, largo = struct.unpack(">BI", buf[i:i + 5])
        if len(buf) - i - 5 < largo:
            break
        carga = buf[i + 5:i + 5 + largo]
        i += 5 + largo
        if tipo == 0:
            al_pcm(carga)
        elif tipo == 1:
            al_evento(json.loads(carga.decode()))
    return buf[i:]


class Puente(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            # Lo unico del servidor que la pagina necesita de serie es la
            # instruccion de sistema por defecto (para arrancar con ella y para
            # el boton de volver): se inyecta aqui como literal JSON y no hace
            # falta un endpoint mas ni una segunda copia del texto.
            cuerpo = PAGINA.replace(
                "__SISTEMA_DEFECTO__",
                json.dumps(CFG["sistema"], ensure_ascii=False)).encode()
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)
        elif self.path == "/voces":
            # Se pregunta al sintetizador en vez de mantener una lista aqui:
            # una copia local se queda vieja en cuanto cambian las voces.
            try:
                pet = urllib.request.Request(f"{CFG['voz_url']}/voces")
                if CFG["token"]:
                    pet.add_header("authorization", f"Bearer {CFG['token']}")
                vs = json.load(urllib.request.urlopen(pet, timeout=5))["voces"]
            except Exception:
                vs = [CFG["voz"]]
            # Espanolas delante: son las unicas que pronuncian bien el castellano.
            vs.sort(key=lambda v: (not v.startswith("sp-"), v))
            cuerpo = json.dumps(vs).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)
        elif self.path == "/modelos":
            # Los de MiniMax van primero: son los rapidos. Detras, lo que haya
            # en Ollama, que puede no estar arrancado y no debe romper la lista.
            ms = list(MINIMAX_MODELOS)
            try:
                d = json.load(urllib.request.urlopen(f"{CFG['ollama']}/api/tags", timeout=5))
                ms += sorted(m["name"] for m in d.get("models", []))
            except Exception:
                pass
            if CFG["modelo"] not in ms:
                ms.insert(0, CFG["modelo"])
            cuerpo = json.dumps(ms).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)
        elif self.path == "/perfiles":
            # `disponible` es si el modelo de huellas esta CARGADO. Tarda ~6 s
            # en un hilo al arrancar; la pagina vuelve a preguntar si le sale
            # que no, en vez de darlo por perdido para toda la sesion.
            if OIDO is None:
                cuerpo = {"disponible": False, "cargando": False,
                          "error": "sin modulo de huellas", "perfiles": []}
            else:
                est = OIDO.estado()
                cuerpo = {"disponible": est["disponible"],
                          "cargando": not est["disponible"] and not est["error"],
                          "error": est["error"], "perfiles": OIDO.lista()}
            cuerpo = json.dumps(cuerpo, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)
        elif self.path.startswith("/herramientas"):
            # EL CATALOGO Y LO QUE HAY A MEDIAS. La pagina lo pinta para que se
            # vea sin abrir un fichero que puede hacer cada perfil, cuales de
            # esas cosas ESCRIBEN, y si hay una accion esperando un si.
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            ses = (q.get("sesion") or ["web"])[0]
            p = HERRAMIENTAS.pendiente(ses) if HERRAMIENTAS else None
            self.responder_json(200, {
                "disponible": HERRAMIENTAS is not None,
                "datos": str(HERRAMIENTAS.sim.ruta) if HERRAMIENTAS else None,
                "catalogo": [{"nombre": h["nombre"], "dominio": h["dominio"],
                              "escribe": h["escribe"],
                              "descripcion": h["descripcion"],
                              "sustituir_por": h["sustituir_por"]}
                             for h in herramientas.CATALOGO],
                "pendiente": ({"nombre": p["nombre"], "resumen": p["resumen"]}
                              if p else None),
                "ultimas": (HERRAMIENTAS.registro[-8:] if HERRAMIENTAS else [])})
        elif self.path == "/asistentes":
            # LOS PERFILES DE ASISTENTE, que no son los de VOZ. /perfiles ya
            # estaba cogido por las huellas de oido.py y son cosas distintas:
            # alli "perfil" es una persona a la que reconocer, aqui es una
            # personalidad con su voz, su vocabulario y sus rellenos
            # (scripts/perfiles.py). De aqui saca la pagina la lista de WAV
            # que tiene que precargar y decodificar al abrirse.
            self.responder_json(200, estado_asistentes())
        elif self.path.startswith("/rellenos/"):
            # El WAV pregenerado, tal cual sale de la cache. La pagina los pide
            # UNA vez al cargarse y los guarda ya decodificados: pedirlos en el
            # momento de necesitarlos metria en el camino critico justo la
            # latencia que vienen a tapar.
            nombre = os.path.basename(urllib.parse.unquote(self.path[10:]))
            base = CFG.get("rellenos_dir")
            ruta = os.path.join(base, nombre) if base else None
            if (not ruta or not nombre.endswith(".wav")
                    or not os.path.isfile(ruta)):
                return self.send_error(404)
            datos = open(ruta, "rb").read()
            self.send_response(200)
            self.send_header("content-type", "audio/wav")
            self.send_header("content-length", str(len(datos)))
            self.send_header("cache-control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(datos)
        else:
            self.send_error(404)

    def responder_json(self, codigo, obj):
        cuerpo = json.dumps(obj).encode()
        self.send_response(codigo)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def _stt(self, audio: bytes, tipo: str):
        """(texto, error): reenvia audio al /stt de la API de voz (whisper).

        El navegador manda el blob del MediaRecorder tal cual -- webm/opus en
        Chrome, mp4/aac en Safari -- porque voz-api pasa lo que llegue por
        ffmpeg a WAV 16k mono antes de darselo a whisper: recodificar aqui
        seria trabajo repetido. La peticion multipart es calcada a la de
        transcribir() en scripts/fidelidad.py, que es la referencia probada.
        """
        # WHISPER NATIVO SOLO PARA WAV. El boton de microfono manda lo que
        # grabe MediaRecorder -- webm/opus en Chrome, mp4/aac en Safari -- y
        # eso necesita el ffmpeg de voz-api. La escucha continua, en cambio,
        # construye WAV s16 ella misma y puede ir por el camino corto.
        base = tipo.split(";")[0].strip()
        if CFG.get("whisper") and base in ("audio/wav", "audio/x-wav"):
            return stt_nativo(CFG["whisper"], audio, CFG.get("prompt_stt", ""))
        # La extension del nombre es cosmetica (ffmpeg huele el contenido),
        # pero que al menos no mienta para los formatos conocidos.
        ext = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/mp4": "mp4",
               "audio/mpeg": "mp3", "audio/ogg": "ogg"}.get(base, "webm")
        lim = "----" + uuid.uuid4().hex
        cuerpo = (f'--{lim}\r\nContent-Disposition: form-data; '
                  f'name="idioma"\r\n\r\nes\r\n'
                  f'--{lim}\r\nContent-Disposition: form-data; name="archivo"; '
                  f'filename="voz.{ext}"\r\nContent-Type: {tipo}\r\n\r\n').encode()
        cuerpo += audio + f"\r\n--{lim}--\r\n".encode()
        pet = urllib.request.Request(
            f"{CFG['voz_api']}/stt", method="POST", data=cuerpo,
            headers={"content-type": f"multipart/form-data; boundary={lim}",
                     **({"authorization": f"Bearer {CFG['token_api']}"}
                        if CFG["token_api"] else {})})
        try:
            d = json.load(urllib.request.urlopen(pet, timeout=120))
        except Exception as e:
            # El motivo en claro: "whisper no responde" a secas obliga a ir a
            # mirar el terminal del puente, y el navegador ya esta abierto.
            return None, (f"whisper no responde en {CFG['voz_api']}/stt "
                          f"({type(e).__name__}: {e}); ¿esta levantado? "
                          f"cd docker && docker compose up -d whisper voz-api")
        return d.get("texto", ""), None

    def transcribir(self):
        n = int(self.headers.get("content-length", 0))
        audio = self.rfile.read(n) if n else b""
        if not audio:
            return self.responder_json(400, {"error": "sin audio"})
        tipo = self.headers.get("content-type", "application/octet-stream")
        texto, err = self._stt(audio, tipo)
        if err:
            return self.responder_json(502, {"error": err})
        self.responder_json(200, {"texto": texto})

    def barrera(self):
        """¿Hay una persona hablando AHORA MISMO? Solo la huella, ~10 ms.

        Es el camino corto de la interrupcion, y a proposito no hace nada
        mas: nada de whisper, nada de compuerta, nada de tocar perfiles. La
        pagina lo llama a media palabra -- con 0,35, 0,5 y 0,7 s de voz
        grabada -- mientras el asistente habla, y en cuanto contesta que si,
        calla. El resto (que dijiste, y que hacer con ello) llega despues por
        /escuchar, cuando el VAD cierre la frase.

        Un `humano: false` NO quiere decir "es el asistente": quiere decir
        "todavia no lo se". Errar por corto solo retrasa la interrupcion al
        camino normal; nunca calla al asistente por equivocacion.
        """
        n = int(self.headers.get("content-length", 0))
        try:
            pet = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self.responder_json(400, {"error": "peticion ilegible"})
        if OIDO is None or not OIDO.listo():
            # Sin huellas no hay barrera posible; la pagina ya lo sabe por
            # /perfiles y se repliega a medio duplex, pero que no reviente.
            return self.responder_json(200, {"humano": False,
                                             "motivo": "sin huellas"})
        try:
            wav = base64.b64decode(pet.get("wav") or "")
        except (ValueError, TypeError):
            wav = b""
        if len(wav) < 100:
            return self.responder_json(400, {"error": "sin audio"})
        self.responder_json(200, OIDO.barrera(wav))

    def escuchar(self):
        """Una intervencion oida por la escucha continua, de punta a punta.

        Entra JSON {wav: base64, hablando: bool, historial, respondio_hace_s}
        y sale NDJSON con una linea por fase, SEGUN OCURREN -- la pagina pinta
        cada estado en cuanto pasa, no al final:

            {"fase":"huella", ...}    quien habla, o descartada (~ms)
            {"fase":"texto", ...}     lo que dijo whisper
            {"fase":"decision", ...}  si iba dirigida al asistente
            {"fase":"fin", ...}

        El ORDEN de las fases es la optimizacion: la huella cuesta ~25 ms y
        descarta la propia voz del asistente ANTES de pagar la transcripcion
        entera de whisper y la llamada de la compuerta.

        LA COMPUERTA YA NO SE PAGA SIEMPRE. Dos atajos, los dos medidos:
          - Si el texto es una orden hecha de interrupcion -- «espera»,
            «para», «sigue», «¿qué has dicho?» -- se resuelve aqui mismo, en
            microsegundos, y no se llama a nadie. La compuerta cuesta
            0,47-0,70 s y sobre esas frases no aporta: son inequivocas.
          - Si no lo es, la compuerta NO se llama aqui: se manda `dirigida`
            sin resolver y la resuelve /preguntar EN PARALELO con el LLM, que
            es donde deja de costar tiempo. Con --sin-solapar vuelve aqui.
        """
        n = int(self.headers.get("content-length", 0))
        try:
            pet = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            pet = {}
        self.send_response(200)
        self.send_header("content-type", "application/x-ndjson")
        self.send_header("cache-control", "no-store")
        # Mismo motivo que en /preguntar: sin longitud conocida, el fin de la
        # respuesta es el fin de la conexion.
        self.send_header("connection", "close")
        self.close_connection = True
        self.end_headers()

        def linea(**kw):
            self.wfile.write((json.dumps(kw, ensure_ascii=False) + "\n").encode())
            self.wfile.flush()

        try:
            try:
                wav = base64.b64decode(pet.get("wav") or "")
            except (ValueError, TypeError):
                wav = b""
            if len(wav) < 100:
                return linea(fase="fin", error="sin audio")
            hablando = bool(pet.get("hablando"))
            if OIDO is not None:
                quien = OIDO.identificar(wav, hablando=hablando)
            else:
                # Sin huellas no hay forma de reconocer la propia voz: si el
                # asistente esta hablando se descarta todo (medio duplex de
                # servidor, por si la pagina no lo aplico ya).
                quien = {"descartada": hablando, "perfil": None,
                         "motivo": "perfiles de voz no disponibles"}
            linea(fase="huella", **quien)
            if quien.get("descartada"):
                return linea(fase="fin", descartada=True)

            t0 = time.perf_counter()
            texto, err = self._stt(wav, "audio/wav")
            if err:
                return linea(fase="fin", error=err)
            texto = (texto or "").strip()
            # La intencion de interrupcion se mira SIEMPRE, no solo cuando el
            # asistente esta hablando: un «para» dicho justo cuando acaba de
            # callarse tambien es un «para», y responderle «¿qué pasa?» a un
            # «espera» es lo suyo aunque llegue medio segundo tarde.
            intencion = clasificar(texto) if texto else None
            linea(fase="texto", texto=texto, intencion=intencion,
                  interrumpe=bool(pet.get("interrumpe")),
                  s=round(time.perf_counter() - t0, 3))
            if not texto:
                return linea(fase="fin", vacia=True)

            if intencion is not None:
                # Frase hecha: ni compuerta ni LLM. Ver la cabecera de
                # scripts/interrupcion.py. La frase que hay que decir la pone
                # interrupcion.py y no la pagina, para que el texto viva en un
                # solo sitio; `null` quiere decir "no hay frase fija" (parar
                # se resuelve callando, repetir necesita lo ultimo dicho, que
                # solo sabe el navegador).
                decir, retoma = que_decir(intencion)
                linea(fase="decision", dirigida=True, s=0.0,
                      intencion=intencion, decir=decir, retoma=retoma,
                      crudo="orden de interrupcion")
            elif CFG.get("solapar"):
                # Sin veredicto: lo dara /preguntar mientras el LLM arranca.
                linea(fase="decision", dirigida=None, s=0.0,
                      crudo="la resuelve /preguntar en paralelo")
            else:
                dirigida, s, crudo = decidir(
                    texto, pet.get("historial"), quien.get("nombre"),
                    CFG["compuerta"], CFG["ollama"],
                    pet.get("respondio_hace_s"))
                linea(fase="decision", dirigida=dirigida, s=round(s, 3),
                      crudo=crudo)
            linea(fase="fin")
        except (BrokenPipeError, ConnectionResetError):
            pass                            # el navegador se fue a mitad
        except Exception as e:
            try:
                linea(fase="fin", error=f"{type(e).__name__}: {e}")
            except OSError:
                pass

    def perfil_matricular(self):
        n = int(self.headers.get("content-length", 0))
        try:
            pet = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self.responder_json(400, {"error": "peticion ilegible"})
        if OIDO is None:
            return self.responder_json(503, {"error": "sin perfiles de voz"})
        nombre = (pet.get("nombre") or "").strip()
        if not nombre:
            return self.responder_json(400, {"error": "falta el nombre"})
        try:
            wav = base64.b64decode(pet.get("wav") or "")
        except (ValueError, TypeError):
            wav = b""
        if len(wav) < 100:
            return self.responder_json(400, {"error": "sin audio"})
        d = OIDO.matricular(nombre, wav)
        self.responder_json(400 if "error" in d else 200, d)

    def perfil_renombrar(self):
        n = int(self.headers.get("content-length", 0))
        try:
            pet = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self.responder_json(400, {"error": "peticion ilegible"})
        if OIDO is None:
            return self.responder_json(503, {"error": "sin perfiles de voz"})
        if OIDO.renombrar(pet.get("id") or "", pet.get("nombre") or ""):
            return self.responder_json(200, {"hecho": True})
        self.responder_json(404, {"error": "no hay tal perfil"})

    def do_POST(self):
        if self.path == "/stt":
            return self.transcribir()
        if self.path == "/escuchar":
            return self.escuchar()
        if self.path == "/barrera":
            return self.barrera()
        if self.path == "/herramientas/cancelar":
            # El boton de "no" de la pagina. Existe porque una accion a medias
            # tiene que poder tirarse SIN hablar: si el asistente entendio mal
            # y ya esta preguntando si manda un correo, lo ultimo que apetece
            # es tener que discutirlo por voz.
            n = int(self.headers.get("content-length", 0))
            d = json.loads(self.rfile.read(n) or b"{}")
            ses = d.get("sesion") or "web"
            había = HERRAMIENTAS.cancelar(ses) if HERRAMIENTAS else False
            return self.responder_json(200, {"cancelada": había})
        if self.path == "/perfiles/matricular":
            return self.perfil_matricular()
        if self.path == "/perfiles/renombrar":
            return self.perfil_renombrar()
        if self.path != "/preguntar":
            return self.send_error(404)
        n = int(self.headers.get("content-length", 0))
        pet = json.loads(self.rfile.read(n) or b"{}")
        self.send_response(200)
        self.send_header("content-type", "application/octet-stream")
        self.send_header("cache-control", "no-store")
        # Connection: close es OBLIGATORIO aqui. Con HTTP/1.1 y sin
        # Content-Length ni chunked, el navegador no tiene forma de saber que
        # la respuesta acabo y se queda esperando datos que no llegan: la
        # pagina se quedaba en "hablando..." para siempre. Cerrando la
        # conexion, el fin de respuesta es el fin de conexion.
        self.send_header("connection", "close")
        self.close_connection = True
        self.end_headers()

        t0 = time.time()

        # PROTOCOLO: marcos de [1 byte tipo][4 bytes longitud BE][carga].
        # tipo 0 = PCM, tipo 1 = evento JSON.
        #
        # El primer intento usaba un byte 0x01 como marca de evento y el resto
        # como PCM. No vale: el PCM contiene 0x01 constantemente, asi que el
        # navegador interpretaba audio como si fuera JSON. Con longitud
        # explicita no hay ambiguedad posible.
        #
        # El servicio de voz usa EL MISMO marco por el websocket, asi que el
        # PCM que baja de la sesion se reenvia al navegador sin tocar un byte.

        def marco(tipo, carga):
            self.wfile.write(struct.pack(">BI", tipo, len(carga)) + carga)
            self.wfile.flush()

        def evento(**kw):
            marco(1, json.dumps(kw).encode())

        if ws_conectar is None:
            evento(tipo="error", texto="falta el paquete 'websockets'; arranca "
                                       "esto con pkgs/vibevoice/.venv/bin/python")
            return

        modelo = pet.get("modelo") or CFG["modelo"]
        # ---- que perfil habla, y con que herramientas --------------------
        # El perfil manda tres cosas a la vez y conviene verlas juntas: su
        # voz (mas abajo, al abrir la sesion), sus coletillas (catalogo) y su
        # juego de herramientas (dominios). Cambiar de perfil en la pagina
        # cambia las tres de golpe, que es lo que hace que suene a OTRO
        # asistente y no al mismo con otro sombrero.
        perfil = pet.get("perfil") or CFG.get("perfil")
        datos_perfil = ((_ASISTENTES["datos"] or {}).get("perfiles") or {}).get(perfil) or {}
        dominios = perfiles.dominios_de(datos_perfil)
        esquemas = (herramientas.esquemas_anthropic(
            herramientas.por_dominios(dominios)) if dominios and HERRAMIENTAS
            else None)
        sesion_herr = pet.get("sesion") or "web"
        historial_pet = pet.get("historial") or []

        # La instruccion de sistema puede venir de la pagina: es la forma de
        # dirigir al LLM sin reiniciar el puente. Vacia o en blanco, vale la
        # del PERFIL, y si el perfil no trae, la de serie del arranque.
        sistema = ((pet.get("sistema") or "").strip()
                   or datos_perfil.get("sistema") or CFG["sistema"])
        registro = (datos_perfil.get("vocabulario") or {}).get("registro")
        if registro:
            sistema += f" Habla en un registro {registro}."
        if esquemas:
            # El anexo de herramientas va SIEMPRE detras y no lo puede pisar
            # la pagina: la regla de que escribir se confirma no es una
            # preferencia de estilo que se cambie desde una casilla de texto.
            sistema += " " + herramientas.instrucciones()
            notas = herramientas.notas_recientes(historial_pet)
            if notas:
                sistema += " " + notas
        # "/no_think" es un truco de qwen bajo Ollama y se añade DESPUES de
        # elegir la instruccion, venga de donde venga. MiniMax manda el
        # razonamiento en bloques aparte, asi que ahi no pinta nada.
        if not pet.get("pensar") and not modelo.lower().startswith("minimax"):
            sistema += " /no_think"
        # DOS colas y no una. Con una sola, el bucle principal se quedaba
        # bloqueado sintetizando y durante ese rato no leia nada, asi que los
        # tokens que el LLM va escribiendo se acumulaban y salian de golpe al
        # terminar: en pantalla el texto dejaba de fluir despues del primer
        # trozo. Con la sesion por websocket el bloqueo cambia de forma -- ya
        # no se espera a que acabe una frase, sino a que llegue el siguiente
        # marco de audio -- pero el remedio es el mismo y sigue haciendo falta:
        # el bucle NUNCA se para mas de 20 ms (recv con plazo) y drena los
        # tokens en cada vuelta.
        cola_texto = queue.Queue()          # tokens y estado: se drena siempre
        cola_trozos = queue.Queue(maxsize=64)   # frases listas para la sesion
        parar = threading.Event()           # el navegador se fue: soltarlo todo
        pendiente, n_frases, dentro = "", 0, False

        # ---- la compuerta, EN PARALELO con el LLM -------------------------
        # Antes iba delante, en fila, y ponia 0,47-0,70 s en el camino
        # critico. Aqui arranca a la vez que el LLM y lo unico que retiene es
        # la ENTREGA de la primera frase a la sesion de voz: mientras no haya
        # veredicto no sale un solo byte de audio. Como la primera frase tarda
        # 1,15-6,5 s en estar escrita y la compuerta 0,7 s, el veredicto llega
        # antes de que haga falta y no se nota.
        #
        # Lo que cuesta si la respuesta era NO: ~0,7 s de LLM tirados. Con
        # Ollama es gratis; con MiniMax son unos pocos tokens. --sin-solapar
        # devuelve la compuerta a /escuchar, delante de todo.
        compuerta = pet.get("compuerta")
        veredicto = {}          # {"dirigida": bool, "s": float, "crudo": str}

        def hilo_compuerta():
            # Pase lo que pase, este hilo TIENE que dejar un veredicto: el
            # bucle de abajo no entrega una sola frase hasta que lo haya, y sin
            # esto una averia aqui dejaria al asistente mudo hasta el plazo de
            # los 90 s. Ante la duda se abre, como hace decidir() por dentro.
            try:
                d, s, crudo = decidir(pet["texto"], compuerta.get("historial"),
                                      compuerta.get("hablante"),
                                      CFG["compuerta"], CFG["ollama"],
                                      compuerta.get("respondio_hace_s"))
            except Exception as e:
                d, s, crudo = True, 0.0, f"averia: {type(e).__name__}: {e}"
            veredicto.update(dirigida=d, s=s, crudo=crudo)

        if compuerta:
            threading.Thread(target=hilo_compuerta, daemon=True).start()

        # ---- los rellenos ------------------------------------------------
        # El perfil manda: cada uno tiene sus coletillas y su voz (ver
        # scripts/perfiles.py). La politica solo dice QUE toca y CUANDO; se
        # consulta desde el bucle de abajo -- que da una vuelta cada 20 ms --
        # y NO desde un hilo, porque emitir marcos desde dos hilos a la vez
        # partiria el flujo binario que lee el navegador.
        catalogo = _ASISTENTES["catalogos"].get(perfil)
        politica = (perfiles.Politica(catalogo, con_compuerta=bool(compuerta))
                    if catalogo is not None else None)

        # ---- ¿es esto el sí de una acción que quedó esperando? -----------
        # Se mira ANTES de tocar el LLM, y si lo es no se toca: el veredicto
        # sale de una lista cerrada de palabras (herramientas.clasificar_
        # respuesta) y el remate es una frase fija. Eso quita del camino la
        # vuelta entera de MiniMax -- 1,2-2 s -- justo en el momento en que
        # menos se perdona esperar, que es despues de haber dicho "sí".
        #
        # Y el que decide NO es el modelo. Ver la cabecera de herramientas.py:
        # dejarselo a el acababa con un «hecho, borrada» sobre algo que no se
        # habia borrado.
        confirmacion, descartada = None, None
        if HERRAMIENTAS is not None and not pet.get("decir"):
            pend = HERRAMIENTAS.pendiente(sesion_herr)
            if pend:
                v = herramientas.clasificar_respuesta(pet.get("texto", ""))
                # UN «SÍ» QUE NO IBA PARA MÍ NO ES UN SÍ. Con el microfono
                # siempre abierto, en una habitacion con gente, alguien dice
                # que si a otra persona y eso mandaria el correo. Aqui SI se
                # espera a la compuerta antes de tocar nada -- es el unico
                # sitio de todo el puente donde vale la pena pagar sus
                # 0,47-0,70 s en el camino critico, porque lo que hay al otro
                # lado no se deshace.
                #
                # Y si no contesta a tiempo, NO se confirma: el pendiente se
                # queda como estaba y el usuario puede repetir el sí. La
                # duda cae siempre del lado de no escribir.
                if compuerta and v in ("si", "no"):
                    limite = time.time() + 3.0
                    while not veredicto and time.time() < limite:
                        time.sleep(0.02)
                    if not veredicto.get("dirigida"):
                        v = "esperar"
                        evento(tipo="pendiente", estado="sigue",
                               resumen=pend["resumen"],
                               motivo="el sí no iba dirigido al asistente"
                                      if veredicto else
                                      "la compuerta no contestó a tiempo")
                if v in ("si", "no"):
                    res, resumen, frase = HERRAMIENTAS.confirmar(sesion_herr,
                                                                 v == "si")
                    confirmacion = {"frase": frase, "resumen": resumen,
                                    "nombre": pend["nombre"], "si": v == "si",
                                    "nota": herramientas.nota_de_remate(
                                        pend["resumen"], v == "si")}
                elif v != "esperar":
                    # NI SÍ NI NO: se descarta y se sigue como pregunta nueva.
                    # Es la unica salida honesta -- "¿y qué hora es?" no es un
                    # sí a mandar un correo -- y ademas la segura: en la duda,
                    # no se escribe.
                    HERRAMIENTAS.cancelar(sesion_herr)
                    descartada = pend["resumen"]
                    evento(tipo="pendiente", estado="descartada",
                           resumen=pend["resumen"],
                           motivo="ni sí ni no: se descarta",
                           nota=herramientas.nota_de_remate(pend["resumen"],
                                                            False))
            HERRAMIENTAS.turno(sesion_herr)
        if confirmacion:
            evento(tipo="herramienta", fase="confirmada" if confirmacion["si"]
                   else "descartada", nombre=confirmacion["nombre"],
                   resumen=confirmacion["resumen"], escribe=True,
                   s=round(time.time() - t0, 3))
            evento(tipo="pendiente", estado="resuelta",
                   nota=confirmacion["nota"])
            # EL REMATE YA ESTA GRABADO. "Enviado." es una de las frases del
            # catalogo del perfil (categoria 'confirmando'), asi que el
            # navegador la tiene decodificada desde que abrio la pagina: se
            # dice sin LLM, sin sesion de voz y sin tocar la VM. Es el unico
            # sitio de todo esto donde la respuesta entera cabe en un relleno,
            # y es justo donde mas se agradece -- despues de decir "sí" nadie
            # quiere esperar dos segundos a saber si se ha mandado.
            pre = (catalogo.buscar("confirmando", confirmacion["frase"])
                   if catalogo is not None else None)
            if pre:
                evento(tipo="hito", hito="token", s=round(time.time() - t0, 3))
                evento(tipo="trozo", id=0, texto=confirmacion["frase"],
                       pendiente="")
                evento(tipo="relleno", clase="confirmando", id=pre["id"],
                       ms=pre["ms"], texto=pre["texto"],
                       url=f"/rellenos/{pre['fichero']}",
                       s=round(time.time() - t0, 3))
                evento(tipo="hecho", id=0, s=round(time.time() - t0, 3))
                return
        if descartada:
            # Que la accion se ha caido se le dice AL MODELO, y en el bloque de
            # sistema. Si no, ve en el historial que le pidieron un correo, no
            # ve ninguna respuesta suya (la pregunta de confirmacion se quita
            # con su pareja, ver _mensajes) y lo vuelve a preparar dos turnos
            # despues, con el usuario hablando ya de otra cosa.
            sistema += " " + herramientas.nota_de_remate(descartada, False)

        # El productor manda SIEMPRE el pendiente que queda tras extraer un
        # trozo, en vez de que la pagina intente descontarlo por su cuenta.
        # Restar longitudes se desalinea en cuanto hay un espacio de mas, y el
        # texto se corrompe en pantalla. Aqui la fuente de verdad es una sola.
        def productor():
            nonlocal pendiente, n_frases, dentro
            # Con historial (escucha continua, o pagina que lo mande) el LLM
            # ve la conversacion entera y quien dice cada cosa; sin el, el
            # camino de siempre, que es el probado.
            historial = historial_pet
            hablante = pet.get("hablante")
            if confirmacion:
                # El sí ya esta resuelto y la frase es fija; se llega aqui solo
                # si el perfil no tenia el WAV pregenerado de ese remate.
                origen = iter([confirmacion["frase"]])
            elif pet.get("decir"):
                # TEXTO FIJO, SIN LLM. Es como se dice el «¿qué pasa?» de una
                # interrupcion: no hay nada que redactar, y meter un LLM en
                # medio costaria 1,15-6,5 s que la frase no aprovecha. Se cuela
                # por el mismo troceador y la misma sesion de voz que el resto.
                origen = iter([pet["decir"]])
            elif esquemas:
                # CON HERRAMIENTAS. El generador produce texto igual que los
                # otros, y ademas diccionarios con lo que va pasando; se
                # distinguen por el tipo unas lineas mas abajo, asi que el
                # troceador de frases no se entera de que hay herramientas.
                origen = herramientas.ciclo(
                    pet["texto"], historial, modelo, sistema, hablante,
                    esquemas, HERRAMIENTAS, sesion_herr, CFG["ollama"])
            elif historial or hablante:
                origen = preguntar_con_historial(pet["texto"], historial,
                                                 modelo, CFG["ollama"],
                                                 sistema, hablante)
            else:
                origen = preguntar(pet["texto"], modelo, CFG["ollama"], sistema)
            try:
                for trozo in origen:
                    if parar.is_set():
                        break       # nadie escucha: no seguir gastando el LLM
                    if isinstance(trozo, dict):
                        # Un suceso de herramienta. Va por la cola de TEXTO y
                        # no por la de frases: no es algo que se diga, es algo
                        # que se pinta, y tiene que llegar a la pagina aunque
                        # la voz vaya atascada.
                        cola_texto.put(("herramienta", time.time() - t0,
                                        trozo, pendiente))
                        continue
                    texto = ""
                    for parte in re.split(r"(<[^>]{0,20}>)", trozo):
                        if ABRE_PENSAMIENTO.fullmatch(parte or ""):
                            dentro = True
                        elif CIERRA_PENSAMIENTO.fullmatch(parte or ""):
                            dentro = False
                        elif not dentro:
                            texto += parte or ""
                    if not texto:
                        continue
                    pendiente += herramientas.sin_json(limpiar(texto))
                    frases, pendiente = trocear(pendiente, primera=n_frases == 0,
                                                minimo_primera=CFG["arranque"])
                    cola_texto.put(("token", time.time() - t0, texto, pendiente))
                    for f in frases:
                        cola_trozos.put(("frase", time.time() - t0, (n_frases, f), pendiente))
                        n_frases += 1
                frases, pendiente = trocear(pendiente, forzar_final=True,
                                            primera=n_frases == 0,
                                            minimo_primera=CFG["arranque"])
                for f in frases:
                    cola_trozos.put(("frase", time.time() - t0, (n_frases, f), pendiente))
                    n_frases += 1
            except Exception as e:
                cola_trozos.put(("error", 0, f"{type(e).__name__}: {e}", ""))
            cola_trozos.put(None)

        threading.Thread(target=productor, daemon=True).start()
        visto = set()

        # Lo que han hecho las herramientas en esta respuesta, para el resumen
        # final y para poder afirmar en las pruebas que se llamo a la que
        # tocaba. `dijo_algo` es lo que decide si el relleno 'consultando'
        # tiene sentido: ver mas abajo.
        usadas, fallos_herr, dijo_algo = [], [], False

        def drenar_texto():
            """Saca los tokens pendientes sin bloquear. Se llama en cada vuelta
            del bucle para que el redactado no se congele mientras baja audio."""
            nonlocal dijo_algo
            while True:
                try:
                    clase, s_t, dato, pend = cola_texto.get_nowait()
                except queue.Empty:
                    return
                if clase == "herramienta":
                    ev = dict(dato, s=round(s_t, 3))
                    if ev.get("fase") == "llamando":
                        usadas.append(ev.get("nombre"))
                        # EL RELLENO 'consultando' SOLO SI EL MODELO NO HABLO.
                        # Cuando el propio modelo ha escrito "voy a mirar tu
                        # calendario", esa frase YA es el relleno, y es mejor
                        # que cualquiera de los nuestros porque nombra lo que
                        # de verdad se esta consultando. Meterle una coletilla
                        # detras seria decir dos veces lo mismo y peor.
                        if (politica is not None and not dijo_algo
                                and not suena):
                            cual = politica.por_suceso("consultando")
                            if cual:
                                elegido = catalogo.elegir(cual)
                                if elegido:
                                    politica.apuntar(cual, s_t, elegido["ms"])
                                    evento(tipo="relleno", clase=cual,
                                           id=elegido["id"], ms=elegido["ms"],
                                           texto=elegido["texto"],
                                           url=f"/rellenos/{elegido['fichero']}",
                                           s=round(s_t, 3))
                    elif ev.get("fase") == "error":
                        # LA HERRAMIENTA REVENTO. Aqui es donde 'negacion'
                        # gana su sitio: el modelo va a tardar otra vuelta
                        # entera en enterarse y redactar la disculpa, y
                        # mientras tanto el usuario esta oyendo silencio
                        # despues de haberle pedido algo. Un "no he podido con
                        # eso" a tiempo vale mas que la frase perfecta tarde.
                        fallos_herr.append(ev.get("nombre"))
                        if politica is not None and not suena:
                            cual = politica.por_suceso("negacion")
                            if cual:
                                elegido = catalogo.elegir(cual)
                                if elegido:
                                    politica.apuntar(cual, s_t, elegido["ms"])
                                    evento(tipo="relleno", clase=cual,
                                           id=elegido["id"], ms=elegido["ms"],
                                           texto=elegido["texto"],
                                           url=f"/rellenos/{elegido['fichero']}",
                                           s=round(s_t, 3))
                    evento(**ev)
                    continue
                dijo_algo = True
                if "token" not in visto:
                    visto.add("token"); evento(tipo="hito", hito="token", s=s_t)
                evento(tipo="token", texto=dato, pendiente=pend)

        # ---- por que trozo va el modelo -----------------------------------
        # La pagina pinta cuatro estados por trozo y hay que seguir sabiendo
        # cual es cual. Con una peticion por frase era trivial: la frase que se
        # estaba pidiendo era la que sonaba. Con UNA sesion continua ya no hay
        # peticiones que contar, asi que se lleva la cuenta en TOKENS:
        #
        #   - cada acuse {"tipo":"texto","tokens":n} dice cuantos tokens ocupo
        #     la frase que se acaba de entregar -> se sabe donde acaba cada una
        #   - el modelo los consume EN ORDEN, asi que "cuantos tokens lleva
        #     consumidos" identifica el trozo que esta diciendo ahora mismo
        #
        # LEER VA POR DELANTE DE SONAR, y por eso se resta ADELANTO_TEXTO antes
        # de mover nada: medido en la VM, cuando el modelo habia consumido el
        # texto entero aun quedaban 2,4 s de audio por bajar. Aun asi el color
        # va un poco por delante de lo que se OYE, porque el navegador reproduce
        # detras de lo que se genera -- el mismo desfase que ya tenia la via
        # HTTP, que marcaba "sonando" al llegar el primer byte de la frase.
        emitidos = []       # ids que la pagina ya conoce, en orden
        sin_acuse = []      # entregados a la sesion, sin acuse todavia
        entregados = []     # {"id", "fin"} en orden, con el token en que acaban
        estado = {}         # id -> ultimo estado que se le mando a la pagina
        acusados = 0        # tokens acusados por la sesion, acumulados
        consumido = 0       # tokens que el modelo ya se comio, segun la sonda
        cabeza = 0          # indice en `entregados` del trozo que se esta diciendo
        suena = False       # ya bajo algo de PCM
        seg_pcm = 0.0       # segundos de audio retransmitidos (24 kHz, s16 mono)

        def marcar(idx, tipo):
            if estado.get(idx) != tipo:
                estado[idx] = tipo
                evento(tipo=tipo, id=idx, s=round(time.time() - t0, 3))

        def avanzar(consumidos=None):
            """Mueve la cabeza hasta donde llegue el texto ya consumido."""
            nonlocal cabeza, consumido
            if consumidos is not None:
                consumido = max(consumido, consumidos)
            # ANTES DEL PRIMER PCM NO SE HA DICHO NADA, por mucho texto que el
            # modelo lleve leido: leer va por delante de sonar y arranca con
            # varias ventanas de ventaja. Sin esta guarda, una primera frase
            # corta se marcaba como dicha sin haber pasado nunca por verde.
            if not suena:
                return
            dicho = consumido - ADELANTO_TEXTO
            while cabeza < len(entregados) and entregados[cabeza]["fin"] <= dicho:
                marcar(entregados[cabeza]["id"], "hecho")
                cabeza += 1
            if cabeza < len(entregados):
                marcar(entregados[cabeza]["id"], "sonando")

        # SONDA. El websocket no dice por donde va: "sonando" llega UNA vez, al
        # primer PCM, y "esperando" solo cuando el modelo se queda sin texto por
        # delante -- que con un LLM rapido no pasa NUNCA (medido: cero eventos
        # "esperando" en una respuesta de 8,7 s alimentada de golpe). Sin este
        # dato los trozos se quedarian todos en amarillo hasta el final.
        #
        # Se pregunta por GET /tts/sesion/{id}, que publica `pendientes`: los
        # tokens entregados que el modelo aun no ha consumido. Es la propia
        # contabilidad del servicio, no una estimacion. Si el sondeo falla se
        # abandona sin ruido y queda el repliegue de "esperando", que es peor
        # pero no rompe nada.
        cola_avance = queue.Queue()

        def sonda(nombre):
            url = f"{CFG['voz_url']}/tts/sesion/{urllib.parse.quote(nombre)}"
            while not parar.is_set():
                # Se lee ANTES de preguntar: si entra texto mientras, viene
                # contado en `pendientes` y la cuenta sale CORTA, nunca larga.
                # Quedarse corto retrasa un color; pasarse lo adelantaria a un
                # trozo que todavia no se ha dicho.
                antes = acusados
                try:
                    p = urllib.request.Request(url)
                    if CFG["token"]:
                        p.add_header("authorization", f"Bearer {CFG['token']}")
                    est = json.load(urllib.request.urlopen(p, timeout=3))
                except Exception:
                    return      # 404 al terminar la sesion, o el servicio no
                                # contesta: se deja de preguntar y ya esta
                cola_avance.put(antes - int(est.get("pendientes", 0)))
                parar.wait(0.2)

        # ---- la sesion ----------------------------------------------------
        destino = (CFG["voz_url"].replace("https://", "wss://")
                                 .replace("http://", "ws://") + "/tts/sesion/ws")
        cabeceras = {}
        if CFG["token"]:
            # Por CABECERA, no por ?token=. El repliegue de query existe para el
            # navegador, que no puede poner cabeceras en new WebSocket(), y deja
            # el secreto en los registros de acceso. Aqui el cliente es Python.
            cabeceras["authorization"] = f"Bearer {CFG['token']}"
        abrir = {"accion": "abrir",
                 "voz": pet.get("voz") or CFG["voz"],
                 "cfg_scale": pet.get("cfg") or CFG["cfg"]}
        if pet.get("pasos"):
            abrir["pasos"] = pet["pasos"]
        if pet.get("semilla") is not None:
            abrir["semilla"] = pet["semilla"]
        # La velocidad NO se manda: el websocket solo admite 1 y lo razona en su
        # bloque VELOCIDAD. La pagina la aplica al reproducir y lo dice ahi.

        terminado = False

        # La voz del asistente, PARA APRENDERLA: se guardan los primeros
        # segundos del PCM que baja de la sesion y al terminar se refresca con
        # ellos el perfil 'asistente' (oido.py). Es la clave del anti-eco: la
        # voz GENERADA no casa con el WAV del prompt (coseno 0,24-0,38,
        # medido), asi que la unica verdad de terreno es lo que suena.
        dicho_pcm, dicho_tope = [], 20 * 24000 * 2

        def al_pcm(carga):
            nonlocal suena, seg_pcm, dicho_tope
            marco(0, carga)
            seg_pcm += len(carga) / 2 / 24000
            if dicho_tope > 0:
                dicho_pcm.append(carga[:dicho_tope])
                dicho_tope -= len(carga)
            if not suena:
                suena = True
                avanzar()       # ya se oye algo: repartir lo que sepa la sonda

        def al_evento(ev):
            nonlocal acusados, terminado
            tipo = ev.get("tipo")
            if tipo == "abierta":
                threading.Thread(target=sonda, args=(ev["sesion"],),
                                 daemon=True).start()
            elif tipo == "texto":
                # Acuse en orden de entrega: el enesimo acuse es la enesima
                # frase que se mando. Con eso se sabe en que token acaba.
                acusados += int(ev.get("tokens", 0))
                if sin_acuse:
                    idx = sin_acuse.pop(0)
                    entregados.append({"id": idx, "fin": acusados})
                    marcar(idx, "sintetizando")
            elif tipo == "esperando":
                # El modelo se quedo sin texto por delante: todo lo entregado
                # esta dicho. Repliegue por si la sonda no esta disponible.
                if ev.get("esperando"):
                    avanzar(acusados)
            elif tipo == "hecho":
                terminado = True
            elif tipo == "error":
                evento(tipo="error", texto=ev.get("texto", "error de la sesion"))
                terminado = True

        ws = None
        # `roto` es "la locucion no llego a su final normal" y `se_fue` es "el
        # navegador colgo". Se distinguen porque en el segundo caso no hay a
        # quien decirle nada: mandar un relleno de cierre a una conexion muerta
        # solo sirve para llenar el registro de errores.
        roto, se_fue = False, False
        try:
            # compression=None: el PCM son muestras, no texto -- deflate solo
            # anadiria CPU en los dos extremos sin quitar bytes.
            ws = ws_conectar(destino, additional_headers=cabeceras,
                             max_size=None, ping_interval=None,
                             compression=None, open_timeout=15)
            ws.send(json.dumps(abrir))
            resto, cerrado, fin_mandado, ultimo = b"", False, False, time.time()
            while not terminado:
                drenar_texto()
                # ¿Toca un relleno? Solo mientras no haya sonado el habla de
                # verdad: `suena` lo pone al_pcm en el primer PCM y a partir de
                # ahi toca() devuelve None para siempre. Lo que se manda es el
                # ID de un WAV que la pagina ya tiene decodificado, no el
                # audio: pedirlo ahora metria en el camino critico justo la
                # latencia que el relleno viene a tapar.
                if politica is not None:
                    t_r = time.time() - t0
                    cual = politica.toca(
                        t_r, suena,
                        veredicto.get("dirigida") if compuerta else True)
                    if cual:
                        elegido = catalogo.elegir(cual)
                        if elegido:
                            politica.apuntar(cual, t_r, elegido["ms"])
                            evento(tipo="relleno", clase=cual,
                                   id=elegido["id"], ms=elegido["ms"],
                                   texto=elegido["texto"],
                                   url=f"/rellenos/{elegido['fichero']}",
                                   s=round(t_r, 3))
                # 1) meter en la sesion TODAS las frases que haya listas. No se
                #    dosifica a proposito: el modelo callado es un silencio en
                #    mitad de la locucion, y ese es justo el problema que la
                #    sesion viene a resolver.
                #
                #    LA PRIMERA ENTREGA NO BASTA POR SI SOLA, y conviene
                #    saberlo: generate() no emite un solo byte hasta poder leer
                #    la ventana en curso Y la siguiente, o sea 10 tokens.
                #    Medido contra la VM mandando UNA frase y sin cerrar el
                #    texto: 3 tokens y 9 tokens se quedan parados esperando la
                #    frase siguiente; 16 y 26 arrancan en 0,235 s. Con
                #    --arranque 15 la primera frase suele rondar los 9, asi que
                #    el modelo espera a la segunda.
                #
                #    Se deja asi A PROPOSITO. Se probo --arranque 40, que da una
                #    primera frase de 50-70 caracteres y arranca sin esperar:
                #    sale PEOR. Medido en la VM, del primer token del LLM al
                #    primer sonido, 6 pasadas cada uno:
                #        una peticion por frase (lo de antes)  0,56 s
                #        sesion, --arranque 15                 0,53 s
                #        sesion, --arranque 40                 0,63 s
                #    Esperar a que el LLM escriba 25 caracteres mas cuesta mas
                #    que la pausa que ahorra, porque escribe mas deprisa de lo
                #    que el modelo habla.
                #
                #    PERO NO ANTES DE QUE LA COMPUERTA DE EL VISTO BUENO. Las
                #    frases se quedan en la cola (que tiene sitio de sobra: 64
                #    frases contra los ~0,7 s que tarda el veredicto) y salen
                #    todas juntas en cuanto llega. Retener AQUI y no antes es
                #    lo que hace que la compuerta salga del camino critico sin
                #    arriesgarse a decir en voz alta algo que no era para el.
                if compuerta and veredicto and not veredicto["dirigida"]:
                    evento(tipo="no_dirigida", s=round(veredicto["s"], 3),
                           crudo=veredicto.get("crudo", ""))
                    break
                entregar = not compuerta or bool(veredicto)
                if entregar and compuerta and "visto" not in visto:
                    visto.add("visto")
                    evento(tipo="hito", hito="compuerta",
                           s=round(veredicto["s"], 3))
                while entregar and not cerrado:
                    try:
                        item = cola_trozos.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        cerrado = True
                        break
                    clase, s_t, dato, pend = item
                    if clase == "error":
                        evento(tipo="error", texto=dato)
                        cerrado = True
                        break
                    idx, frase = dato
                    if "frase" not in visto:
                        visto.add("frase"); evento(tipo="hito", hito="frase", s=s_t)
                    evento(tipo="trozo", id=idx, texto=frase, pendiente=pend)
                    emitidos.append(idx)
                    estado[idx] = "trozo"
                    ws.send(json.dumps({"accion": "texto", "texto": frase}))
                    sin_acuse.append(idx)
                    ultimo = time.time()
                if cerrado and not fin_mandado:
                    ws.send(json.dumps({"accion": "fin"}))
                    fin_mandado = True
                    ultimo = time.time()
                # 2) lo que haya bajado por el socket. El plazo corto es lo que
                #    mantiene vivo el redactado: nunca se espera mas de 20 ms
                #    sin volver a drenar los tokens del LLM.
                try:
                    msg = ws.recv(timeout=0.02)
                except TimeoutError:
                    msg = None
                except ConnectionClosed:
                    break
                if msg is not None:
                    ultimo = time.time()
                    resto = desmarcar(resto + msg, al_pcm, al_evento)
                # 3) por donde va el modelo, segun la sonda
                while True:
                    try:
                        avanzar(cola_avance.get_nowait())
                    except queue.Empty:
                        break
                if time.time() - ultimo > SILENCIO_MAXIMO:
                    evento(tipo="error",
                           texto=f"la sesion de voz lleva {SILENCIO_MAXIMO:.0f} s "
                                 f"sin decir nada; se corta")
                    roto = True
                    break
                # El descarrile no calla: emite. Si baja bastante mas audio del
                # que el texto acusado puede justificar, la locucion perdio su
                # EOS y no va a terminar sola; cortar aqui cierra el websocket
                # y abortar() la desmonta en el servidor.
                if seg_pcm > TOPE_AUDIO_BASE + TOPE_AUDIO_POR_TOKEN * acusados:
                    evento(tipo="error",
                           texto=f"{seg_pcm:.0f} s de audio para {acusados} "
                                 f"tokens de texto: la locucion descarrilo y se "
                                 f"corta")
                    roto = True
                    break
            drenar_texto()
            for idx in emitidos:            # lo que quede a medias, cerrado
                marcar(idx, "hecho")
        except (BrokenPipeError, ConnectionResetError):
            se_fue = True                   # el navegador se fue a mitad
        except Exception as e:
            roto = True
            try:
                evento(tipo="error", texto=f"{type(e).__name__}: {e}")
            except OSError:
                pass
        # ---- 'cerrando' y 'negacion': los dos rellenos que faltaban -------
        # Aqui es donde tienen sentido, y no antes, porque hasta este punto no
        # se sabe COMO ha acabado la respuesta. La regla, entera:
        #
        #   la locucion se rompio  +  ya sonaba algo  ->  CERRANDO
        #     Es el unico caso en que meterle palabras que el LLM no dijo
        #     mejora la cosa. La alternativa es una frase que se corta en seco
        #     a mitad, que no suena a "se acabo": suena a averia. Un "y eso es
        #     todo" cierra la cadencia y el usuario sabe que le toca hablar.
        #     En un final NORMAL no se usa -- a la tercera vez, una coletilla
        #     de cierre pegada a cada respuesta cansa mas que el silencio.
        #
        #   la locucion se rompio  +  no sono nada  ->  NEGACION
        #     Callarse del todo despues de que te pregunten es lo peor que
        #     puede hacer: el usuario no sabe si le has oido, y repite. "No he
        #     podido con eso" cuesta 1 s y cierra la duda.
        #
        # Son excluyentes por `suena`, y ninguno de los dos suena si la
        # respuesta salio bien.
        try:
            if roto and not se_fue and politica is not None:
                cual = politica.remate(suena)
                elegido = catalogo.elegir(cual) if cual else None
                if elegido:
                    politica.apuntar(cual, time.time() - t0, elegido["ms"])
                    evento(tipo="relleno", clase=cual, id=elegido["id"],
                           ms=elegido["ms"], texto=elegido["texto"],
                           url=f"/rellenos/{elegido['fichero']}",
                           s=round(time.time() - t0, 3))
            if not se_fue and (usadas or fallos_herr):
                p = (HERRAMIENTAS.pendiente(sesion_herr) or {}) if HERRAMIENTAS else {}
                evento(tipo="herramientas_resumen", usadas=usadas,
                       fallos=fallos_herr, pendiente=p.get("resumen"),
                       # La NOTA con la que la pagina tiene que apuntar esta
                       # respuesta en el historial. Ver _mensajes en
                       # herramientas.py: la pregunta de confirmacion la
                       # escribio el programa, y si entra en el historial como
                       # si la hubiera dicho el modelo, el modelo aprende a
                       # contestar asi y deja de llamar a las herramientas.
                       nota=(herramientas.nota_de_pregunta(p["resumen"])
                             if p.get("resumen") else None))
        except OSError:
            pass
        finally:
            # Cerrar el socket es lo que desmonta la sesion en el servidor: al
            # caerse, abortar() corta la generate() en curso y la saca del
            # registro. Sin esto quedaria una sesion hablando para nadie con el
            # candado del modelo tomado.
            parar.set()
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
            # Aprender la voz que se acaba de emitir, fuera del camino de la
            # respuesta: son ~25 ms de huella, pero ni eso se le cobra aqui.
            if OIDO is not None and dicho_pcm:
                threading.Thread(target=OIDO.aprender_asistente,
                                 args=(b"".join(dicho_pcm), 24000),
                                 daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 para abrirlo a la red local")
    ap.add_argument("--tls-cert", default="", help="certificado, para HTTPS")
    ap.add_argument("--tls-clave", default="", help="clave del certificado")
    ap.add_argument("--modelo", default=os.environ.get("ASISTENTE_MODELO", "MiniMax-M3"),
                    help="MiniMax-M3 (por defecto) o cualquier modelo de Ollama")
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--voz-url", default=os.environ.get("VOZ_STREAM_URL", "http://127.0.0.1:8082"))
    # El sintetizador y whisper pueden estar en MAQUINAS distintas -- lo normal
    # aqui: la voz en la VM y whisper en local -- y cada uno tiene su propio
    # token. Mandarle a whisper el de la VM da 401 y parece que no esta
    # levantado cuando si lo esta.
    ap.add_argument("--token-api", default=os.environ.get("VOZ_API_TOKEN", ""),
                    help="token de la API de voz (whisper); por defecto, el mismo")
    ap.add_argument("--api-url", default=os.environ.get("VOZ_API_URL", "http://127.0.0.1:8080"),
                    help="la API de voz con /stt (whisper), para el microfono. "
                         "En local: cd docker && docker compose up -d whisper voz-api")
    ap.add_argument("--token", default=os.environ.get("VOZ_TOKEN", ""))
    ap.add_argument("--voz", default=os.environ.get("VIBEVOICE_VOZ", "sp-Spk1_man"))
    ap.add_argument("--cfg", type=float, default=3.5,
                    help="guia CFG. 3.5 por defecto. Entre 3.0 y 3.5 el WER es\n                         IDENTICO (9,7 %% medio, 11,1 %% peor); lo que cambia es\n                         que 3.5 habla un 7 %% mas despacio y recorre 12,1\n                         semitonos frente a 10,4, o sea entona mas. A 4.5 el\n                         peor caso se dobla (22,2 %%) y encima aplana la melodia")
    ap.add_argument("--arranque", type=int, default=15)
    ap.add_argument("--sistema", default="Responde en español, breve y natural, "
                                         "en frases cortas. Sin listas ni markdown.")
    ap.add_argument("--compuerta",
                    default=os.environ.get("ASISTENTE_COMPUERTA", "qwen3:4b"),
                    help="modelo que decide si una frase oida va dirigida al "
                         "asistente. qwen3:4b en Ollama local por defecto: "
                         "~0,4 s con think:false. Los MiniMax-* valen pero "
                         "razonan aunque se les pida que no y tardan 3-6 s "
                         "(medido); ver scripts/conversacion.py")
    ap.add_argument("--perfiles",
                    default=os.environ.get(
                        "ASISTENTE_PERFILES",
                        os.path.join(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))), "perfiles_voz.json")),
                    help="donde guardar los perfiles de voz (JSON)")
    ap.add_argument("--whisper-url",
                    default=os.environ.get("WHISPER_NATIVO_URL", ""),
                    help="un whisper.cpp NATIVO (scripts/whisper-mac.sh) para "
                         "la escucha continua, saltandose voz-api y ffmpeg. "
                         "Medido en el mismo Mac con el mismo modelo small: "
                         "0,28 s frente a 2,18 s en Docker, y la MISMA "
                         "transcripcion. Sin esto todo va por --api-url")
    ap.add_argument("--prompt-stt", default=os.environ.get(
        "VOZ_PROMPT_STT",
        "Vocabulario tecnico: homelab, Proxmox, WireGuard, Docker, contenedor, "
        "Caddy, systemd, OpenClaw, Piper, whisper, backup, deploy, NixOS."),
        help="sesgo de vocabulario para el whisper nativo. Es el mismo que "
             "voz-api aplica por su cuenta, y se nota: sin el, 'WireGuard' "
             "sale como 'We The War'")
    # ---- perfiles de ASISTENTE (no los de voz de --perfiles) --------------
    ap.add_argument("--asistentes",
                    default=os.environ.get("VOZ_PERFILES"),
                    help="fichero JSON con los perfiles de asistente "
                         "(scripts/perfiles.py). Sin esto se busca "
                         "perfiles_asistente.json en la raiz del repo, y si "
                         "tampoco esta se usa el perfil de serie")
    ap.add_argument("--asistente", default=None,
                    help="perfil activo. Por defecto, el 'perfil_por_defecto' "
                         "del fichero")
    ap.add_argument("--rellenos-cache", default=None,
                    help="donde viven los WAV pregenerados. Por defecto "
                         "$VOZ_PERFILES_CACHE, $STATE_DIRECTORY o "
                         "~/.local/state/voz-perfiles")
    ap.add_argument("--sin-rellenos", action="store_true",
                    help="no genera ni usa audios de relleno. El ciclo queda "
                         "exactamente como antes de que existieran")
    ap.add_argument("--prebufer", type=float,
                    default=float(os.environ.get("VOZ_PREBUFER", "0.15")),
                    help="segundos de audio que la pagina acumula antes de "
                         "empezar a sonar. SE MIDIO y 0,15 basta en esta red "
                         "(retraso maximo acumulado 0,0 ms por el puente, "
                         "18,7 ms por el websocket directo): subirlo a ciegas "
                         "solo costaria latencia. El margen de verdad lo dan "
                         "los rellenos, que son 0,7-2,4 s de bufer gratis")
    ap.add_argument("--datos-herramientas", default=None,
                    help="el JSON con los datos SIMULADOS de las herramientas "
                         "(scripts/herramientas.py). Por defecto "
                         "herramientas_simuladas.json en la raiz del repo. Se "
                         "relee solo cuando cambia, asi que se puede editar "
                         "con el asistente en marcha")
    ap.add_argument("--sin-herramientas", action="store_true",
                    help="apaga el uso de herramientas. El asistente vuelve a "
                         "contestar solo con lo que sabe, que es como estaba "
                         "antes de que existieran")
    ap.add_argument("--sin-solapar", action="store_true",
                    help="devuelve la compuerta a su sitio de antes -- delante "
                         "del LLM y en fila -- en vez de correrla en paralelo. "
                         "Cuesta 0,47-0,70 s de latencia y ahorra los tokens "
                         "de LLM que se tiran cuando la respuesta es NO")
    a = ap.parse_args()
    CFG.update(modelo=a.modelo, ollama=a.ollama, voz_url=a.voz_url, token=a.token,
               voz=a.voz, arranque=a.arranque, sistema=a.sistema, cfg=a.cfg,
               voz_api=a.api_url, token_api=a.token_api or a.token,
               compuerta=a.compuerta, whisper=a.whisper_url.rstrip("/"),
               prompt_stt=a.prompt_stt, solapar=not a.sin_solapar,
               perfil=a.asistente, prebufer=a.prebufer)
    global OIDO, HERRAMIENTAS
    # Las herramientas, ANTES de servir nada: si el JSON de datos esta roto es
    # mejor enterarse en el arranque que a la tercera pregunta. Cuesta leer un
    # fichero, no retrasa nada.
    if not a.sin_herramientas:
        try:
            HERRAMIENTAS = herramientas.Ejecutor(a.datos_herramientas)
            HERRAMIENTAS.sim.datos      # fuerza la lectura y valida el JSON
        except Exception as e:
            HERRAMIENTAS = None
            print(f"  [aviso] sin herramientas: {type(e).__name__}: {e}")
    OIDO = Oido(a.perfiles)
    OIDO.precargar()        # ~6 s de carga del modelo, en un hilo aparte
    # Los perfiles de asistente y sus rellenos. En un hilo y sin poder tumbar
    # el arranque: ver el bloque PERFILES DE ASISTENTE Y RELLENOS.
    arrancar_perfiles(a.asistentes, a.rellenos_cache, a.voz_url, a.token,
                      generar=not a.sin_rellenos)
    if _ASISTENTES["datos"] and not a.asistente:
        CFG["perfil"] = _ASISTENTES["datos"]["perfil_por_defecto"]
    print(f"asistente en http://127.0.0.1:{a.puerto}")
    print(f"  LLM : {a.modelo}"
          f"{'' if a.modelo.lower().startswith('minimax') else ' via ' + a.ollama}")
    print(f"  voz : {a.voz_url}/tts/sesion/ws (una sesion por respuesta)")
    print(f"  stt : {a.api_url}/stt (el microfono de la pagina; whisper)")
    if a.whisper_url:
        print(f"        {a.whisper_url}/inference para la escucha continua "
              f"(whisper nativo, ~8x mas rapido)")
    else:
        print("        [aviso] sin --whisper-url la escucha continua paga "
              "2,1-2,9 s por frase en whisper.\n"
              "                levanta el nativo con scripts/whisper-mac.sh")
    if _ASISTENTES["datos"] and not a.sin_rellenos:
        print(f"  perf: {', '.join(sorted(_ASISTENTES['datos']['perfiles']))} "
              f"(activo: {CFG.get('perfil')}) · rellenos en "
              f"{CFG.get('rellenos_dir')} · prebufer {a.prebufer:.2f} s")
    if HERRAMIENTAS is not None:
        n_esc = len(herramientas.ESCRITURAS)
        print(f"  herr: {len(herramientas.CATALOGO)} herramientas SIMULADAS "
              f"({n_esc} escriben y se confirman por voz) · datos en "
              f"{HERRAMIENTAS.sim.ruta}")
    else:
        print("  herr: apagadas")
    print(f"  oido: compuerta {a.compuerta}"
          f"{' (en paralelo con el LLM)' if not a.sin_solapar else ' (en fila)'}"
          f" · perfiles en {a.perfiles}")
    if ws_conectar is None:
        print("  [aviso] falta el paquete 'websockets': el puente sirve la "
              "pagina pero no podra hablar.\n"
              "          arranca con pkgs/vibevoice/.venv/bin/python")
    srv = ThreadingHTTPServer((a.host, a.puerto), Puente)
    if a.tls_cert and a.tls_clave:
        # HTTPS no es por paranoia: el navegador EXIGE contexto seguro para
        # getUserMedia, y sin el no hay microfono desde otra maquina. En
        # 127.0.0.1 no hace falta porque localhost cuenta como seguro.
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(a.tls_cert, a.tls_clave)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
        print(f"  https activo (certificado propio: el navegador avisara la "
              f"primera vez; acepta la excepcion y el microfono funcionara)")
    if a.host not in ("127.0.0.1", "localhost"):
        print(f"  ABIERTO A LA RED en {a.host}: cualquiera que llegue a este "
              f"puerto puede usar tu LLM y oir tus perfiles de voz. Red de casa "
              f"si, internet no.")
    srv.serve_forever()


if __name__ == "__main__":
    main()
