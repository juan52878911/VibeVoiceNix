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
from oido import Oido  # noqa: E402

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

PAGINA = """<!doctype html><html lang="es"><head><meta charset="utf-8">
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
    <select id="modelo"></select>
    <label style="color:var(--s);font-size:.88rem">
      <input type="checkbox" id="pensar"> dejar que razone
    </label>
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
  dice) y desecha la suya propia, así que puedes interrumpirle mientras habla.
  Los desconocidos reciben un perfil automático; ponles nombre abajo.
  Arranca apagada y solo funciona en <code>127.0.0.1</code>, como el botón
  Hablar.</p>
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
function lanzarPregunta(q,extra){
  preguntaEnCurso=preguntarVoz(q,extra||{}).finally(()=>{preguntaEnCurso=null;});
  return preguntaEnCurso;
}
async function preguntarVoz(q,extra){
  $("ir").disabled=true; $("parar").hidden=false; enPregunta=true;
  ["h1","h2","h3","h4"].forEach(i=>$(i).textContent="—");
  trozos=[]; pendiente=""; pintar(); di("preguntando…");
  const actx=new AudioContext(); const ab=new AbortController();
  ctx=actx; cabeza=0; aborto=ab;
  // La velocidad NO viaja al servidor: el websocket de sesion la rechaza a
  // proposito (ver la nota del panel). Se aplica aqui con playbackRate, que
  // es gratis pero mueve el tono. Se congela al empezar para que moverla a
  // mitad no descuadre el reloj de encolado.
  const vel=+$("vel").value;
  const t0=performance.now(); let resto=new Uint8Array(0), primero=0, hitos={};
  const marcas=extra.marcas||{}; marcas.t0=t0;
  let respuesta="";
  historial.push({rol:"usuario",texto:q,quien:extra.hablante||undefined});
  while(historial.length>24) historial.shift();
  try{
    const r=await fetch("/preguntar",{method:"POST",signal:ab.signal,
      headers:{"content-type":"application/json"},
      body:JSON.stringify({texto:q,modelo:$("modelo").value,pensar:$("pensar").checked,
        sistema:$("sistema").value,
        historial:historial.slice(0,-1),   // lo anterior a esta pregunta
        hablante:extra.hablante||null,
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
          cabeza=actx.currentTime+0.15; }
        const buf=actx.createBuffer(1,f32.length,24000);
        buf.copyToChannel(f32,0);
        const src=actx.createBufferSource(); src.buffer=buf; src.connect(actx.destination);
        src.playbackRate.value=vel;
        if(cabeza<actx.currentTime) cabeza=actx.currentTime;
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
  }catch(e){ di(e.name==="AbortError"?"parado.":"error: "+e.message,e.name!=="AbortError"); }
  $("ir").disabled=false; $("parar").hidden=true;
  enPregunta=false; enAudio=false;
  try{ actx.close(); }catch(_){}
  // Al historial va lo que LLEGO A DECIR: si le interrumpieron a mitad, eso
  // es lo que la otra persona oyo, y es a eso a lo que contestara.
  if(respuesta.trim()) historial.push({rol:"asistente",texto:respuesta.trim()});
  while(historial.length>24) historial.shift();
  if(escucha.activa&&!enPregunta) estEsc("escuchando");
  return marcas;
}
$("ir").addEventListener("click",()=>{
  const q=$("q").value.trim(); if(!q) return;
  lanzarPregunta(q);
});
$("parar").addEventListener("click",()=>aborto&&aborto.abort());

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
// ~100 ms seguidos por encima del umbral alto y cierra tras 600 ms por
// debajo del bajo. 600 y no 300: las pausas internas de una frase dictada
// llegan a 400-500 ms (medido con locuciones de Piper: hasta 0,46 s entre
// clausulas) y un cierre de 300 ms parte la frase en dos. Se antepone
// ademas ~400 ms de antesala para no comerse el arranque de la primera
// palabra, que el umbral solo pilla ya empezada.
class Vad{
  constructor(al){
    this.al=al; this.rate=48000;
    this.cierreMs=600; this.preMs=400; this.minVozMs=250; this.maxMs=15000;
    this.ruido=0.004; this.resto=new Float32Array(0);
    this.enVoz=false; this.pre=[]; this.seg=[]; this.silencio=0;
    this.conVoz=0; this.arranque=0;
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
               vozConAudio:false};
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
             if(escucha.activa&&!enPregunta) estEsc("voz detectada","voz"); }
      else if(escucha.activa&&!enPregunta&&!escucha.cola.length&&!escucha.procesando)
        estEsc("escuchando");
    },
    segmento:(f32,rate)=>alSegmento(f32,rate)});
}
function alSegmento(f32,rate){
  if(!escucha.activa) return;
  // "hablando": sono voz del asistente durante ALGUNA parte del trozo. El
  // servidor endurece el filtro con eso (ver /escuchar).
  const hablando=enAudio||escucha.vozConAudio||__escucha.hablando;
  escucha.vozConAudio=false;
  // Repliegue sin huellas: medio duplex. Sin timbre no hay forma fiable de
  // distinguir al asistente del que interrumpe, asi que mientras hay una
  // pregunta en marcha el microfono no cuenta.
  if(!escucha.huellas&&(enPregunta||__escucha.hablando)) return;
  if(escucha.cola.length>=2) escucha.cola.shift();   // no acumular retraso
  escucha.cola.push({f32,rate,fin:performance.now(),hablando});
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
        hablando:!!s.hablando,
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
          if(!ev.descartada) estEsc("transcribiendo","proc"); }
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
    return;
  }
  if(!texto){ apunta(`<span class="meta">(voz sin palabras)</span>`,"fuera"); return; }
  const cabecera=`<span class="quien">${escapar(nombre)}</span> <span class="dicho">«${escapar(texto)}»</span>`;
  const tiempos=`huella ${((quien&&quien.s)||0).toFixed(2)}s · stt ${(traza.stt||0).toFixed(2)}s · compuerta ${(traza.compuerta||0).toFixed(2)}s`;
  if(!decision||!decision.dirigida){
    apunta(`${cabecera} <span class="meta">no era para mí · ${tiempos}</span>`,"fuera");
    return;
  }
  apunta(`${cabecera} <span class="meta">para mí · ${tiempos}</span>`);
  if(__escucha.sinPreguntar){       // el arnes de pruebas corta aqui
    __escucha.traza.push({fase:"preguntaria",texto,hablante:quien&&quien.perfil?nombre:null});
    if(escucha.activa) estEsc("escuchando");
    return;
  }
  // Interrupcion: si estaba hablando, que se calle y atienda. Se espera a
  // que la pregunta vieja LIMPIE (cierra su contexto de audio) antes de
  // lanzar la nueva; son milisegundos y evita pisarse los globales.
  if(enPregunta&&aborto){ aborto.abort(); if(preguntaEnCurso) await preguntaEnCurso; }
  estEsc("pensando","pensando");
  const marcas={};
  // NO se espera al final del audio: los trozos que el VAD saque mientras
  // el asistente habla se procesan (asi es como se le puede interrumpir).
  lanzarPregunta(texto,{hablante:quien&&quien.perfil?nombre:null,marcas}).then(()=>{
    escucha.ultimaRespuesta=performance.now();
    if(marcas.sonido!==undefined){
      const total=(marcas.t0+marcas.sonido*1000-s.fin)/1000;
      apunta(`<span class="meta">fin de tu voz → primer sonido: ${total.toFixed(2)}s `+
             `(stt ${(traza.stt||0).toFixed(2)} + compuerta ${(traza.compuerta||0).toFixed(2)} `+
             `+ LLM y voz ${(marcas.sonido||0).toFixed(2)})</span>`);
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
// MISMO camino que el microfono (VAD -> /escuchar -> compuerta), y
// sinPreguntar corta justo antes del LLM grande y de la voz. Los usa el
// arnes de pruebas del repo; a la pagina no le estorban.
window.__escucha={traza:[],sinPreguntar:false,hablando:false,interno:escucha,
  armar(){ if(!escucha.vad) prepararVad();
           escucha.activa=true; escucha.huellas=true; estEsc("escuchando"); },
  inyectarB64(cad,rate){
    const crudo=atob(cad), n=crudo.length>>1, f=new Float32Array(n);
    for(let i=0;i<n;i++){
      let v=crudo.charCodeAt(2*i)|(crudo.charCodeAt(2*i+1)<<8);
      if(v>=32768) v-=65536;
      f[i]=v/32768;
    }
    escucha.vad.alimentar(f,rate||16000);
  },
  estado(){ return {chip:$("escEst").textContent,activa:escucha.activa,
    cola:escucha.cola.length,procesando:escucha.procesando,
    enVoz:escucha.vad?escucha.vad.enVoz:false,huellas:escucha.huellas}; }};
</script></body></html>"""


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
        # La extension del nombre es cosmetica (ffmpeg huele el contenido),
        # pero que al menos no mienta para los formatos conocidos.
        ext = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/mp4": "mp4",
               "audio/mpeg": "mp3", "audio/ogg": "ogg"}.get(
            tipo.split(";")[0].strip(), "webm")
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
            linea(fase="texto", texto=texto,
                  s=round(time.perf_counter() - t0, 3))
            if not texto:
                return linea(fase="fin", vacia=True)

            dirigida, s, crudo = decidir(
                texto, pet.get("historial"), quien.get("nombre"),
                CFG["compuerta"], CFG["ollama"], pet.get("respondio_hace_s"))
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
        # La instruccion de sistema puede venir de la pagina: es la forma de
        # dirigir al LLM sin reiniciar el puente. Vacia o en blanco, vale la
        # de serie del arranque (--sistema).
        sistema = (pet.get("sistema") or "").strip() or CFG["sistema"]
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

        # El productor manda SIEMPRE el pendiente que queda tras extraer un
        # trozo, en vez de que la pagina intente descontarlo por su cuenta.
        # Restar longitudes se desalinea en cuanto hay un espacio de mas, y el
        # texto se corrompe en pantalla. Aqui la fuente de verdad es una sola.
        def productor():
            nonlocal pendiente, n_frases, dentro
            # Con historial (escucha continua, o pagina que lo mande) el LLM
            # ve la conversacion entera y quien dice cada cosa; sin el, el
            # camino de siempre, que es el probado.
            historial = pet.get("historial") or []
            hablante = pet.get("hablante")
            if historial or hablante:
                origen = preguntar_con_historial(pet["texto"], historial,
                                                 modelo, CFG["ollama"],
                                                 sistema, hablante)
            else:
                origen = preguntar(pet["texto"], modelo, CFG["ollama"], sistema)
            try:
                for trozo in origen:
                    if parar.is_set():
                        break       # nadie escucha: no seguir gastando el LLM
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
                    pendiente += limpiar(texto)
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

        def drenar_texto():
            """Saca los tokens pendientes sin bloquear. Se llama en cada vuelta
            del bucle para que el redactado no se congele mientras baja audio."""
            while True:
                try:
                    _, s_t, texto, pend = cola_texto.get_nowait()
                except queue.Empty:
                    return
                if "token" not in visto:
                    visto.add("token"); evento(tipo="hito", hito="token", s=s_t)
                evento(tipo="token", texto=texto, pendiente=pend)

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
                while not cerrado:
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
                    break
            drenar_texto()
            for idx in emitidos:            # lo que quede a medias, cerrado
                marcar(idx, "hecho")
        except (BrokenPipeError, ConnectionResetError):
            pass                            # el navegador se fue a mitad
        except Exception as e:
            try:
                evento(tipo="error", texto=f"{type(e).__name__}: {e}")
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
    a = ap.parse_args()
    CFG.update(modelo=a.modelo, ollama=a.ollama, voz_url=a.voz_url, token=a.token,
               voz=a.voz, arranque=a.arranque, sistema=a.sistema, cfg=a.cfg,
               voz_api=a.api_url, token_api=a.token_api or a.token,
               compuerta=a.compuerta)
    global OIDO
    OIDO = Oido(a.perfiles)
    OIDO.precargar()        # ~6 s de carga del modelo, en un hilo aparte
    print(f"asistente en http://127.0.0.1:{a.puerto}")
    print(f"  LLM : {a.modelo}"
          f"{'' if a.modelo.lower().startswith('minimax') else ' via ' + a.ollama}")
    print(f"  voz : {a.voz_url}/tts/sesion/ws (una sesion por respuesta)")
    print(f"  stt : {a.api_url}/stt (el microfono de la pagina; whisper)")
    print(f"  oido: compuerta {a.compuerta} · perfiles en {a.perfiles}")
    if ws_conectar is None:
        print("  [aviso] falta el paquete 'websockets': el puente sirve la "
              "pagina pero no podra hablar.\n"
              "          arranca con pkgs/vibevoice/.venv/bin/python")
    ThreadingHTTPServer(("127.0.0.1", a.puerto), Puente).serve_forever()


if __name__ == "__main__":
    main()
