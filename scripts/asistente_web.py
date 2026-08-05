#!/usr/bin/env python
"""Asistente de voz en el navegador: escribes, responde hablando.

    python scripts/asistente_web.py
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

LO QUE SE VE EN LA PAGINA
Los tres hitos que importan, en vivo: cuanto tarda el LLM en soltar el primer
token, cuanto en tener la primera frase, y cuanto hasta que suena. Medido en
la VM, la VOZ aporta ~0,8 s; el resto es el LLM. Por eso la pagina los separa:
para que se vea donde esta el tiempo de verdad.
"""
import argparse
import json
import os
import queue
import re
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from narrador import RITMO, sintetizar, trocear  # noqa: E402
from asistente import ABRE_PENSAMIENTO, CIERRA_PENSAMIENTO, limpiar, preguntar  # noqa: E402

CFG = {}

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
select{background:var(--f);color:var(--t);border:1px solid var(--b);
 border-radius:7px;padding:.5rem}
.hitos{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:.75rem;margin-top:1rem}
.h{background:var(--f);border:1px solid var(--b);border-radius:7px;padding:.7rem .85rem}
.h .n{font:600 1.35rem/1 ui-monospace,monospace;color:var(--a);font-variant-numeric:tabular-nums}
.h .e{font-size:.72rem;color:var(--s);text-transform:uppercase;letter-spacing:.08em;margin-top:.35rem}
.resp{margin-top:1rem;padding-top:1rem;border-top:1px solid var(--b);
 font-size:.98rem;line-height:1.9;min-height:2rem}
/* Los cuatro estados por los que pasa cada trozo. El color no decora: dice
   en que fase del proceso esta ese texto ahora mismo. */
.t{border-radius:4px;padding:.1rem .25rem;transition:background .25s,color .25s}
.t.pend{color:#6b7280}                                   /* el LLM aun escribe */
.t.seg{background:#2b3550;color:#a9c3f5}                  /* segmentado, en cola */
.t.sint{background:#4a3a1a;color:#f0c274;
        animation:latir 1.1s ease-in-out infinite}        /* sintetizando */
.t.son{background:#1f4033;color:#7fd6a8}                  /* sonando */
.t.fin{color:#cfd4dc}                                     /* dicho */
@keyframes latir{0%,100%{opacity:1}50%{opacity:.55}}
@media (prefers-reduced-motion:reduce){.t.sint{animation:none}}
.ley{display:flex;gap:.9rem;flex-wrap:wrap;margin-top:.8rem;font-size:.74rem;color:var(--s)}
.ley i{font-style:normal;padding:.1rem .35rem;border-radius:3px}
.est{margin-top:.75rem;font-size:.88rem;color:var(--s)}
.est.err{color:#e0725f}
</style></head><body><main>
<header><h1>Asistente de voz</h1>
<p class="sub">Escribe y responde hablando. Los tres tiempos de abajo separan
lo que tarda el modelo de lenguaje de lo que tarda la voz.</p></header>

<div class="caja">
  <textarea id="q" placeholder="¿Cómo va el despliegue de anoche?">¿Cómo va el despliegue de anoche?</textarea>
  <div class="fila">
    <button id="ir">Preguntar</button>
    <button id="parar" class="sec" hidden>Parar</button>
    <select id="modelo"></select>
    <label style="color:var(--s);font-size:.88rem">
      <input type="checkbox" id="pensar"> dejar que razone
    </label>
  </div>
  <div class="hitos">
    <div class="h"><div class="n" id="h1">—</div><div class="e">1er token</div></div>
    <div class="h"><div class="n" id="h2">—</div><div class="e">1ª frase</div></div>
    <div class="h"><div class="n" id="h3">—</div><div class="e">1er sonido</div></div>
    <div class="h"><div class="n" id="h4">—</div><div class="e">solo la voz</div></div>
  </div>
  <div class="resp" id="texto"></div>
  <div class="ley">
    <span><i class="t pend">escribiendo</i> el LLM aún redacta</span>
    <span><i class="t seg">segmentado</i> trozo cerrado, en cola</span>
    <span><i class="t sint">sintetizando</i> generando voz</span>
    <span><i class="t son">sonando</i> ya se oye</span>
  </div>
  <div class="est" id="est"></div>
</div>
</main><script>
const $=i=>document.getElementById(i);
let ctx,aborto,cabeza=0;
fetch("/modelos").then(r=>r.json()).then(m=>{
  $("modelo").innerHTML=m.map((x,i)=>`<option${i===0?" selected":""}>${x}</option>`).join("");
});
function di(t,e){$("est").className="est"+(e?" err":"");$("est").textContent=t}

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

$("ir").addEventListener("click",async()=>{
  const q=$("q").value.trim(); if(!q) return;
  $("ir").disabled=true; $("parar").hidden=false;
  ["h1","h2","h3","h4"].forEach(i=>$(i).textContent="—");
  trozos=[]; pendiente=""; pintar(); di("preguntando…");
  ctx=new AudioContext(); cabeza=0; aborto=new AbortController();
  const t0=performance.now(); let resto=new Uint8Array(0), primero=0, hitos={};
  try{
    const r=await fetch("/preguntar",{method:"POST",signal:aborto.signal,
      headers:{"content-type":"application/json"},
      body:JSON.stringify({texto:q,modelo:$("modelo").value,pensar:$("pensar").checked})});
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
              if(ev.hito==="token") $("h1").textContent=ev.s.toFixed(2)+"s";
              if(ev.hito==="frase"){ hitos.frase=ev.s; $("h2").textContent=ev.s.toFixed(2)+"s"; }
              break;
            case "token":   // el LLM escribio: solo cambia lo pendiente
              pendiente=ev.pendiente; pintar(); break;
            case "trozo":   // se cerro un trozo: pasa a tener entidad propia
              trozos.push({id:ev.id,texto:ev.texto,estado:"seg"});
              pendiente=ev.pendiente; pintar(); break;
            case "sintetizando": marca(ev.id,"sint"); break;
            case "sonando":     marca(ev.id,"son");  break;
            case "hecho":       marca(ev.id,"fin");  break;
            case "error":       di(ev.texto,true);   break;
          }
          continue;
        }
        const pares=carga.length-(carga.length%2);
        if(!pares) continue;
        const pcm=new Int16Array(carga.slice(0,pares).buffer);
        const f32=new Float32Array(pcm.length);
        for(let k=0;k<pcm.length;k++) f32[k]=pcm[k]/32768;
        if(!primero){ primero=(performance.now()-t0)/1000;
          $("h3").textContent=primero.toFixed(2)+"s";
          if(hitos.frase) $("h4").textContent=(primero-hitos.frase).toFixed(2)+"s";
          di("hablando…"); cabeza=ctx.currentTime+0.15; }
        const buf=ctx.createBuffer(1,f32.length,24000);
        buf.copyToChannel(f32,0);
        const src=ctx.createBufferSource(); src.buffer=buf; src.connect(ctx.destination);
        if(cabeza<ctx.currentTime) cabeza=ctx.currentTime;
        src.start(cabeza); cabeza+=buf.duration;
      }
      resto=d.subarray(i);
    }
    // Los datos acaban ANTES que el sonido: se genera mas rapido de lo que
    // se escucha (RTF < 1), asi que al terminar la descarga aun queda cola
    // encolada en Web Audio. Se avisa cuando de verdad se calla.
    const restante=Math.max(0,(cabeza-ctx.currentTime)*1000);
    di(restante>200?"terminando de hablar…":"listo.");
    setTimeout(()=>{ di("listo."); $("ir").disabled=false; $("parar").hidden=true;
                     ctx.close(); }, restante+250);
    return;
  }catch(e){ di(e.name==="AbortError"?"parado.":"error: "+e.message,e.name!=="AbortError"); }
  $("ir").disabled=false; $("parar").hidden=true;
  try{ ctx.close(); }catch(_){}
});
$("parar").addEventListener("click",()=>aborto&&aborto.abort());
</script></body></html>"""


class Puente(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            cuerpo = PAGINA.encode()
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
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
        else:
            self.send_error(404)

    def do_POST(self):
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

        import time
        t0 = time.time()

        # PROTOCOLO: marcos de [1 byte tipo][4 bytes longitud BE][carga].
        # tipo 0 = PCM, tipo 1 = evento JSON.
        #
        # El primer intento usaba un byte 0x01 como marca de evento y el resto
        # como PCM. No vale: el PCM contiene 0x01 constantemente, asi que el
        # navegador interpretaba audio como si fuera JSON. Con longitud
        # explicita no hay ambiguedad posible.
        import struct as _st

        def marco(tipo, carga):
            self.wfile.write(_st.pack(">BI", tipo, len(carga)) + carga)
            self.wfile.flush()

        def evento(**kw):
            marco(1, json.dumps(kw).encode())

        modelo = pet.get("modelo") or CFG["modelo"]
        # "/no_think" es un truco de qwen bajo Ollama. MiniMax manda el
        # razonamiento en bloques aparte, asi que ahi no pinta nada.
        sistema = CFG["sistema"]
        if not pet.get("pensar") and not modelo.lower().startswith("minimax"):
            sistema += " /no_think"
        # DOS colas y no una. Con una sola, el bucle principal se queda 4 s
        # bloqueado sintetizando un trozo y durante ese rato no lee nada, asi
        # que los tokens que el LLM va escribiendo se acumulan y salen de
        # golpe al terminar: en pantalla el texto dejaba de fluir despues del
        # primer trozo. Separadas, los tokens se drenan ENTRE los fotogramas
        # de audio y el redactado se ve siempre en vivo.
        cola_texto = queue.Queue()          # tokens y estado: se drena siempre
        cola_trozos = queue.Queue(maxsize=64)   # lo que hay que sintetizar
        pendiente, n_frases, dentro = "", 0, False

        # El productor manda SIEMPRE el pendiente que queda tras extraer un
        # trozo, en vez de que la pagina intente descontarlo por su cuenta.
        # Restar longitudes se desalinea en cuanto hay un espacio de mas, y el
        # texto se corrompe en pantalla. Aqui la fuente de verdad es una sola.
        def productor():
            nonlocal pendiente, n_frases, dentro
            try:
                for trozo in preguntar(pet["texto"], modelo, CFG["ollama"], sistema):
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
            """Saca los tokens pendientes sin bloquear. Se llama entre
            fotogramas de audio para que el redactado no se congele mientras
            se sintetiza."""
            while True:
                try:
                    _, s_t, texto, pend = cola_texto.get_nowait()
                except queue.Empty:
                    return
                if "token" not in visto:
                    visto.add("token"); evento(tipo="hito", hito="token", s=s_t)
                evento(tipo="token", texto=texto, pendiente=pend)

        try:
            while True:
                # Esperar un trozo SIN dejar de atender los tokens.
                item = None
                while item is None:
                    drenar_texto()
                    try:
                        item = cola_trozos.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    break
                if item is None:
                    break
                clase, s_t, dato, pend = item
                if clase == "error":
                    evento(tipo="error", texto=dato)
                    break
                idx, frase = dato
                if "frase" not in visto:
                    visto.add("frase"); evento(tipo="hito", hito="frase", s=s_t)
                evento(tipo="trozo", id=idx, texto=frase, pendiente=pend)
                evento(tipo="sintetizando", id=idx, s=time.time() - t0)
                primero = True
                for pcm in sintetizar(frase, CFG["voz_url"], CFG["token"], CFG["voz"], CFG["cfg"]):
                    if primero:
                        evento(tipo="sonando", id=idx, s=time.time() - t0)
                        primero = False
                    marco(0, pcm)
                    drenar_texto()      # <- lo que arregla el congelado
                evento(tipo="hecho", id=idx, s=time.time() - t0)
            drenar_texto()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", type=int, default=8090)
    ap.add_argument("--modelo", default=os.environ.get("ASISTENTE_MODELO", "MiniMax-M3"),
                    help="MiniMax-M3 (por defecto) o cualquier modelo de Ollama")
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--voz-url", default=os.environ.get("VOZ_STREAM_URL", "http://127.0.0.1:8082"))
    ap.add_argument("--token", default=os.environ.get("VOZ_TOKEN", ""))
    ap.add_argument("--voz", default=os.environ.get("VIBEVOICE_VOZ", "sp-Spk1_man"))
    ap.add_argument("--cfg", type=float, default=3.0,
                    help="guia CFG; 3.0 medido como el mas fiel")
    ap.add_argument("--arranque", type=int, default=15)
    ap.add_argument("--sistema", default="Responde en español, breve y natural, "
                                         "en frases cortas. Sin listas ni markdown.")
    a = ap.parse_args()
    CFG.update(modelo=a.modelo, ollama=a.ollama, voz_url=a.voz_url, token=a.token,
               voz=a.voz, arranque=a.arranque, sistema=a.sistema, cfg=a.cfg)
    print(f"asistente en http://127.0.0.1:{a.puerto}")
    print(f"  LLM : {a.modelo}"
          f"{'' if a.modelo.lower().startswith('minimax') else ' via ' + a.ollama}")
    print(f"  voz : {a.voz_url}")
    ThreadingHTTPServer(("127.0.0.1", a.puerto), Puente).serve_forever()


if __name__ == "__main__":
    main()
