#!/usr/bin/env python
"""Asistente de voz en el navegador: escribes, responde hablando.

    python scripts/asistente_web.py
    # y abre http://127.0.0.1:8090

Hace de puente entre tres cosas que ya funcionan por separado:

    navegador  ->  este puente  ->  Ollama        (el texto)
                                ->  voz-stream    (la voz)

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
from asistente import ABRE_PENSAMIENTO, CIERRA_PENSAMIENTO, limpiar, preguntar_a_ollama  # noqa: E402

CFG = {}

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
.resp{margin-top:1rem;padding-top:1rem;border-top:1px solid var(--b);color:var(--s);
 font-size:.94rem;white-space:pre-wrap;min-height:1.5rem}
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
  <div class="est" id="est"></div>
</div>
</main><script>
const $=i=>document.getElementById(i);
let ctx,aborto,cabeza=0;
fetch("/modelos").then(r=>r.json()).then(m=>{
  $("modelo").innerHTML=m.map(x=>`<option${x==="qwen3:1.7b"?" selected":""}>${x}</option>`).join("");
});
function di(t,e){$("est").className="est"+(e?" err":"");$("est").textContent=t}

$("ir").addEventListener("click",async()=>{
  const q=$("q").value.trim(); if(!q) return;
  $("ir").disabled=true; $("parar").hidden=false;
  ["h1","h2","h3","h4"].forEach(i=>$(i).textContent="—");
  $("texto").textContent=""; di("preguntando…");
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
          if(ev.hito){ hitos[ev.hito]=ev.s;
            if(ev.hito==="token") $("h1").textContent=ev.s.toFixed(2)+"s";
            if(ev.hito==="frase") $("h2").textContent=ev.s.toFixed(2)+"s"; }
          if(ev.texto) $("texto").textContent+=ev.texto;
          if(ev.error) di(ev.error,true);
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
            try:
                d = json.load(urllib.request.urlopen(f"{CFG['ollama']}/api/tags", timeout=5))
                ms = sorted(m["name"] for m in d.get("models", []))
            except Exception:
                ms = [CFG["modelo"]]
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

        sistema = CFG["sistema"] if pet.get("pensar") else CFG["sistema"] + " /no_think"
        cola, pendiente, n_frases, dentro = queue.Queue(maxsize=64), "", 0, False

        def productor():
            nonlocal pendiente, n_frases, dentro
            try:
                for trozo in preguntar_a_ollama(pet["texto"], pet.get("modelo") or CFG["modelo"],
                                                CFG["ollama"], sistema):
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
                    cola.put(("token", time.time() - t0, texto))
                    pendiente += limpiar(texto)
                    frases, pendiente = trocear(pendiente, primera=n_frases == 0,
                                                minimo_primera=CFG["arranque"])
                    for f in frases:
                        n_frases += 1
                        cola.put(("frase", time.time() - t0, f))
                frases, _ = trocear(pendiente, forzar_final=True, primera=n_frases == 0,
                                    minimo_primera=CFG["arranque"])
                for f in frases:
                    n_frases += 1
                    cola.put(("frase", time.time() - t0, f))
            except Exception as e:
                cola.put(("error", 0, f"{type(e).__name__}: {e}"))
            cola.put(None)

        threading.Thread(target=productor, daemon=True).start()
        visto = set()
        try:
            while True:
                item = cola.get()
                if item is None:
                    break
                clase, s, dato = item
                if clase == "error":
                    evento(error=dato)
                    break
                if clase == "token":
                    if "token" not in visto:
                        visto.add("token"); evento(hito="token", s=s)
                    evento(texto=dato)
                    continue
                if "frase" not in visto:
                    visto.add("frase"); evento(hito="frase", s=s)
                for pcm in sintetizar(dato, CFG["voz_url"], CFG["token"], CFG["voz"], 1.5):
                    marco(0, pcm)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", type=int, default=8090)
    ap.add_argument("--modelo", default=os.environ.get("OLLAMA_MODELO", "qwen3:1.7b"))
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--voz-url", default=os.environ.get("VOZ_STREAM_URL", "http://127.0.0.1:8082"))
    ap.add_argument("--token", default=os.environ.get("VOZ_TOKEN", ""))
    ap.add_argument("--voz", default=os.environ.get("VIBEVOICE_VOZ", "sp-Spk1_man"))
    ap.add_argument("--arranque", type=int, default=15)
    ap.add_argument("--sistema", default="Responde en español, breve y natural, "
                                         "en frases cortas. Sin listas ni markdown.")
    a = ap.parse_args()
    CFG.update(modelo=a.modelo, ollama=a.ollama, voz_url=a.voz_url, token=a.token,
               voz=a.voz, arranque=a.arranque, sistema=a.sistema)
    print(f"asistente en http://127.0.0.1:{a.puerto}")
    print(f"  LLM : {a.ollama} ({a.modelo})")
    print(f"  voz : {a.voz_url}")
    ThreadingHTTPServer(("127.0.0.1", a.puerto), Puente).serve_forever()


if __name__ == "__main__":
    main()
