#!/usr/bin/env python3
"""Pruebas de MECANICA de scripts/red con pesos aleatorios: no miden calidad, miden que cada pieza hace lo que dice.

    python3 scripts/red/probar_mecanica.py --voz ~/.cache/vibevoice-nix/voces/sp-Spk1_man.pt [--modelo <ruta>]

Sin --modelo se construye la arquitectura del 0.5B con pesos al azar (37 s en 4 nucleos). Las propiedades que se
comprueban NO dependen de los pesos:
  paridad      el bucle transparente da el mismo audio que generate() de Microsoft (misma semilla)
  bifurcar     foto en el fotograma 6 y reponer: los 6 primeros identicos, la continuacion cambia con otra semilla
               y se repite bit a bit con la misma
  dirigir      con lambda 0 el audio es el de base; con lambda > 0 cambia; sumar en una rama o en las dos difiere
  enfasis      escalar la ficha k del texto solo cambia el audio DESDE la ventana que la lee (causalidad)
  negativo     un prefijo negativo con latentes propios se acepta y cambia el audio
  registro     por fotograma: condicion, negativa, latente, residual de 20 capas, p_fin
  sorpresa     la perdida v de la cabeza por fotograma existe y es finita
  atencion     masa por cabeza sobre [latentes del prefijo, texto del prefijo, generado] suma 1
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "lora"))
import modelo as MO  # noqa: E402
from bucle import Generador  # noqa: E402
import prefijo_negativo as PN  # noqa: E402
import sorpresa as SO  # noqa: E402

K = 12          # fotogramas por prueba (dos ventanas de texto)
FR = 3200       # muestras por fotograma


def igual(a, b, n=None):
    if n is not None:
        a, b = a[:n], b[:n]
    return a.shape == b.shape and torch.equal(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voz", required=True)
    ap.add_argument("--modelo", default=None)
    ap.add_argument("--texto", default="Hoy vamos a hablar de como la nube cambia la manera en que una empresa maneja sus datos.")
    a = ap.parse_args()
    t0 = time.time()
    m, tok = MO.cargar(a.modelo, aleatorio=a.modelo is None)
    # que el clasificador de fin no pare la prueba con pesos al azar
    if a.modelo is None:
        with torch.no_grad():
            m.tts_eos_classifier.__dict__.setdefault("_bias_original", None)
            for p in m.tts_eos_classifier.parameters():
                if p.dim() == 1:
                    p.fill_(-20.0)
    base = MO.prefijo(a.voz)
    ids = MO.fichas(a.texto, tok)
    print(f"[modelo listo en {time.time() - t0:.0f} s; {len(ids)} fichas; prefijo lm {base['lm'].last_hidden_state.shape[1]} "
          f"tts {base['tts_lm'].last_hidden_state.shape[1]}]", flush=True)
    resultados = {}

    def gen(**kw):
        torch.manual_seed(11)
        g = Generador(m, base, ids, **kw)
        return g, g.correr(parar_en=K)

    # 1. paridad con generate()
    t = time.time()
    _, mio = gen()
    torch.manual_seed(11)
    L_lm, L_tts = base["lm"].last_hidden_state.shape[1], base["tts_lm"].last_hidden_state.shape[1]
    vueltas = {"n": 0}

    def parar():
        vueltas["n"] += 1
        return vueltas["n"] > 2          # dos vueltas del bucle externo = 12 fotogramas
    with torch.no_grad():
        sal = m.generate(input_ids=torch.full((1, L_lm), MO.IMAGE_PAD), attention_mask=torch.ones(1, L_lm, dtype=torch.long),
                         tts_lm_input_ids=torch.full((1, L_tts), MO.IMAGE_PAD), tts_lm_attention_mask=torch.ones(1, L_tts, dtype=torch.long),
                         tts_text_ids=torch.tensor([ids]), all_prefilled_outputs=copy.deepcopy(base), tokenizer=tok,
                         cfg_scale=3.0, generation_config={"do_sample": False}, max_new_tokens=None, return_speech=True,
                         show_progress_bar=False, stop_check_fn=parar)
    suyo = sal.speech_outputs[0].flatten()
    resultados["paridad"] = igual(mio, suyo, K * FR) and suyo.shape[0] >= K * FR
    print(f"paridad: {resultados['paridad']} (mio {mio.shape[0]} muestras, generate {suyo.shape[0]}; {time.time() - t:.0f} s)", flush=True)

    # 2. bifurcar
    torch.manual_seed(11)
    g = Generador(m, base, ids)
    g.correr(parar_en=6)
    foto = g.foto()
    A = g.correr(parar_en=K)
    g.reponer(foto)
    A2 = g.correr(parar_en=K)
    g.reponer(foto)
    torch.manual_seed(101)
    B = g.correr(parar_en=K)
    resultados["bifurcar"] = (igual(A, mio) and igual(A2, A) and igual(B, A, 6 * FR) and not igual(B[6 * FR:], A[6 * FR:]))
    print(f"bifurcar: {resultados['bifurcar']} (A==base {igual(A, mio)}, repetir {igual(A2, A)}, "
          f"prefijo comun {igual(B, A, 6 * FR)}, continuacion distinta {not igual(B[6 * FR:], A[6 * FR:])})", flush=True)

    # 3. dirigir
    d = torch.randn(896)
    d = d / d.norm()

    def suma(lam, ambas):
        def f(c, n, j):
            e = lam * c.pow(2).mean(-1, keepdim=True).sqrt() * d
            return c + e, (n + e if ambas else n)
        return f
    _, s0 = gen(al_condicion=suma(0.0, True))
    _, s1 = gen(al_condicion=suma(0.1, False))
    _, s2 = gen(al_condicion=suma(0.1, True))
    resultados["dirigir"] = igual(s0, mio) and not igual(s1, mio) and not igual(s2, s1)
    print(f"dirigir: {resultados['dirigir']} (lambda 0 == base {igual(s0, mio)}, lambda 0,1 cambia {not igual(s1, mio)}, "
          f"una rama != dos ramas {not igual(s2, s1)})", flush=True)

    # 4. enfasis por ficha: la ficha 7 esta en la 2a ventana (fichas 5-9), que se lee antes del fotograma 6
    _, e = gen(ganancia_texto={7: 1.5})
    resultados["enfasis"] = igual(e, mio, 6 * FR) and not igual(e[6 * FR:], mio[6 * FR:])
    print(f"enfasis: {resultados['enfasis']} (fotogramas 0-5 iguales {igual(e, mio, 6 * FR)}, desde el 6 cambian "
          f"{not igual(e[6 * FR:], mio[6 * FR:])})", flush=True)

    # 5. registro y 6. prefijo negativo con los latentes registrados
    gr, r = gen(registrar=True, residuales=True)
    reg = gr.reg
    ok_reg = (len(reg) == K and reg[0]["cond"].shape == (896,) and reg[0]["res"].shape == (MO.CAPAS_TTS, 896)
              and reg[0]["lat"].shape == (64,) and all(0 <= x["p_fin"] <= 1 for x in reg))
    resultados["registro"] = ok_reg and igual(r, mio)
    print(f"registro: {resultados['registro']} ({len(reg)} fotogramas, residual {tuple(reg[0]['res'].shape)})", flush=True)
    lat = torch.stack([x["lat"] for x in reg])
    neg = PN.construir(m, tok, lat)
    _, n = gen(neg_tts_lm=neg)
    resultados["negativo"] = neg.last_hidden_state.shape[1] == K + 1 and not igual(n, mio)
    print(f"negativo: {resultados['negativo']} (prefijo negativo de {neg.last_hidden_state.shape[1]} posiciones; audio cambia {not igual(n, mio)})", flush=True)

    # 7. sorpresa
    s = SO.por_fotograma(m, reg, semilla=0)
    resultados["sorpresa"] = s.shape == (K,) and bool(torch.isfinite(s).all())
    print(f"sorpresa: {resultados['sorpresa']} (media {s.mean():.3f}, min {s.min():.3f}, max {s.max():.3f})", flush=True)

    # 8. atencion (eager)
    m.model.tts_language_model.config._attn_implementation = "eager"
    ga, _ = gen(registrar=True, atenciones=True)
    att = ga.reg[K - 1]["att"]
    m.model.tts_language_model.config._attn_implementation = "sdpa"
    resultados["atencion"] = att is not None and att.shape == (MO.CAPAS_TTS, 14, 3) and bool((att.sum(-1) - 1).abs().max() < 1e-3)
    if att is not None:
        print(f"atencion: {resultados['atencion']} (forma {tuple(att.shape)}; masa media prefijo-latentes {att[..., 0].mean():.2f}, "
              f"prefijo-texto {att[..., 1].mean():.2f}, generado {att[..., 2].mean():.2f})", flush=True)
    else:
        print("atencion: False (sin atenciones)")

    print("\nRESUMEN:", {k: ("pasa" if v else "FALLA") for k, v in resultados.items()}, f"({time.time() - t0:.0f} s)")
    return 0 if all(resultados.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
