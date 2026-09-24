#!/usr/bin/env python3
"""Puerta 0 del bucle de entrenamiento: la pasada unica de forzado.py tiene que dar las MISMAS
condiciones que generate() de Microsoft forzado con los mismos latentes (el mecanismo de
fase3_condiciones.py). Si no, el bucle entrenaria algo distinto de lo que el modelo usa al generar.

  python3 validar_forzado.py --ref ref.wav --ref-txt "..." --obj obj.wav --obj-txt "..."
"""
import argparse
import copy
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
import forzado as FZ  # noqa: E402
from auditar_encoder import encoder_comunitario  # noqa: E402


def cargar(modelo_dir, cache, dispositivo):
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference)
    from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
    proc = VibeVoiceStreamingProcessor.from_pretrained(modelo_dir)
    modelo = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        modelo_dir, dtype=torch.float32, device_map="cpu", attn_implementation="sdpa").eval()
    m = modelo.model
    m.tts_language_model.embed_tokens = m.language_model.embed_tokens
    m.acoustic_tokenizer.encoder.load_state_dict(
        {k: v.float() for k, v in encoder_comunitario(Path(cache)).items()}, strict=True)
    return proc, modelo.to(dispositivo)


def a_cpu(salida):
    """Como clonar_voz.a_cpu: el objeto de salida con su cache, para generate()."""
    return copy.deepcopy(salida)


def prefijo(modelo, tok, lat_ref, ref_txt, d):
    """La receta EXACTA de clonar_voz.py, con los latentes que se le pasan."""
    m = modelo.model
    with torch.no_grad():
        neg = torch.tensor([[tok.convert_tokens_to_ids("<|image_pad|>")]], device=d)
        neg_lm = modelo.forward_lm(input_ids=neg, attention_mask=torch.ones_like(neg), use_cache=True, return_dict=True)
        neg_tts = modelo.forward_tts_lm(input_ids=neg, attention_mask=torch.ones_like(neg),
                                        lm_last_hidden_state=neg_lm.last_hidden_state,
                                        tts_text_masks=torch.ones_like(neg), use_cache=True, return_dict=True)
        ids = torch.tensor([tok.encode(ref_txt, add_special_tokens=False)], device=d)
        con = m.acoustic_connector(lat_ref[None])
        n = con.shape[1]
        lm = modelo.forward_lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, return_dict=True)
        M = lm.last_hidden_state.shape[1]
        emb = torch.cat([con, torch.zeros(1, M, con.shape[-1], device=d)], 1)
        tts = modelo.forward_tts_lm(attention_mask=torch.ones(1, n + M, dtype=torch.long, device=d),
                                    inputs_embeds=emb, lm_last_hidden_state=lm.last_hidden_state,
                                    tts_text_masks=torch.zeros(1, n + M, dtype=torch.long, device=d),
                                    use_cache=True, return_dict=True)
    return {"lm": lm, "tts_lm": tts, "neg_lm": neg_lm, "neg_tts_lm": neg_tts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-txt", required=True)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--obj-txt", required=True)
    ap.add_argument("--modelo", default=str(Path.home() / ".cache/vibevoice-nix/modelo"))
    ap.add_argument("--cache", default=str(Path.home() / ".cache/vibevoice-nix"))
    ap.add_argument("--dispositivo", default="cpu")
    a = ap.parse_args()
    d = a.dispositivo
    proc, modelo = cargar(a.modelo, a.cache, d)
    tok = proc.tokenizer
    ref_txt = a.ref_txt if a.ref_txt.endswith("\n") else a.ref_txt + "\n"
    lat_ref = FZ.latentes(modelo, sf.read(a.ref, dtype="float32")[0], d, muestrear=False)
    lat = FZ.latentes(modelo, sf.read(a.obj, dtype="float32")[0], d, muestrear=False)
    T = lat.shape[0]
    # (a) generate() de Microsoft, forzado
    pref = prefijo(modelo, tok, lat_ref, ref_txt, d)
    estado, conds, negs, fines = {"i": 0}, [], [], []

    def forzado(condition, neg_condition, cfg_scale=3.0):
        i = estado["i"]
        estado["i"] += 1
        if i < T:
            conds.append(condition[0].detach().float().clone())
            negs.append(neg_condition[0].detach().float().clone())
        return lat[min(i, T - 1)][None]

    modelo.sample_speech_tokens = forzado
    fin_original = modelo.tts_eos_classifier

    class Espia(nn.Module):
        def forward(self, h):
            fines.append(fin_original(h).detach().float().clone())
            return torch.full((*h.shape[:-1], 1), -30.0, device=h.device)

    modelo.tts_eos_classifier = Espia()
    m = modelo.model
    decod = m.acoustic_tokenizer.decode
    m.acoustic_tokenizer.decode = lambda *args, **kw: torch.zeros(1, 1, 3200)
    entradas = proc.process_input_with_cached_prompt(
        text=a.obj_txt.strip() + "\n", cached_prompt=copy.deepcopy(pref),
        padding=True, return_tensors="pt", return_attention_mask=True)
    with torch.no_grad():
        modelo.generate(**{k: (v.to(d) if hasattr(v, "to") else v) for k, v in entradas.items()},
                        max_new_tokens=None, cfg_scale=3.0, tokenizer=tok,
                        generation_config={"do_sample": False}, verbose=False, return_speech=False,
                        all_prefilled_outputs=copy.deepcopy(pref), stop_check_fn=lambda: estado["i"] >= T,
                        show_progress_bar=False)
    modelo.tts_eos_classifier = fin_original
    m.acoustic_tokenizer.decode = decod
    ids_gen = entradas["tts_text_ids"][0].tolist()
    ids_mios = tok.encode(a.obj_txt.strip() + "\n", add_special_tokens=False)
    print(f"fichas de texto: generate {len(ids_gen)} · forzado.py {len(ids_mios)} · iguales {ids_gen == ids_mios}")
    # (b) la pasada unica
    with torch.no_grad():
        _, info = FZ.pasada(modelo, tok, {"ref_lat": lat_ref, "ref_txt": ref_txt, "lat": lat, "txt": a.obj_txt},
                            FZ.Programa(dispositivo=d))
    if info is None:
        sys.exit("forzado.py descarto el ejemplo (audio mas corto que el texto)")
    A, B = torch.stack(conds), info["cond"].float()
    n = min(len(A), len(B))
    cos = torch.nn.functional.cosine_similarity(A[:n], B[:n], dim=-1)
    err = (A[:n] - B[:n]).norm(dim=-1) / A[:n].norm(dim=-1)
    print(f"fotogramas: generate {len(A)} · forzado {len(B)} (T={T})")
    print(f"condicion positiva: coseno min {cos.min():.6f} medio {cos.mean():.6f} · error relativo max {err.max():.2e}")
    peor = int(cos.argmin())
    print(f"  peor fotograma {peor}: coseno {cos[peor]:.6f}")
    An, Bn = torch.stack(negs)[:n], info["cond_neg"].float()[:n]
    cos_n = torch.nn.functional.cosine_similarity(An, Bn, dim=-1)
    print(f"condicion negativa: coseno min {cos_n.min():.6f} medio {cos_n.mean():.6f}")
    ok = n == T and cos.min() > 0.999 and cos_n.min() > 0.999
    print("PUERTA 0:", "PASA" if ok else "NO PASA")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
