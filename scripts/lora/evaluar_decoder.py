#!/usr/bin/env python3
"""Puerta del decodificador destilado: el MISMO latente decodificado por el maestro y por el alumno.

Audio real apartado (datos/eval_audio: lectores que no entraron en el entrenamiento, con su texto en
evaluacion.json) -> codificador -> latentes -> decodificador maestro y alumno -> dos wav por clip.
Deja maestro/lote.json y alumno/lote.json para juez_lote.py (WER frente al texto, UTMOS, ECAPA frente a
los audios reales del lector) y comparar.py los empareja clip a clip.

  python3 evaluar_decoder.py --datos datos/ --alumno dec1/alumno_mejor.pt --salida evdec/
"""
import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent))
sys.path.insert(0, str(AQUI.parent.parent / "pkgs" / "vibevoice-ov"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    ap.add_argument("--alumno", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--modelo", default=str(Path.home() / ".cache/vibevoice-nix/modelo"))
    ap.add_argument("--cache", default=str(Path.home() / ".cache/vibevoice-nix"))
    a = ap.parse_args()
    d = "cuda" if torch.cuda.is_available() else "cpu"
    from entrenar import cargar_modelo
    from auditar_encoder import encoder_comunitario
    from decoder_alumno import DecodificadorAlumno
    from decoder_manual import cargar
    import forzado as FZ
    _, modelo = cargar_modelo(a.modelo, d)
    modelo.model.acoustic_tokenizer.encoder.load_state_dict(
        {k: v.float() for k, v in encoder_comunitario(Path(a.cache)).items()}, strict=True)
    modelo.eval()
    sesgo, escala = float(modelo.model.speech_bias_factor), float(modelo.model.speech_scaling_factor)
    maestro = cargar(f"{a.modelo}/model.safetensors").to(d).eval()
    e = torch.load(a.alumno, map_location="cpu")
    alumno = DecodificadorAlumno(e["escala"], e["profundidades"]).to(d).eval()
    alumno.load_state_dict(e["estado"])
    sal = Path(a.salida)
    ev = json.loads((Path(a.datos) / "evaluacion.json").read_text())
    lotes = {"maestro": [], "alumno": []}
    for quien in lotes:
        (sal / quien / "wav").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(11)
    for idioma, hs in ev.items():
        for h in hs:
            ident = f"{idioma}-{h['hablante']}"
            reales = [f["audio"] for f in h["frases"]]
            for quien in lotes:
                dest = sal / quien / "identidades" / ident
                dest.mkdir(parents=True, exist_ok=True)
                for k, w in enumerate(reales):
                    x, hz = sf.read(w, dtype="float32")
                    sf.write(str(dest / f"real{k}.wav"), x, hz)
            for k, f in enumerate(h["frases"]):
                x, _ = sf.read(f["audio"], dtype="float32")
                lat = FZ.latentes(modelo, x, d, muestrear=True)                  # escala de la difusion
                z = (lat / escala - sesgo).T[None]                                # [1, 64, T]
                with torch.no_grad():
                    for quien, dec in (("maestro", maestro), ("alumno", alumno)):
                        est = [torch.zeros(1, c, kk, device=d) for c, kk in dec.formas_estado]
                        y = dec(z, *est)[0][0, 0].float().cpu().numpy()
                        w = sal / quien / "wav" / f"{ident}__f{k}.wav"
                        sf.write(str(w), y, 24000, subtype="PCM_16")
                        lotes[quien].append({"clave": f"{ident}__f{k}", "audio": str(w), "texto": f["texto"],
                                             "idioma": idioma, "identidad": ident})
        print(f"[evdec] {idioma}: {len(hs)} lectores", flush=True)
    for quien, lote in lotes.items():
        (sal / quien / "lote.json").write_text(json.dumps(lote, ensure_ascii=False, indent=1))
    print(f"[evdec] {len(lotes['alumno'])} clips por decodificador en {sal}", flush=True)


if __name__ == "__main__":
    main()
