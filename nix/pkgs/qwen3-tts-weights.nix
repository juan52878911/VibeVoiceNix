# Pesos de Qwen3-TTS-12Hz-0.6B-Base (Apache 2.0) en el formato que esperan
# tanto qwen-tts (PyTorch) como el motor C: un directorio con la disposicion
# exacta del repo de Hugging Face.
#
# Se descarga fichero a fichero con hash fijo (mismo patron que
# vibevoice-weights.nix): el repo de HF puede cambiar y esto no. Los hashes se
# sacaron con `nix store prefetch-file` el 2026-09-05.
#
# El modelo Base es el que CLONA (x-vector o audio+texto en contexto) y el
# que se afina. Los CustomVoice/VoiceDesign traen voces fijas y no se usan.
{ lib, fetchurl, runCommand }:

let
  base = "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/resolve/main";

  ficheros = {
    "config.json" = "sha256-LnFMeHyO25iwVDJoXN22NK3S3k1OZF9lPWglHvcroBE=";
    "generation_config.json" = "sha256-8bkLRRPzs0xihRBJ4kkte0xZQNrxJ2+JyCuO8EEn86o=";
    # El tokenizador de texto es el de Qwen (mismos merges/vocab que el
    # Qwen2.5 que ya vendoriza vibevoice-weights.nix).
    "merges.txt" = "sha256-WZurVAdQiHdLFzP96GXVvXR8vMelR8W8EmEOh04m9eM=";
    "vocab.json" = "sha256-yhDX6fs+0YV13R4neiV5wW0QjjLydDloSvoOELFECRA=";
    "tokenizer_config.json" = "sha256-3Dwxw72u3VAWOCuzy+BzIwJnda1R9aT7VkUFmSrkpnA=";
    "preprocessor_config.json" = "sha256-793hAi6p12kov3qc1TE5E49bouRm6Dfwj2EFqxrxwRk=";
    # 1,83 GB en bf16: talker (28 capas x 1024) + code predictor + codificador
    # de locutor.
    "model.safetensors" = "sha256-GAs7EOscnxtNt4BtVHW64wccAkPCmdSZJrqx2jtpRvY=";
    # Tokenizador de voz de 12,5 Hz (codificador + decodificador acustico),
    # 682 MB.
    "speech_tokenizer/config.json" = "sha256-7mW7kByHZmSrhwfEhxV6oabuV8ZZabKPteydwhHmgWc=";
    "speech_tokenizer/configuration.json" = "sha256-a8JtZOtQJLTR2rWlI3GVi0KSVtbJ1ZeH8fUpSlTgzr0=";
    "speech_tokenizer/preprocessor_config.json" = "sha256-/LOAXll+eG1AZ3BuYC9miFJGQPjTOWeQ4uCbWUL8vfs=";
    "speech_tokenizer/model.safetensors" = "sha256-g2t7NX9epD6ImTajcJr2jf43UYgazv5Ozw29MLpXElg=";
  };

  descargas = lib.mapAttrsToList
    (nombre: hash: {
      inherit nombre;
      fichero = fetchurl { url = "${base}/${nombre}"; inherit hash; };
    })
    ficheros;
in
runCommand "qwen3-tts-12hz-0.6b-base" { } ''
  mkdir -p "$out/speech_tokenizer"
  ${lib.concatMapStringsSep "\n"
    (d: ''ln -s "${d.fichero}" "$out/${d.nombre}"'')
    descargas}
''
