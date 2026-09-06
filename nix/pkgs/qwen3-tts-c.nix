# Motor de inferencia en C puro para Qwen3-TTS (gabriele-mastrapasqua/qwen3-tts).
#
# POR QUE ESTE Y NO PYTORCH
# En esta CPU (i7-8700T, AVX2 sin AVX-512) el PyTorch de referencia queda muy
# por encima de tiempo real. El motor C lee los safetensors de Hugging Face
# tal cual, cuantiza a int8/int4 al vuelo, hace streaming y trae un servidor
# HTTP con /v1/tts/stream. Es la PUERTA BARATA del plan: si con el no baja de
# RTF 1,0 en la VM, el modelo va a otro hardware y no se invierte en OpenVINO.
#
# FLAGS DE ARQUITECTURA
# El Makefile autodetecta la CPU del que COMPILA (en darwin/ARM usa
# -march=native). En Nix eso es un error: el binario se construye en una
# maquina y corre en otra. Se fuerza SIMD=portable en x86 (AVX2 + FMA, que es
# lo que tiene el i7-8700T) y se deja el native solo en darwin, donde el
# binario no sale de la maquina.
{ lib, stdenv, fetchFromGitHub, openblas }:

stdenv.mkDerivation {
  pname = "qwen3-tts-c";
  version = "unstable-2026-09-01";

  src = fetchFromGitHub {
    owner = "gabriele-mastrapasqua";
    repo = "qwen3-tts";
    rev = "e56ec7e6eabbed608b13bfbd3fba431708b2077f";
    hash = "sha256-Fi8S3cQJ06yH7r6OJ5lQWeQGTHXOcCQiW3Ee7vocUxk=";
  };

  buildInputs = lib.optional stdenv.isLinux openblas;

  # El Makefile busca openblas en /usr/include/openblas; en Nix vive en el
  # store. Se le pasa la ruta por CFLAGS y el linker la encuentra por
  # buildInputs.
  makeFlags = [ "blas" "CC=${stdenv.cc.targetPrefix}cc" ]
    ++ lib.optionals stdenv.isLinux [
      "SIMD=portable"
      "EXTRA_CFLAGS=-I${openblas}/include -I${openblas}/include/openblas"
    ];

  enableParallelBuilding = true;

  # No hay `make install`: el binario queda en la raiz como qwen_tts.
  installPhase = ''
    runHook preInstall
    install -Dm755 qwen_tts "$out/bin/qwen_tts"
    # Voces y perfiles de emocion que el binario busca junto a si mismo.
    mkdir -p "$out/share/qwen3-tts"
    cp -r presets configs "$out/share/qwen3-tts/" 2>/dev/null || true
    runHook postInstall
  '';

  meta = with lib; {
    description = "Motor C de Qwen3-TTS: int8/int4, streaming, clonado y servidor HTTP";
    homepage = "https://github.com/gabriele-mastrapasqua/qwen3-tts";
    license = licenses.mit;
    platforms = platforms.unix;
    mainProgram = "qwen_tts";
  };
}
