# Pesos de VibeVoice-Realtime-0.5B y los recursos del repo que el paquete
# Python NO incluye.
#
# El pyproject de VibeVoice solo empaqueta `vibevoice*` y `vllm_plugin*`, asi
# que las voces (demo/voices/streaming_model/*.pt) y el script de inferencia se
# traen aparte desde el mismo commit que fija el uv.lock.
#
# Importante sobre idiomas: el 1.5B y el Large-7B son ingles+chino. Solo este
# Realtime-0.5B tiene voces en espanol (sp-Spk0_woman / sp-Spk1_man), anadidas
# por Microsoft en diciembre de 2025 y marcadas como experimentales.
{ lib, fetchurl, fetchFromGitHub, runCommand, jq }:

let
  # Mismo commit que fija pkgs/vibevoice/uv.lock. Si cambia uno, cambia el otro.
  commit = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600";

  repo = fetchFromGitHub {
    owner = "microsoft";
    repo = "VibeVoice";
    rev = commit;
    hash = "sha256-v/QpYOmUYwAi8s0QYuGbZkM8JYa0OC6zAbSqLlEhDQA=";
  };

  baseHF = "https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B/resolve/main";

  ficheros = {
    "config.json" = "sha256-yu4mkeeQsEBUu+FKdTtAFJ+nwMFvrbWNmt9UEjQ9z1c=";
    # 1,9 GB en fp32: es el motivo de que el servicio pida ~4 GB de RAM.
    "model.safetensors" = "sha256-d1ixULgTnetIrB/28YH3Rcj+3VURIy/ZdLPrIX2DtRQ=";
    "preprocessor_config.json" = "sha256-6/UUtdMKAS5a4A2aGdAec141sndow5JtmAgV24+nQuU=";
  };

  # El repo del modelo NO incluye tokenizador: vibevoice_streaming_processor.py
  # cae a "Qwen/Qwen2.5-1.5B" y se lo baja de Hugging Face en tiempo de
  # ejecucion. Eso rompe la reproducibilidad y ademas falla en una maquina
  # limpia sin red (o con HF_HUB_OFFLINE), asi que se vendoriza aqui.
  #
  # Con tokenizer.json presente se usa el tokenizador RAPIDO; sin el,
  # transformers cae al lento y exige protobuf + sentencepiece.
  baseQwen = "https://huggingface.co/Qwen/Qwen2.5-1.5B/resolve/main";

  tokenizador = {
    "tokenizer.json" = "sha256-wDghF+oynN8JcEETL21zWSS2l5JNb2/DlFcT6WzodTk=";
    "tokenizer_config.json" = "sha256-yR78oVzv9unulCTbWKb1nNQSlOVQqGy9B+PB+1ALNPk=";
    "vocab.json" = "sha256-yhDX6fs+0YV13R4neiV5wW0QjjLydDloSvoOELFECRA=";
    "merges.txt" = "sha256-WZurVAdQiHdLFzP96GXVvXR8vMelR8W8EmEOh04m9eM=";
  };

  descargas = lib.mapAttrsToList
    (nombre: hash: {
      inherit nombre;
      fichero = fetchurl { url = "${baseHF}/${nombre}"; inherit hash; };
    })
    ficheros;

  # El nombre de la derivacion DEBE contener "qwen": el procesador comprueba
  # `if 'qwen' in language_model_pretrained_name.lower()` y si no, lanza
  # ValueError. Como le pasamos una ruta del store, esa ruta lleva el nombre.
  tokenizadorQwen = runCommand "qwen2.5-1.5b-tokenizer" { } ''
    mkdir -p "$out"
    ${lib.concatStringsSep "\n" (lib.mapAttrsToList
      (nombre: hash: ''
        ln -s ${fetchurl { url = "${baseQwen}/${nombre}"; inherit hash; }} "$out/${nombre}"
      '')
      tokenizador)}
  '';
in
rec {
  inherit repo;

  inherit tokenizadorQwen;

  # Directorio en formato "modelo de Hugging Face": se le pasa tal cual como
  # --model_path y evita que el servicio tenga que salir a la red.
  #
  # Se PARCHEA la config para apuntar el tokenizador al store. Sin esto,
  # vibevoice_streaming_processor.py cae al literal "Qwen/Qwen2.5-1.5B" y se lo
  # baja de Hugging Face: en una maquina limpia con HF_HUB_OFFLINE eso revienta
  # con un TypeError de vocab_file=None. Copiar el tokenizador dentro del
  # modelo NO basta, porque el procesador ni mira ese directorio.
  #
  # El fichero que lee es preprocessor_config.json (from_pretrained hace
  # os.path.join(ruta, "preprocessor_config.json")). Se parchean los dos por si
  # alguna ruta del codigo usa el otro.
  modelo = runCommand "vibevoice-realtime-0.5b" { nativeBuildInputs = [ jq ]; } ''
    mkdir -p "$out"
    ${lib.concatMapStringsSep "\n"
      (d: lib.optionalString (!lib.elem d.nombre [ "config.json" "preprocessor_config.json" ])
        ''ln -s ${d.fichero} "$out/${d.nombre}"'')
      descargas}

    ${lib.concatMapStringsSep "\n"
      (nombre: ''
        jq --arg tok "${tokenizadorQwen}" \
          '.language_model_pretrained_name = $tok' \
          ${(lib.findFirst (d: d.nombre == nombre) null descargas).fichero} \
          > "$out/${nombre}"
      '')
      [ "config.json" "preprocessor_config.json" ]}

    # Comprobacion: si el campo no queda puesto, el fallo aparece mucho mas
    # tarde y con un error que no lo señala.
    for f in config.json preprocessor_config.json; do
      grep -q "language_model_pretrained_name" "$out/$f"
    done
  '';

  # Voces preentrenadas (.pt). Las de espanol son sp-Spk0_woman y sp-Spk1_man.
  voces = runCommand "vibevoice-voces" { } ''
    mkdir -p "$out"
    cp -r ${repo}/demo/voices/streaming_model/. "$out/"
  '';

  # CLI propia, en pkgs/vibevoice-cli/. Antes se parcheaba el demo de Microsoft
  # con tres `substitute` encadenados (ruta de voces, weights_only, y ningun
  # control de cuantizacion); cualquier cambio suyo rompia el build. Un fichero
  # propio depende solo de la API publica del modelo.
  inferencia = runCommand "vibevoice-cli" { } ''
    mkdir -p "$out/bin"
    cp ${../../pkgs/vibevoice-cli/vibevoice_cli.py} "$out/bin/vibevoice-cli.py"
    cp ${../../pkgs/vibevoice-cli/voz_stream.py} "$out/bin/voz-stream.py"
    # La pagina de prueba va JUNTO al servidor: la sirve el propio servicio en
    # "/" porque desde otro origen el navegador la bloquearia por CORS.
    # voz_stream.py la busca con Path(__file__).with_name(...).
    cp ${../../pkgs/vibevoice-cli/prueba.html} "$out/bin/prueba.html"
    chmod +x "$out/bin/vibevoice-cli.py" "$out/bin/voz-stream.py"
  '';
}
