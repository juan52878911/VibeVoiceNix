# Voces de Piper en espanol como descargas con hash fijo.
#
# Los hashes se midieron sobre los ficheros reales; si Hugging Face sirviera
# otra cosa, el build falla en vez de instalar algo distinto en silencio.
#
# Velocidades medidas en un i7-8700T (mismo texto, ~10 s de audio):
#   es_MX-claude-high      RTF 0.152   <- por defecto: LatAm y la mas rapida de las "high"
#   es_MX-ald-medium       RTF 0.151
#   es_ES-sharvard-medium  RTF 0.177
#   es_ES-davefx-medium    RTF 0.191
#   es_AR-daniela-high     RTF 0.408   <- la mas pesada (109 MB)
{ lib, fetchurl, runCommand }:

let
  baseUrl = "https://huggingface.co/rhasspy/piper-voices/resolve/main";

  # nombre -> { ruta en el repo; hash del .onnx; hash del .onnx.json }
  catalogo = {
    "es_MX-claude-high" = {
      ruta = "es/es_MX/claude/high";
      onnx = "sha256-PvQKcepjhSzYq35vp9Ls3PpnoLR8nEjj8Q4C7gIIPqA=";
      json = "sha256-GvyB9wPA5Ms7TXwNyglri1SpiAaAfwFwz161VXcjwS0=";
    };
    "es_MX-ald-medium" = {
      ruta = "es/es_MX/ald/medium";
      onnx = "sha256-AZs4Ayk8k+NKIG3S5To4iSCaUU54b9cUT3twGWxXm2M=";
      json = "sha256-WnFJgVjgSvyAmb/QGcfofGjrnQQlBaKxqH5cGsKxph0=";
    };
    "es_ES-davefx-medium" = {
      ruta = "es/es_ES/davefx/medium";
      onnx = "sha256-ZliwOxpsMW7kwmWpiWq8E5M1PC2eG8p9ZsLEQuIiqRc=";
      json = "sha256-Dg3ah8cy9vOHcf8nSmOA2SUvMn3Kd6opY9X7357FSEI=";
    };
    "es_ES-sharvard-medium" = {
      ruta = "es/es_ES/sharvard/medium";
      onnx = "sha256-QP6/sWecaaRQX/MR3BNuEh40GaE6KQ7yZP30Pd7dD7E=";
      json = "sha256-dDjJtpnHKwwziNrhto0/Nk3GaiFQ/lVKHBHwM3KVeyw=";
    };
    "es_AR-daniela-high" = {
      ruta = "es/es_AR/daniela/high";
      onnx = "sha256-fOsfwNqzSUGMW1SmOa6e5ZUhLXyepCIiDYQZFj1cyYU=";
      json = "sha256-rtv2lkfh11TGLs+OA2bKXxavPnaOPGtTKa9utr3jhSs=";
    };
  };

  descargar = nombre: v: {
    modelo = fetchurl {
      url = "${baseUrl}/${v.ruta}/${nombre}.onnx";
      hash = v.onnx;
    };
    config = fetchurl {
      url = "${baseUrl}/${v.ruta}/${nombre}.onnx.json";
      hash = v.json;
    };
  };

  descargas = lib.mapAttrs descargar catalogo;
in
{
  inherit descargas;

  nombres = lib.attrNames catalogo;

  # Junta las voces pedidas en un unico directorio del store, que es lo que
  # el servicio monta como VOZ_VOICES_DIR.
  paquete = voces:
    let
      elegidas = lib.filterAttrs (n: _: lib.elem n voces) descargas;
      enlaces = lib.concatStringsSep "\n" (lib.mapAttrsToList
        (nombre: f: ''
          ln -s ${f.modelo} "$out/${nombre}.onnx"
          ln -s ${f.config} "$out/${nombre}.onnx.json"
        '')
        elegidas);
      faltan = lib.subtractLists (lib.attrNames catalogo) voces;
    in
    assert lib.assertMsg (faltan == [ ])
      "voces de Piper desconocidas: ${lib.concatStringsSep ", " faltan}. Disponibles: ${lib.concatStringsSep ", " (lib.attrNames catalogo)}";
    runCommand "piper-voces-es" { } ''
      mkdir -p "$out"
      ${enlaces}
    '';
}
