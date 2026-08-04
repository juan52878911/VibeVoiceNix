{ uv2nix, pyproject-nix, pyproject-build-systems }:

final: prev:

let
  inherit (final) lib;

  # Un mismo interprete para los dos entornos: 3.12 cae dentro del
  # requires-python de ambos workspaces (voz-api >=3.11,<3.14; vibevoice <3.13).
  python = final.python312;

  # Construye un virtualenv a partir de un workspace de uv.
  #
  #   raiz       directorio con pyproject.toml + uv.lock
  #   nombre     nombre del entorno resultante
  #   extraOverrides  ajustes por paquete (deps nativas que la rueda no declara)
  mkEntorno = { raiz, nombre, extraOverrides ? (_: _: { }) }:
    let
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = raiz; };

      # sourcePreference = "wheel": las ruedas manylinux se parchean con
      # autoPatchelf. Compilar torch desde fuente en este hardware no es viable.
      overlayWorkspace = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      pythonSet =
        (final.callPackage pyproject-nix.build.packages { inherit python; })
          .overrideScope (lib.composeManyExtensions [
            pyproject-build-systems.overlays.default
            overlayWorkspace
            extraOverrides
          ]);
    in
    pythonSet.mkVirtualEnv nombre workspace.deps.default;

  # Las ruedas binarias declaran sus deps de Python, no las librerias del
  # sistema contra las que enlazan. autoPatchelf las necesita en el build.
  libsNativas = paquetes: final: prev:
    lib.genAttrs paquetes (nombre:
      prev.${nombre}.overrideAttrs (old: {
        buildInputs = (old.buildInputs or [ ]) ++ [
          final.pkgs.stdenv.cc.cc.lib # libstdc++
          final.pkgs.zlib
        ];
      })
    );
in
{
  # ------------------------------------------------------------------
  # API de voz: FastAPI + Piper. Es el camino de produccion, y es ligero.
  # ------------------------------------------------------------------
  voz-api = mkEntorno {
    raiz = ../pkgs/voz-api;
    nombre = "voz-api";
    extraOverrides = final': prev':
      (libsNativas [ "onnxruntime" ] final' prev');
  };

  # ------------------------------------------------------------------
  # VibeVoice: torch CPU + el paquete de Microsoft fijado por commit.
  # Pesa ~2 GB y solo se usa para experimentar, no para servir.
  # ------------------------------------------------------------------
  vibevoice-env = mkEntorno {
    raiz = ../pkgs/vibevoice;
    nombre = "vibevoice-env";
    extraOverrides = final': prev':
      (libsNativas [ "torch" "numpy" "scipy" "llvmlite" ] final' prev');
  };

  # Voces de Piper y pesos de VibeVoice: descargas con hash fijo.
  vozPiperVoces = final.callPackage ./pkgs/piper-voices.nix { };
  vibevoicePesos = final.callPackage ./pkgs/vibevoice-weights.nix { };
}
