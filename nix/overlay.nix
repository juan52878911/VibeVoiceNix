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
  #
  #   paquetes  lista de nombres que solo necesitan libstdc++ y zlib
  #   extra     nombre -> librerias adicionales, para los que piden algo mas
  libsNativas = { paquetes ? [ ], extra ? { } }: final: prev:
    let
      base = [ final.pkgs.stdenv.cc.cc.lib final.pkgs.zlib ];
      conLibs = nombre: adicionales:
        prev.${nombre}.overrideAttrs (old: {
          buildInputs = (old.buildInputs or [ ]) ++ base ++ adicionales;
        });
    in
    lib.genAttrs paquetes (nombre: conLibs nombre [ ])
    // lib.mapAttrs (nombre: libs: conLibs nombre (libs final.pkgs)) extra;

  # Paquetes que se construyen desde fuente y no declaran su backend de
  # construccion. uv2nix compila sin aislamiento, asi que hay que ponerselo:
  # sin esto VibeVoice falla con "No module named 'setuptools'".
  sistemasDeConstruccion = mapa: final: prev:
    lib.mapAttrs
      (nombre: sistemas:
        prev.${nombre}.overrideAttrs (old: {
          nativeBuildInputs =
            (old.nativeBuildInputs or [ ]) ++ final.resolveBuildSystem sistemas;
        }))
      mapa;

  # soundfile carga libsndfile con dlopen por NOMBRE en tiempo de ejecucion.
  # autoPatchelf no puede arreglarlo (no es una dependencia enlazada), asi que
  # se sustituye por la ruta absoluta del store, igual que hace nixpkgs.
  arreglarSoundfile = final: prev: {
    soundfile = prev.soundfile.overrideAttrs (old: {
      postInstall = (old.postInstall or "") + ''
        archivo="$out/${final.python.sitePackages}/soundfile.py"
        substituteInPlace "$archivo" \
          --replace-fail "_explicit_libname = 'libsndfile.so'" \
            "_explicit_libname = '${final.pkgs.libsndfile.out}/lib/libsndfile.so'"
      '';
    });
  };

  # Encadena varios conjuntos de ajustes sobre el mismo scope.
  componer = lib.composeManyExtensions;
in
{
  # ------------------------------------------------------------------
  # API de voz: FastAPI + Piper. Es el camino de produccion, y es ligero.
  # ------------------------------------------------------------------
  voz-api = mkEntorno {
    raiz = ../pkgs/voz-api;
    nombre = "voz-api";
    extraOverrides = libsNativas { paquetes = [ "onnxruntime" ]; };
  };

  # ------------------------------------------------------------------
  # VibeVoice: torch CPU + el paquete de Microsoft fijado por commit.
  # Pesa ~2 GB y solo se usa para experimentar, no para servir.
  # ------------------------------------------------------------------
  vibevoice-env = mkEntorno {
    raiz = ../pkgs/vibevoice;
    nombre = "vibevoice-env";
    extraOverrides = componer [
      (libsNativas {
        paquetes = [ "torch" "numpy" "scipy" "llvmlite" ];
        # numba enlaza el planificador de hilos de oneTBB, que la rueda no
        # declara: pkgs.tbb es quien trae libtbb.so.12.
        extra.numba = p: [ p.tbb ];
        # La rueda de OpenVINO incluye su plugin de GPU, que enlaza
        # libOpenCL.so.1. Aqui NO se usa -se midio que la iGPU es 2,5x mas
        # lenta que la CPU-, pero autoPatchelf exige que la dependencia
        # exista para dar el paquete por bueno.
        extra.openvino = p: [ p.ocl-icd ];
      })
      # VibeVoice se instala desde git y su pyproject usa setuptools.build_meta,
      # pero uv2nix construye sin aislamiento y no se lo encuentra.
      (sistemasDeConstruccion { vibevoice = { setuptools = [ ]; }; })
      arreglarSoundfile
    ];
  };

  # Voces de Piper y pesos de VibeVoice: descargas con hash fijo.
  vozPiperVoces = final.callPackage ./pkgs/piper-voices.nix { };
  vibevoicePesos = final.callPackage ./pkgs/vibevoice-weights.nix { };

  # Scripts de conversion e inferencia con OpenVINO. Van sueltos en el store y
  # se importan por PYTHONPATH, en vez de empaquetarse: dependen de importarse
  # entre si por nombre de modulo (decoder_manual, qwen2_manual, motor).
  vibevoiceOvCodigo = final.runCommand "vibevoice-ov-codigo" { } ''
    mkdir -p "$out"
    cp ${../pkgs/vibevoice-ov}/*.py "$out/"
  '';
}
