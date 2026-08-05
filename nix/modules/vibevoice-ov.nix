# Motor OpenVINO: convierte el modelo a grafos compilados y baja el RTF a la mitad.
#
#   componente        PyTorch int8    OpenVINO
#   tts_lm              37-41 ms       24,5 ms
#   cabeza                4,2 ms        2,9 ms
#   acustico          165-168 ms       65,6 ms
#   -----------------------------------------------
#   RTF end-to-end          2,19          1,09
#
# El decodificador acustico era el 58% del tiempo y NO estaba limitado por
# memoria: leer sus pesos a los 17,2 GB/s medidos costaria 20-40 ms y tardaba
# 165. Ese sobrante era despacho de Python sobre decenas de convoluciones
# pequenas, y es lo que elimina un grafo compilado.
#
# POR QUE LOS IR NO VIVEN EN EL STORE
# Generarlos pica ~4,6 GB de RAM. El CT de construccion tiene 2.560 MB y el
# host esta sobresuscrito, asi que no caben en un sandbox de Nix. Se generan
# en la propia VM, una vez, mediante un servicio oneshot.
#
# Siguen siendo reproducibles -- salen de entradas fijadas: el modelo (hash
# fijo en el store), los scripts de conversion (versionados en este repo) y
# openvino/nncf clavados en el uv.lock. Lo que no es reproducible es el
# *momento* de generarlos, no el resultado.
#
# Con mas RAM en el host esto podria pasar a ser una derivacion normal.
{ config, lib, pkgs, ... }:

let
  cfg = config.services.vibevoice.openvino;
  vv = config.services.vibevoice;
  pesos = pkgs.vibevoicePesos;

  codigo = pkgs.vibevoiceOvCodigo;

  # Genera los IR si faltan. Es idempotente: si ya estan, no hace nada.
  convertir = pkgs.writeShellApplication {
    name = "vibevoice-ov-convertir";
    runtimeInputs = [ pkgs.vibevoice-env ];
    text = ''
      destino="''${VIBEVOICE_IR:-${cfg.directorioIR}}"
      export VIBEVOICE_IR="$destino"
      export VIBEVOICE_MODELO="${pesos.modelo}"
      export HF_HUB_OFFLINE=1
      # SIN anclaje de nucleos: ralentiza OpenVINO un 118% (medido).
      export OMP_NUM_THREADS="${toString vv.hilos}"
      unset OMP_PLACES OMP_PROC_BIND

      mkdir -p "$destino"

      # Cada conversion en su PROPIO proceso: cargar el fp32 pica ~4,6 GB y
      # encadenarlas en uno solo desborda los 5 GB de la VM.
      for paso in convertir_lm_estado convertir_cabeza convertir_decoder; do
        salida="$destino/.$paso.hecho"
        if [ -f "$salida" ]; then
          echo "[ov] $paso ya estaba hecho"
          continue
        fi
        echo "[ov] $paso ..."
        python ${codigo}/$paso.py
        touch "$salida"
      done

      echo "[ov] IR listos en $destino"
      # Un bucle sobre el glob y no `ls`: writeShellApplication pasa shellcheck
      # y SC2012 lo rechaza.
      for ir in "$destino"/*.xml; do
        [ -e "$ir" ] && echo "   $(basename "$ir")"
      done
    '';
  };
in
{
  options.services.vibevoice.openvino = {
    enable = lib.mkEnableOption ''
      el motor OpenVINO. Baja el RTF de 2,19 a 1,09, pero exige generar los IR
      la primera vez (~15 min y un pico de 4,6 GB de RAM)
    '';

    directorioIR = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/voz/ov";
      description = ''
        Donde viven los grafos compilados (~2,1 GB). Fuera del store porque
        generarlos no cabe en el sandbox de construccion; ver la cabecera de
        este modulo.
      '';
    };

    precisionLM = lib.mkOption {
      type = lib.types.enum [ "int4a" "int8a" "fp16" ];
      default = "int4a";
      description = ''
        Compresion del backbone TTS. `int4a` es la mas rapida y la validada;
        `int8a` cuesta ~0,13 de RTF y es el plan B si el int4 sonara raro.
      '';
    };

    precisionCabeza = lib.mkOption {
      type = lib.types.enum [ "int8" "int4" "fp16" ];
      default = "int8";
      description = ''
        Compresion de la cabeza de difusion. int8 a proposito y NO int4:
        con semilla fija se midio que el int4 sesga la decision de fin de
        frase (95 tokens frente a 84 de la base, y 83 con int8), o sea alarga
        los audios. Empata en RTF, asi que no compensa.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [{
      assertion = vv.enable;
      message = "services.vibevoice.openvino necesita services.vibevoice.enable = true";
    }];

    environment.systemPackages = [ convertir ];

    # Genera los IR antes de que arranque nada que los use.
    systemd.services.vibevoice-ov-ir = {
      description = "Genera los grafos OpenVINO de VibeVoice si faltan";
      wantedBy = [ "multi-user.target" ];
      before = lib.optional config.services.voz-stream.enable "voz-stream.service";
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStart = "${convertir}/bin/vibevoice-ov-convertir";
        # La conversion completa son ~15 min en este hardware.
        TimeoutStartSec = "45min";
        StateDirectory = "voz";
      };
      environment.VIBEVOICE_IR = toString cfg.directorioIR;
    };
  };
}
