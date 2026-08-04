# VibeVoice-Realtime-0.5B: el laboratorio, no el camino de produccion.
#
# Por que no sirve para responder en vivo, medido en un i7-8700T:
#   RTF 4,80x  — 44 s de computo para 9 s de audio
#   pico de RAM 3,9 GB (el modelo son 1,9 GB en fp32 y en CPU no baja de ahi)
# Piper hace lo mismo a RTF 0,042. La diferencia es de ~100x.
#
# Sobre el espanol: el 1.5B y el Large-7B estan entrenados solo con ingles y
# chino. Este Realtime-0.5B es el unico con voces en espanol (sp-Spk0_woman y
# sp-Spk1_man), anadidas en diciembre de 2025 y marcadas como experimentales
# por el propio Microsoft.
#
# No se expone como servicio: se instala una orden que se lanza a mano.
{ config, lib, pkgs, ... }:

let
  cfg = config.services.vibevoice;
  pesos = pkgs.vibevoicePesos;

  # Envoltorio que cablea el entorno: modelo local (sin salir a la red), voces
  # del repo y el script ya parcheado.
  vibevoice = pkgs.writeShellApplication {
    name = "vibevoice";
    runtimeInputs = [ pkgs.vibevoice-env pkgs.ffmpeg ];
    text = ''
      # Directorio de trabajo: el script busca las voces en ./demo/voices/...
      trabajo="$(mktemp -d)"
      trap 'rm -rf "$trabajo"' EXIT
      mkdir -p "$trabajo/demo/voices"
      ln -s ${pesos.voces} "$trabajo/demo/voices/streaming_model"

      cd "$trabajo"
      export OMP_NUM_THREADS="''${VIBEVOICE_HILOS:-${toString cfg.hilos}}"
      export HF_HUB_OFFLINE=1

      exec python ${pesos.inferencia}/bin/vibevoice-inferencia.py \
        --model_path ${pesos.modelo} \
        --device cpu \
        --cfg_scale "''${VIBEVOICE_CFG:-${toString cfg.cfgScale}}" \
        "$@"
    '';
  };
in
{
  options.services.vibevoice = {
    enable = lib.mkEnableOption ''
      las herramientas de VibeVoice (laboratorio de TTS). Instala la orden
      `vibevoice`; no levanta ningun servicio
    '';

    hilos = lib.mkOption {
      type = lib.types.int;
      default = 8;
      description = "Hilos de OpenMP para la inferencia en CPU.";
    };

    cfgScale = lib.mkOption {
      type = lib.types.float;
      default = 1.5;
      description = ''
        Escala del classifier-free guidance. Con CFG cada paso de difusion hace
        dos pasadas (condicional e incondicional), asi que bajarlo a 1.0 casi
        duplica la velocidad a cambio de expresividad. 1.5 es el valor original.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    environment.systemPackages = [ vibevoice ];

    # 1,9 GB de modelo mas el runtime dan ~3,9 GB de pico. Sin swap, una
    # generacion en una VM justa se lleva por delante al que pida memoria.
    assertions = [{
      assertion = config.swapDevices != [ ];
      message = ''
        services.vibevoice necesita ~4 GB de RAM durante la generacion y esta
        configuracion no define swap. Anade swapDevices o desactiva vibevoice.
      '';
    }];
  };
}
