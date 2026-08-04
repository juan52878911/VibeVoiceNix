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
  # del repo y el script ya parcheado. El script trae la ruta de las voces
  # sustituida, asi que no hace falta preparar ningun directorio.
  vibevoice = pkgs.writeShellApplication {
    name = "vibevoice";
    runtimeInputs = [ pkgs.vibevoice-env pkgs.ffmpeg ];
    text = ''
      export OMP_NUM_THREADS="''${VIBEVOICE_HILOS:-${toString cfg.hilos}}"
      ${lib.optionalString cfg.anclarNucleos ''
        export OMP_PLACES="''${OMP_PLACES:-cores}"
        export OMP_PROC_BIND="''${OMP_PROC_BIND:-close}"
      ''}
      # El modelo y el tokenizador estan en el store: nada que descargar.
      export HF_HUB_OFFLINE=1
      # El modelo pinta una barra de tqdm por token que hace ilegible la
      # salida y la vuelve inutil en una tuberia. VIBEVOICE_PROGRESO=1 la trae
      # de vuelta si hace falta ver el avance de una generacion larga.
      if [ -z "''${VIBEVOICE_PROGRESO:-}" ]; then
        export TQDM_DISABLE=1
      fi
      export VIBEVOICE_MODELO="${pesos.modelo}"
      export VIBEVOICE_VOCES="${pesos.voces}"
      export VIBEVOICE_PASOS="''${VIBEVOICE_PASOS:-${toString cfg.pasosDifusion}}"
      export VIBEVOICE_VOZ="''${VIBEVOICE_VOZ:-${cfg.vozDefecto}}"

      exec python ${pesos.inferencia}/bin/vibevoice-cli.py \
        --cfg-scale "''${VIBEVOICE_CFG:-${toString cfg.cfgScale}}" \
        ${lib.optionalString (!cfg.cuantizar) "--sin-cuantizar"} \
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
      default = 6;
      description = ''
        Hilos de OpenMP. 6 = los nucleos FISICOS del i7-8700T. Medido en la VM
        (RTF, menor es mejor):

          2 hilos  4,19    8 hilos  4,31
          4 hilos  4,19   10 hilos  4,46
          6 hilos  4,24   12 hilos  5,18  <- 24% PEOR que con 2

        Mas hilos empeora: la carga esta limitada por ancho de banda de
        memoria, no por computo. Con 2 hilos ya se satura el bus DDR4 de un
        solo canal, y del 7 al 12 encima compiten por las mismas unidades AVX2
        de los 6 nucleos fisicos.
      '';
    };

    anclarNucleos = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Ancla cada hilo a un nucleo fisico (OMP_PLACES=cores). Evita que dos
        hilos compartan unidad vectorial. Medido: 4,04 frente a 4,24 con 6
        hilos, un 3% gratis y sin tocar la calidad.
      '';
    };

    cuantizar = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        int8 dinamico en las capas Linear. Medido en la VM: RTF 5,39 -> 2,75 a
        20 pasos, y 2,18 a 6 pasos. El usuario comparo las muestras y NO
        distingue la salida int8 de la fp32.

        Funciona pese a que esta CPU no tiene VNNI (la multiplicacion int8 va
        emulada) porque divide por cuatro los BYTES de peso que hay que traer
        de RAM, y el cuello es justo ese.

        Coste: el pico de memoria al cargar sube a ~4,6 GB, porque hay que
        materializar el fp32 antes de convertirlo. Con menos de 5 GB de RAM en
        la maquina, esto puede acabar en OOM.
      '';
    };

    pasosDifusion = lib.mkOption {
      type = lib.types.int;
      default = 6;
      description = ''
        Pasos del muestreador de difusion. El modelo usa
        DPMSolverMultistepScheduler, disenado para pocos pasos, y viene
        configurado a 20.

        Medido con int8: 20 pasos RTF 2,75 · 8 pasos 2,18 · 6 pasos 2,18 ·
        4 pasos 2,11. De 6 a 4 solo se gana un 3%, asi que 6 deja margen de
        calidad casi gratis.

        Por debajo de 6 apenas se gana: la cabeza de difusion (84 MB) deja de
        dominar y pasa a mandar el backbone (869 MB), que no depende de los
        pasos.
      '';
    };

    vozDefecto = lib.mkOption {
      type = lib.types.str;
      default = "sp-Spk1_man";
      description = ''
        Hablante por defecto. Las voces en espanol del modelo son
        sp-Spk1_man y sp-Spk0_woman, ambas experimentales segun Microsoft.
      '';
    };

    cfgScale = lib.mkOption {
      type = lib.types.float;
      default = 1.5;
      description = ''
        Escala del classifier-free guidance. Afecta a la CALIDAD, no a la
        velocidad: medido en un i7-8700T da RTF 3,92 (1.5), 4,02 (1.3) y 4,20
        (1.0), y a 1.0 el modelo ademas divaga (17 s de audio para un texto de
        11 s). Dejalo en 1.5.

        El motivo de que no acelere es que sample_speech_tokens concatena
        siempre condicional e incondicional en un mismo batch, sin rama que se
        salte el segundo. Parchearlo tampoco sirve: se probo y dio RTF 3,90,
        porque con dim 896 y batch 2 el cuello es el ancho de banda de memoria
        y no los FLOPs, asi que la segunda mitad del batch sale casi gratis.
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
