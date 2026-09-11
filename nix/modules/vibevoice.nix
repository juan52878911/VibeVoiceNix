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
      # hilos = 0 significa "que lo decida la maquina": se deja OMP_NUM_THREADS
      # SIN poner y detectar_hilos() (voz_stream.py) cuenta nucleos fisicos.
      # Exportarlo vacio no vale: OpenMP lee la variable, no su contenido.
      hilos="''${VIBEVOICE_HILOS:-${toString cfg.hilos}}"
      if [ "$hilos" != "0" ]; then
        export OMP_NUM_THREADS="$hilos"
      fi
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
      default = 0;
      description = ''
        Hilos de OpenMP. **0 = detectarlos** (lo recomendado): cuenta nucleos
        FISICOS -- no hilos logicos -- respetando el cpuset del cgroup, y a
        partir de 8 deja uno libre para el resto del stack. Asi la misma
        configuracion sirve en una maquina de 4, de 8 o de 12 hilos sin tocar
        nada. La deteccion se imprime al arrancar.

        Un numero fijo pisa la deteccion, que es lo que hay que hacer para
        comparar mediciones entre si.

        MAS NO ES MEJOR, y por eso se cuentan fisicos. Medido en la VM
        (i7-8700T, 6 fisicos / 12 logicos; RTF, menor es mejor):

          2 hilos  4,19    8 hilos  4,31
          4 hilos  4,19   10 hilos  4,46
          6 hilos  4,24   12 hilos  5,18  <- 24% PEOR que con 2

        La carga esta limitada por ancho de banda de memoria, no por computo.
        Con 2 hilos ya se satura el bus DDR4 de un solo canal, y del 7 al 12
        encima compiten por las mismas unidades AVX2 de los 6 nucleos fisicos.

        La forma de aprovechar los hilos que sobran NO es subir este numero,
        sino solapar etapas: ver services.voz-stream.solaparDecodificador.
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

        Medido con int8 en su dia: 20 pasos RTF 2,75 · 8 pasos 2,18 · 6 pasos
        2,18 · 4 pasos 2,11. De 6 a 4 solo se ganaba un 3 %, asi que 6 dejaba
        margen de calidad casi gratis.

        CON EL MOTOR OPENVINO YA NO ES ASI. Comparacion pareada, alternando 6
        y 4 en tandas seguidas para que la deriva de la maquina no sesgue el
        resultado (decodificador int4, sin solapar, 6 hilos, semilla fija):

          tanda    pasos 6   pasos 4
            1       0,964     0,890
            2       0,932     0,889
            3       0,937     0,901
          media     0,944     0,893

        Un 5,4 % consistente, y es lo que separa el "casi tiempo real" del
        "tiempo real". La cabeza de difusion pasa de 15,6 a 10,6 ms por
        fotograma; el resto del bucle no se entera.

        LA CALIDAD NO SE RESIENTE, medido con 24 clips (6 frases x 4
        semillas), transcritos con whisper:

                      WER medio   frases exactas   deriva de tono   recorte
          6 pasos       12,7 %         58 %          -3,81 st        0,000 %
          4 pasos        9,8 %         79 %          -3,13 st        0,000 %

        El unico clip descarrilado con 4 pasos sale de la semilla 42, que
        tambien descarrila TRES clips con 6 pasos: es la semilla, no los
        pasos.

        Y HACIA ARRIBA TAMPOCO COMPENSA (medido el 10-09-2026 en la VM, banco
        pareado de 6 frases x 6 semillas con la misma semilla en cada
        variante, cfg 3,0, sp-Spk1_man, WER con whisper y naturalidad con
        UTMOS22):

          pasos   WER medio   UTMOS medio   dUTMOS   mejora en   RTF
            6       11,8 %       3,547         -          -      1,057
            8       11,2 %       3,592      +0,046      22/36    1,081
           10       10,7 %       3,523      -0,024      19/36    1,137

        El umbral se fijo ANTES de medir (dUTMOS >= +0,10 en >= 24/36 clips):
        8 no llega y ademas produjo la peor alucinacion del banco; 10 baja la
        naturalidad. Cada paso cuesta 2,5-2,6 ms de cabeza por fotograma.

        LO QUE SI MUEVE LA CALIDAD ES LA SEMILLA, y por mucho: con 18 semillas
        sobre las mismas frases, la mejor da 0 % de WER y la peor 40,7 %. Entre
        6 y 10 pasos hay un punto. Ver services.voz-stream.semilla y
        docs/plan-determinismo-calidad.md.

        El defecto se queda en 6 porque bajarlo cambia la locucion (otro
        audio, otra duracion) y esa decision es del que despliega, no de la
        biblioteca. Ponlo en 4 si el objetivo es RTF < 0,9.
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
      default = 3.0;
      description = ''
        Escala del classifier-free guidance. Afecta a la CALIDAD, no a la
        velocidad: medido en un i7-8700T da RTF 3,92 (1.5), 4,02 (1.3) y 4,20
        (1.0), y a 1.0 el modelo ademas divaga (17 s de audio para un texto de
        11 s).

        3.0 y no 1.5, que fue el defecto hasta septiembre de 2026: es lo que
        ya usaba voz-stream, es el defecto del propio upstream, y esta medido
        con el banco de fidelidad (texto -> voz -> whisper -> texto, 6 frases
        x 3 repeticiones; scripts/fidelidad.py):

          cfg 1,5   WER medio 13,6 %   peor caso 85,7 %   3/6 frases inestables
          cfg 3,0   WER medio  3,6 %   peor caso 14,3 %   1/6

        Y sale gratis en tiempo: la difusion evalua las dos ramas en un lote
        de 2 pase lo que pase. VIBEVOICE_CFG lo pisa en una ejecucion.

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
