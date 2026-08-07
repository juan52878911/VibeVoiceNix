# TTS con respuesta en streaming: se oye el principio mientras se genera el resto.
#
# POST /tts/stream -> audio/wav troceado. Medido en la VM:
#
#   primer sonido    23,21 s  ->  0,20 s     (116x menos espera)
#   tiempo total     23,21 s  ->  22,42 s    (sin sobrecoste)
#
# El audio es bit a bit identico al de la generacion normal: mismo md5. No es
# una aproximacion ni una version degradada, es el mismo resultado entregado
# segun se produce.
#
# POR QUE UN SERVICIO APARTE Y NO DENTRO DE voz-api
# Este carga VibeVoice (~2,3 GB residentes); voz-api solo tiene las voces de
# Piper (~100 MB) y responde en decimas de segundo. Juntarlos haria que una
# sintesis pesada bloqueara las notas de voz rapidas, y la regla en esta VM es
# un modelo por proceso.
{ config, lib, pkgs, ... }:

let
  cfg = config.services.voz-stream;
  vv = config.services.vibevoice;
  ov = config.services.vibevoice.openvino;
  pesos = pkgs.vibevoicePesos;
in
{
  options.services.voz-stream = {
    enable = lib.mkEnableOption ''
      el servicio de TTS en streaming. Necesita ~2,5 GB de RAM en regimen y
      hace un pico de ~4,6 GB al cargar (materializa el fp32 antes de
      cuantizarlo), asi que conviene mirar la memoria libre antes de activarlo
    '';

    puerto = lib.mkOption {
      type = lib.types.port;
      default = 8082;
      description = "Puerto HTTP. El 8080 lo usa voz-api y el 8081 whisper.";
    };

    direccion = lib.mkOption {
      type = lib.types.str;
      default = "0.0.0.0";
      description = "Interfaz de escucha.";
    };

    abrirCortafuegos = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Abre el puerto en la LAN. Con el tunel activo no hace falta: la red
        de WireGuard ya llega, y asi el servicio no queda expuesto en la LAN.
      '';
    };

    ficheroToken = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = "/var/lib/voz/token.env";
      description = ''
        Fichero con `VOZ_TOKEN=...`. Lo natural es reutilizar el mismo que
        voz-api para no manejar dos credenciales.
      '';
    };

    solaparDecodificador = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Saca el decodificador acustico del camino critico y lo corre en su
        propio hilo, a la vez que el resto del bucle.

        ES LA FORMA DE APROVECHAR LOS HILOS QUE SOBRAN. Subir `hilos` no vale
        -- esta medido que empeora --, porque una sola etapa ya satura el bus
        de memoria. Solapar DOS etapas si suma, porque no compiten por lo
        mismo: el bucle (tts_lm + cabeza) se pasa el rato leyendo pesos y el
        decodificador es convolucion, mas densa en computo.

        POR QUE ES SEGURO: el decodificador es un SUMIDERO. En el bucle de
        Microsoft la realimentacion pasa por acoustic_connector(speech_latent);
        el audio decodificado solo se guarda y se emite, no vuelve a entrar en
        el modelo. Se mantiene un unico hilo trabajador y una cola FIFO, asi
        que el orden de las llamadas es el mismo que en la version sincrona --
        que es lo que hace que el audio salga IDENTICO BIT A BIT.

        MEDIDO en un Apple M4 (motor torch-int8, 12 s de audio, semilla 11),
        RTF con y sin solapar, mismo md5 en las 20 pasadas:

          hilos      sincrono   solapado   gana
            2          1,038      0,893    14%
            4          0,700      0,594    15%
            6          0,739      0,587    21%
            8          0,785      0,644    18%
           10          0,775      0,630    19%

        Cuesta ~40 ms mas de espera al primer sonido (0,12 -> 0,16 s), que es
        la profundidad de la tuberia.

        PENDIENTE DE MEDIR EN EL i7-8700T de la VM, que es donde vive el motor
        OpenVINO. La ganancia deberia ser parecida o mayor -- alli el
        decodificador pesa mas en el reparto --, pero eso hay que verlo, no
        suponerlo.
      '';
    };

    hilosDecodificador = lib.mkOption {
      type = lib.types.int;
      default = 0;
      description = ''
        Hilos para el decodificador cuando va solapado; el resto son para el
        bucle. 0 = la mitad de `services.vibevoice.hilos`.

        Solo el motor OpenVINO reparte de verdad: INFERENCE_NUM_THREADS es por
        modelo compilado. En el camino torch el pool intra-op es uno para todo
        el proceso y este numero solo sirve de documentacion.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [{
      assertion = vv.enable;
      message = ''
        services.voz-stream necesita services.vibevoice.enable = true: usa su
        mismo modelo, sus voces y su configuracion de pasos de difusion.
      '';
    }];

    systemd.services.voz-stream = {
      description = "TTS en streaming (VibeVoice)";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];

      environment = {
        VIBEVOICE_MODELO = "${pesos.modelo}";
        VIBEVOICE_VOCES = "${pesos.voces}";
        VIBEVOICE_PASOS = toString vv.pasosDifusion;
        VIBEVOICE_VOZ = vv.vozDefecto;
        VOZ_STREAM_HOST = cfg.direccion;
        VOZ_STREAM_PUERTO = toString cfg.puerto;
        HF_HUB_OFFLINE = "1";
        # glibc crea una arena por hilo y no devuelve lo liberado; con 6 hilos
        # eso fragmenta cientos de MB en un servicio que ya va justo de RAM.
        # Con el decodificador solapado hay un hilo mas que asigna de verdad,
        # asi que este tope importa mas que antes, no menos.
        MALLOC_ARENA_MAX = "2";
        VIBEVOICE_MOTOR = if ov.enable then "openvino" else "torch";
        VIBEVOICE_SOLAPAR_DECODER = if cfg.solaparDecodificador then "1" else "0";
        VIBEVOICE_HILOS_DECODER = toString cfg.hilosDecodificador;
      }
      # hilos = 0 -> no se pone la variable y voz_stream.py cuenta nucleos
      # fisicos. Ponerla vacia NO vale: OpenMP mira si existe, no su valor.
      // lib.optionalAttrs (vv.hilos != 0) {
        OMP_NUM_THREADS = toString vv.hilos;
      }
      // lib.optionalAttrs ov.enable {
        VIBEVOICE_OV_CODIGO = "${pkgs.vibevoiceOvCodigo}";
        VIBEVOICE_IR_LM = "${ov.directorioIR}/tts_lm_estado_${ov.precisionLM}.xml";
        VIBEVOICE_IR_CABEZA = "${ov.directorioIR}/cabeza_${ov.precisionCabeza}.xml";
        VIBEVOICE_IR_ACUSTICO = "${ov.directorioIR}/decoder_estado_int8.xml";
      }
      # El anclaje a nucleos acelera PyTorch un 3% pero RALENTIZA OpenVINO un
      # 118% (medido: 89 ms/llamada sin anclaje, 195 con el). El mismo ajuste,
      # efectos opuestos: se aplica solo cuando el motor es torch.
      // lib.optionalAttrs (vv.anclarNucleos && !ov.enable) {
        OMP_PLACES = "cores";
        OMP_PROC_BIND = "close";
      };

      serviceConfig = {
        ExecStart = "${pkgs.vibevoice-env}/bin/python ${pesos.inferencia}/bin/voz-stream.py";
        EnvironmentFile = lib.mkIf (cfg.ficheroToken != null) cfg.ficheroToken;

        # El arranque carga el modelo y hace una sintesis de calentamiento:
        # son ~2 minutos antes de aceptar la primera peticion.
        TimeoutStartSec = "10min";
        Restart = "on-failure";
        RestartSec = 15;

        DynamicUser = true;
        NoNewPrivileges = true;
        PrivateTmp = true;
        PrivateDevices = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictAddressFamilies = [ "AF_INET" "AF_INET6" "AF_UNIX" ];
        RestrictNamespaces = true;
        LockPersonality = true;
        SystemCallArchitectures = "native";
      };
    };

    networking.firewall.allowedTCPPorts =
      lib.mkIf cfg.abrirCortafuegos [ cfg.puerto ];

    # Por el tunel si esta activo: asi se llega desde fuera sin exponer nada.
    networking.firewall.interfaces.wg0.allowedTCPPorts =
      lib.mkIf config.homelab.tunel.enable [ cfg.puerto ];
  };
}
