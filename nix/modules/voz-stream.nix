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
        OMP_NUM_THREADS = toString vv.hilos;
        # glibc crea una arena por hilo y no devuelve lo liberado; con 6 hilos
        # eso fragmenta cientos de MB en un servicio que ya va justo de RAM.
        MALLOC_ARENA_MAX = "2";
      }
      # El anclaje a nucleos acelera PyTorch un 3%, que es el motor de este
      # servicio. OJO si algun dia se cambia a OpenVINO: ahi el MISMO ajuste
      # lo ralentiza un 118% (medido: 89 ms/llamada sin anclaje, 195 con el),
      # porque su planificador de hilos interpreta el binding de otra forma.
      // lib.optionalAttrs vv.anclarNucleos {
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
