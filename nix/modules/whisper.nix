# STT: whisper.cpp como servidor residente.
#
# Vive en loopback: quien entra desde la red pasa por voz-api, que es quien
# tiene el token. Se mantiene levantado para no recargar el modelo (466 MB en
# `small`) en cada nota de voz.
#
# Medido en un i7-8700T sobre 7,72 s de audio en espanol:
#   base   1,53 s (RTF 0,198)  — rapido, pero falla en terminos tecnicos
#   small  5,18 s (RTF 0,671)  — acierta "Proxmox" y "backup"; el elegido
#
# La iGPU Intel UHD 630 se probo con el backend Vulkan y resulto 2,5x MAS LENTA
# que la CPU (matrix cores: none). Por eso aqui no hay opcion de GPU.
{ config, lib, pkgs, ... }:

let
  cfg = config.services.homelab-whisper;

  modelos = {
    base = {
      url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin";
      hash = "sha256-YO1bw90U7qhWST0zQ0m0BXgt3K8AKNS130CINF+6Lv4=";
    };
    small = {
      url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin";
      hash = "sha256-G+OpsgY4Z7k35k4ux0gzZKeZF+FX+pjF2UtcH//qmHs=";
    };
  };

  ficheroModelo = pkgs.fetchurl modelos.${cfg.modelo};
in
{
  options.services.homelab-whisper = {
    enable = lib.mkEnableOption "servidor de transcripcion whisper.cpp";

    modelo = lib.mkOption {
      type = lib.types.enum (lib.attrNames modelos);
      default = "small";
      description = ''
        Modelo ggml. `small` acierta el vocabulario tecnico; `base` es 3x mas
        rapido pero confunde terminos como "Proxmox".
      '';
    };

    idioma = lib.mkOption {
      type = lib.types.str;
      default = "es";
      description = "Idioma por defecto. `auto` para detectarlo.";
    };

    puerto = lib.mkOption {
      type = lib.types.port;
      default = 8081;
      description = "Puerto en loopback. No se abre en el cortafuegos a proposito.";
    };

    hilos = lib.mkOption {
      type = lib.types.int;
      default = 6;
      description = "Hilos de inferencia. Conviene dejar alguno libre para voz-api.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.homelab-whisper = {
      description = "whisper.cpp — transcripcion de voz (STT)";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];

      serviceConfig = {
        ExecStart = lib.escapeShellArgs [
          "${pkgs.whisper-cpp}/bin/whisper-server"
          "--model" "${ficheroModelo}"
          "--language" cfg.idioma
          "--threads" (toString cfg.hilos)
          "--host" "127.0.0.1"
          "--port" (toString cfg.puerto)
        ];

        DynamicUser = true;
        Restart = "on-failure";
        RestartSec = 5;

        # Solo lee un modelo del store y escucha en loopback: no necesita nada mas.
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
        MemoryDenyWriteExecute = false; # ggml usa JIT en algunos backends
        SystemCallArchitectures = "native";
      };
    };
  };
}
