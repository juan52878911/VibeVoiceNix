# TTS + fachada HTTP: es el camino de produccion, el que consume OpenClaw.
#
# Piper con la voz cargada en memoria da RTF 0,042 en un i7-8700T (24x tiempo
# real), asi que responde una nota de voz de 10 s en menos de medio segundo.
{ config, lib, pkgs, ... }:

let
  cfg = config.services.voz-api;
  whisperCfg = config.services.homelab-whisper;

  dirVoces = pkgs.vozPiperVoces.paquete cfg.voces;
in
{
  options.services.voz-api = {
    enable = lib.mkEnableOption "API de voz del homelab (Piper TTS + whisper STT)";

    puerto = lib.mkOption {
      type = lib.types.port;
      default = 8080;
      description = "Puerto HTTP de la API.";
    };

    direccion = lib.mkOption {
      type = lib.types.str;
      default = "0.0.0.0";
      description = "Interfaz de escucha.";
    };

    voces = lib.mkOption {
      type = lib.types.listOf (lib.types.enum pkgs.vozPiperVoces.nombres);
      default = [ "es_MX-claude-high" "es_MX-ald-medium" "es_ES-davefx-medium" ];
      description = ''
        Voces de Piper a instalar. Cada "high" pesa ~60-109 MB, asi que conviene
        no meterlas todas si el disco va justo.
      '';
    };

    vozDefecto = lib.mkOption {
      type = lib.types.str;
      default = "es_MX-claude-high";
      description = "Voz usada cuando la peticion no especifica ninguna.";
    };

    promptSTT = lib.mkOption {
      type = lib.types.str;
      default =
        "Vocabulario tecnico: homelab, Proxmox, WireGuard, Docker, contenedor, "
        + "LXC, Caddy, Oracle, systemd, OpenClaw, Piper, whisper, backup, deploy, "
        + "log, NixOS, Terraform, Ansible, SSH, VPN, Postgres.";
      description = ''
        Prompt que sesga el vocabulario de whisper. Es el ajuste que mas cambia
        la calidad: sin el, "WireGuard" se transcribe "We The War" y "homelab"
        se convierte en "omelab". Merece la pena meter aqui la jerga propia.
      '';
    };

    ficheroToken = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = "/run/secrets/voz-token";
      description = ''
        Fichero con la linea `VOZ_TOKEN=...`. Si es null la API queda abierta,
        lo cual solo tiene sentido en una red de confianza.
        El fichero NO debe estar en el store: se lee en arranque.
      '';
    };

    abrirCortafuegos = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Abre el puerto en la LAN. Ponlo a true solo si la red es de confianza.";
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = lib.elem cfg.vozDefecto cfg.voces;
        message = "services.voz-api.vozDefecto (${cfg.vozDefecto}) tiene que estar en services.voz-api.voces";
      }
      {
        assertion = cfg.abrirCortafuegos -> cfg.ficheroToken != null;
        message = "abrir voz-api en la LAN sin ficheroToken deja el TTS/STT accesible a cualquiera de la red";
      }
    ];

    systemd.services.voz-api = {
      description = "API de voz del homelab (Piper TTS + whisper STT)";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ] ++ lib.optional whisperCfg.enable "homelab-whisper.service";
      wants = [ "network-online.target" ] ++ lib.optional whisperCfg.enable "homelab-whisper.service";

      environment = {
        VOZ_VOICES_DIR = "${dirVoces}";
        VOZ_DEFECTO = cfg.vozDefecto;
        VOZ_WHISPER_URL = "http://127.0.0.1:${toString whisperCfg.puerto}";
        VOZ_PROMPT_STT = cfg.promptSTT;
        VOZ_FFMPEG = "${pkgs.ffmpeg}/bin/ffmpeg";
        VOZ_HOST = cfg.direccion;
        VOZ_PORT = toString cfg.puerto;
      };

      serviceConfig = {
        ExecStart = "${pkgs.voz-api}/bin/voz-api";

        # El token entra por EnvironmentFile para no acabar en el store, que es
        # legible por cualquiera del sistema.
        EnvironmentFile = lib.mkIf (cfg.ficheroToken != null) cfg.ficheroToken;

        DynamicUser = true;
        Restart = "on-failure";
        RestartSec = 5;

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

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.abrirCortafuegos [ cfg.puerto ];
  };
}
