# TTS + fachada HTTP: es el camino de produccion, el que consume OpenClaw.
#
# Piper con la voz cargada en memoria da RTF 0,042 en un i7-8700T (24x tiempo
# real), asi que responde una nota de voz de 10 s en menos de medio segundo.
{ config, lib, pkgs, ... }:

let
  cfg = config.services.voz-api;
  whisperCfg = config.services.homelab-whisper;
  streamCfg = config.services.voz-stream;

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

    vibevoiceURL = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "http://127.0.0.1:8082";
      description = ''
        Donde escucha voz-stream, para que `POST /v1/audio/speech` pueda servir
        tambien VibeVoice (`model: "vibevoice"`). Si es null se apunta al de
        esta maquina cuando `services.voz-stream.enable` esta puesto; si no lo
        esta, el modelo simplemente no aparece en `GET /v1/models`.
      '';
    };

    vocesOpenAI = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      example = { alloy = "es_MX-claude-high"; nova = "es_ES-davefx-medium"; };
      description = ''
        Mapa de las voces canonicas de OpenAI (`alloy`, `nova`, ...) a voces
        reales de este servidor. Un cliente que las trae fijas en el codigo no
        puede pedir otra cosa; sin mapa caen todas a `vozDefecto`.
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
        assertion = lib.all (v: lib.elem v cfg.voces) (lib.attrValues cfg.vocesOpenAI);
        message = "services.voz-api.vocesOpenAI apunta a voces que no estan en services.voz-api.voces";
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
      }
      # La fachada de OpenAI delega en voz-stream cuando le piden VibeVoice.
      # Solo se declara la URL si hay a quien llamar; sin ella el codigo prueba
      # igualmente en 127.0.0.1:8082, y como `GET /v1/models` sondea antes de
      # anunciar, un puerto muerto se traduce en "ese modelo no existe" y no en
      # un 503 a mitad de una peticion.
      // lib.optionalAttrs (cfg.vibevoiceURL != null || streamCfg.enable) {
        VOZ_VIBEVOICE_URL =
          if cfg.vibevoiceURL != null
          then cfg.vibevoiceURL
          else "http://127.0.0.1:${toString streamCfg.puerto}";
      }
      // lib.optionalAttrs (cfg.vocesOpenAI != { }) {
        VOZ_OPENAI_VOCES = lib.concatStringsSep ","
          (lib.mapAttrsToList (k: v: "${k}=${v}") cfg.vocesOpenAI);
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
