# Sistema base de la VM de voz.
#
# Todo lo que define la maquina esta aqui o en los modulos: no hay pasos
# manuales despues de instalar. Si algo no esta en el flake, no existe.
{ config, lib, pkgs, modulesPath, ... }:

{
  imports = [
    (modulesPath + "/profiles/qemu-guest.nix")
  ];

  # ------------------------------------------------------------------
  # Arranque y disco
  # ------------------------------------------------------------------
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = false;

  # Consola serie: `qm terminal 210` desde el host Proxmox. Es la unica via de
  # entrada si la red falla, asi que conviene tenerla desde el primer arranque.
  boot.kernelParams = [ "console=tty1" "console=ttyS0,115200" ];
  boot.initrd.availableKernelModules = [
    "ahci"
    "xhci_pci"
    "virtio_pci"
    "virtio_scsi"
    "sd_mod"
    "sr_mod"
  ];

  # ------------------------------------------------------------------
  # Identidad y red
  # ------------------------------------------------------------------
  networking = {
    hostName = lib.mkDefault "voz";
    # DHCP solo si no se declaro IP fija en host.nix.
    useDHCP = lib.mkDefault (config.homelab.ip == null);
    firewall.enable = true;
  };

  # Red estatica con systemd-networkd.
  #
  # Se empareja por TIPO, no por nombre. Un glob como "en*" parece razonable
  # hasta que la interfaz se llama eth0 y la maquina arranca sin IP: como aqui
  # solo se entra por clave SSH, eso deja la VM inaccesible. `Type = "ether"`
  # casa con cualquier interfaz ethernet y no depende del esquema de nombres.
  systemd.network = lib.mkIf (config.homelab.ip != null) {
    enable = true;
    networks."10-lan" = {
      matchConfig.Type = "ether";
      address = [ config.homelab.ip ];
      gateway = lib.optional (config.homelab.puertaEnlace != null) config.homelab.puertaEnlace;
      dns = config.homelab.dns;
      networkConfig.IPv6AcceptRA = true;
    };
  };
  networking.useNetworkd = lib.mkIf (config.homelab.ip != null) true;

  time.timeZone = lib.mkDefault "America/Bogota";
  i18n.defaultLocale = "es_ES.UTF-8";
  console.keyMap = "es";

  # ------------------------------------------------------------------
  # Acceso: solo por clave. Sin contrasenas, sin root por SSH.
  # ------------------------------------------------------------------
  services.openssh = {
    enable = true;
    settings = {
      PasswordAuthentication = false;
      KbdInteractiveAuthentication = false;
      PermitRootLogin = lib.mkDefault "prohibit-password";
    };
  };

  users.users.root.openssh.authorizedKeys.keys = config.homelab.clavesSSH;

  # Contrasena de emergencia SOLO para la consola serie. SSH la ignora
  # (PasswordAuthentication = false), asi que no abre la maquina a la red;
  # sirve para no quedarte fuera si la red se cae, que es exactamente lo que
  # paso en el primer despliegue de esta VM.
  users.users.root.hashedPassword = lib.mkIf (config.homelab.passwordConsola != null)
    config.homelab.passwordConsola;
  users.users.juan = {
    isNormalUser = true;
    extraGroups = [ "wheel" ];
    openssh.authorizedKeys.keys = config.homelab.clavesSSH;
  };
  security.sudo.wheelNeedsPassword = false;

  assertions = [{
    assertion = config.homelab.clavesSSH != [ ];
    message = ''
      homelab.clavesSSH esta vacio: la VM se instalaria sin ninguna forma de
      entrar. Pon tu clave publica en nix/host.nix.
    '';
  }];

  # ------------------------------------------------------------------
  # Los tres motores de voz
  # ------------------------------------------------------------------
  services.homelab-whisper = {
    enable = true;
    modelo = "small";
    idioma = "es";
    hilos = 6;
  };

  services.voz-api = {
    enable = true;
    puerto = 8080;
    voces = [ "es_MX-claude-high" "es_MX-ald-medium" "es_ES-davefx-medium" ];
    vozDefecto = "es_MX-claude-high";
    # Se abre a la LAN solo porque hay token; el assert del modulo lo exige.
    abrirCortafuegos = true;
    ficheroToken = "/var/lib/voz/token.env";
  };

  services.vibevoice = {
    enable = true;
    # Motor OpenVINO: RTF 1,09 frente a 2,19 de PyTorch. La primera activacion
    # genera los grafos (~15 min, pico de 4,6 GB de RAM); despues arranca solo.
    openvino.enable = true;
  };

  services.voz-stream = {
    enable = true;
    puerto = 8082;
    # Reusa el token de voz-api: una sola credencial para todo el stack.
    ficheroToken = "/var/lib/voz/token.env";
    # Sin abrir en la LAN: se llega por el tunel de WireGuard, que ya alcanza
    # la VM desde fuera de casa sin exponer nada a internet ni a la red local.
    abrirCortafuegos = false;
  };

  # El token no puede vivir en el store (es legible por todo el sistema). Se
  # genera en el primer arranque si no existe y se queda fuera de Nix.
  systemd.services.voz-token = {
    description = "Genera el token de la API de voz si no existe";
    wantedBy = [ "multi-user.target" ];
    before = [ "voz-api.service" ];
    serviceConfig = {
      Type = "oneshot";
      RemainAfterExit = true;
    };
    script = ''
      destino=/var/lib/voz/token.env
      if [ ! -s "$destino" ]; then
        mkdir -p /var/lib/voz
        printf 'VOZ_TOKEN=%s\n' "$(${pkgs.openssl}/bin/openssl rand -hex 24)" > "$destino"
        chmod 600 "$destino"
        echo "token de voz-api generado en $destino"
      fi
    '';
  };

  # ------------------------------------------------------------------
  # Utilidades minimas. La VM es de un solo proposito.
  # ------------------------------------------------------------------
  environment.systemPackages = with pkgs; [
    curl
    jq
    ffmpeg
    sox
    htop
    # git es OBLIGATORIO, no una comodidad: uv2nix resuelve VibeVoice desde un
    # repositorio git, asi que sin el la VM no puede construir su propio
    # sistema. Con git aqui, esta maquina se reconstruye sola y no hace falta
    # un host de construccion aparte -- que es de donde vino el fallo mas caro
    # de este proyecto: desplegar durante horas desde una copia del repo
    # atrasada tres commits.
    git
  ];

  nix.settings = {
    experimental-features = [ "nix-command" "flakes" ];
    auto-optimise-store = true;
  };
  nix.gc = {
    automatic = true;
    dates = "weekly";
    options = "--delete-older-than 30d";
  };

  system.stateVersion = "25.05";
}
