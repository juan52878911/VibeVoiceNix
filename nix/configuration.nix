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
    useDHCP = lib.mkDefault true;
    firewall.enable = true;
  };

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
    hilos = 8;
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
