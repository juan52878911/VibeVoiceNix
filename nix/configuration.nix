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

    # 6 Y NO LA AUTODETECCION. Aqui la autodeteccion se equivoca, y no por un
    # fallo suyo: el hipervisor presenta las 12 vCPU con `core id` distinto y
    # sin `physical id`, asi que dentro de la VM parecen 12 nucleos fisicos y
    # no 6 con sus hermanos SMT. detectar_hilos() cuenta 12, deja uno libre y
    # sale con 11, que es sobresuscribir al doble.
    #
    # Medido con el IR real y la maquina libre (ms por llamada, menor mejor):
    #
    #   hilos      1      2      3      4      5      6      8     11     12
    #   backbone  31,3   19,7   16,4   14,5   14,4   14,0   16,2   20,4   20,2
    #   decoder  157,7   87,8   60,8   50,8   43,2   43,3   51,8   97,0   46,8
    #
    # A partir de 6 no hay nada que ganar y sí que perder: los hilos 7 a 12
    # son hermanos SMT de los seis primeros y se pelean por la misma unidad
    # AVX2. End-to-end, sin solapar: 6 hilos 0,988 · 8 hilos 1,154.
    hilos = 6;
  };

  services.voz-stream = {
    enable = true;
    puerto = 8082;
    # Reusa el token de voz-api: una sola credencial para todo el stack.
    ficheroToken = "/var/lib/voz/token.env";
    # Prefijos de voz propios, fuera del store por el mismo motivo que el
    # token: un .pt ES la voz clonable de una persona y no va al repositorio.
    # Se suben con scp y se suman a las 61 oficiales.
    vocesPropias = "/var/lib/voz/voces-propias";
    # Abierto en la LAN ademas de en el tunel: la pagina de prueba en "/" se
    # usa tambien desde casa, y exigir el tunel estando en la misma red no
    # aporta seguridad -- voz-api (8080) ya esta abierto igual. Sigue pidiendo
    # bearer token, y a internet no se expone nada.
    abrirCortafuegos = true;
    # Semilla FIJA del ruido de la difusion: el servicio es determinista por
    # defecto (mismo texto, misma voz -> mismo audio, byte a byte) y un
    # cliente que quiera variedad manda "semilla": null.
    #
    # 101 y no el sorteo, MEDIDO el 10-09-2026 en esta VM: 18 semillas x las
    # 6 frases del banco de fidelidad, sp-Spk1_man, cfg 3,0, 6 pasos. La 101
    # (y la 17) aciertan las 6 frases con 0 % de WER y con 6, 8 y 10 pasos;
    # la 42 falla 5 de 6 (40,7 %) y la 37 tiene 29,6 %. Sorteando, una de
    # cada seis peticiones caia en una de esas. Entre la mejor y la peor
    # semilla hay 40 puntos de WER; entre 6 y 10 pasos, uno. La tabla entera
    # esta en docs/plan-determinismo-calidad.md. Si cambia la voz por defecto
    # hay que repetir el banco: la semilla buena es de la voz.
    semilla = 101;
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
