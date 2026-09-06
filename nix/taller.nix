# La VM `taller`: la misma base que `voz`, sin servir nada. Corre en modo
# taller (scripts/modo_taller.sh on), con voz y app-noticias apagadas, y se
# lleva ~11 GB y los 6 nucleos del M920q para el banco de comparacion, el
# doblaje por lotes con Qwen3 y los experimentos de LoRA.
#
# Es un clon de la VM voz en Proxmox al que se le aplica esta configuracion
# con nixos-rebuild: misma imagen, otro nombre, otra IP, servicios apagados y
# el entorno de Qwen3 instalado.
{ config, lib, pkgs, ... }:

{
  networking.hostName = lib.mkForce "taller";
  homelab.ip = lib.mkForce "192.168.2.55/24";
  homelab.tunel.enable = lib.mkForce false;

  # Todo lo que sirve voz queda apagado: la RAM es para trabajar, no para
  # tener modelos residentes esperando peticiones.
  services.voz-stream.enable = lib.mkForce false;
  services.vibevoice.enable = lib.mkForce false;
  services.vibevoice.openvino.enable = lib.mkForce false;
  services.voz-api.enable = lib.mkForce false;
  services.homelab-whisper.enable = lib.mkForce false;

  environment.systemPackages = with pkgs; [
    qwen3-tts-env       # PyTorch de referencia: banco, lote, LoRA
    qwen3TtsC           # motor C: puerta de RTF y servidor
    vibevoice-env       # el motor actual, para el mismo banco
    ffmpeg
    tmux
    htop
  ];

  # Rutas fijas para los scripts: los pesos viven en el store.
  environment.variables = {
    QWEN3TTS_MODELO = "${pkgs.qwen3TtsPesos}";
    QWEN3TTS_BIN = "${pkgs.qwen3TtsC}/bin/qwen_tts";
    VIBEVOICE_MODELO = "${pkgs.vibevoicePesos.modelo}";
    VIBEVOICE_VOCES = "${pkgs.vibevoicePesos.voces}";
  };

  # Los trabajos de noche van con nice 19 dentro de la VM y la VM entera con
  # cpuunits bajos desde Proxmox, para no pisar a AuraCRM.
  systemd.tmpfiles.rules = [
    "d /var/lib/taller 0755 root root -"
    "d /var/lib/taller/cola 0755 root root -"
  ];

  # Sin swap-off ni anclaje de nucleos: aqui no hay tiempo real que proteger.
  zramSwap.enable = true;
}
