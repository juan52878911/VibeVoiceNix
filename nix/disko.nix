# Particionado declarativo. nixos-anywhere ejecuta esto contra el disco vacio
# de la VM, asi que el instalador no pregunta nada.
#
# GPT + ESP + raiz ext4. Sin LVM ni cifrado a proposito: la VM se reconstruye
# desde el flake, no se repara, y el estado que importa (el token) vive fuera.
{ lib, config, ... }:

let
  disco = config.homelab.disco;
in
{
  disko.devices.disk.principal = {
    device = disco;
    type = "disk";
    content = {
      type = "gpt";
      partitions = {
        ESP = {
          priority = 1;
          size = "512M";
          type = "EF00";
          content = {
            type = "filesystem";
            format = "vfat";
            mountpoint = "/boot";
            mountOptions = [ "umask=0077" ];
          };
        };

        raiz = {
          size = "100%";
          content = {
            type = "filesystem";
            format = "ext4";
            mountpoint = "/";
          };
        };
      };
    };
  };

  # El host tiene 7,7 GB de RAM y VibeVoice pide ~4 GB de pico. El swap es la
  # red de seguridad para que una generacion no se lleve por delante el sistema.
  swapDevices = [{
    device = "/var/lib/swapfile";
    size = 4 * 1024;
  }];
}
