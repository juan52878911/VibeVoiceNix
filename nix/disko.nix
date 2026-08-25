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

  # RED DE SEGURIDAD, NO SITIO DONDE VIVIR.
  # El pico de 4,6 GB al cargar el modelo, en una maquina de 4,9 GB, empuja
  # paginas al swap; con el swappiness de serie (60) el kernel ademas expulsa
  # anonimo por gusto para engordar la cache. Medido: el proceso se quedaba con
  # 337 MB fuera de RAM EN REGIMEN, y cada peso que caia ahi costaba un fallo
  # de pagina mayor. Devolverlos a RAM (swapoff -a && swapon -a) bajo el RTF de
  # 1,082 a 1,058 con el mismo audio, un 2,2 % que no era del modelo sino del
  # disco.
  #
  # Con 1 el kernel solo expulsa bajo presion de verdad, que es justo para lo
  # que esta el fichero. Si tras un arranque el proceso vuelve a tener VmSwap
  # (se mira en /proc/<pid>/status), un `swapoff -a && swapon -a` lo devuelve.
  boot.kernel.sysctl."vm.swappiness" = 1;
}
