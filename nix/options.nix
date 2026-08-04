# Opciones propias del despliegue. Van aparte porque un modulo NixOS que
# declara `options` no puede llevar ademas atributos de config sueltos en la
# raiz: o todo bajo `config`, o las opciones en su propio fichero.
{ lib, ... }:

{
  options.homelab = {
    clavesSSH = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [ "ssh-ed25519 AAAA... juan@mac" ];
      description = ''
        Claves publicas con acceso a la VM. Sin al menos una, la maquina queda
        inaccesible tras instalar: no hay contrasenas ni consola configurada.
      '';
    };

    disco = lib.mkOption {
      type = lib.types.str;
      default = "/dev/vda";
      example = "/dev/sda";
      description = ''
        Disco donde instala disko. En Proxmox con virtio-blk es /dev/vda; con
        virtio-scsi es /dev/sda. Comprobar con `lsblk` antes del primer
        despliegue: disko formatea lo que le digas.
      '';
    };
  };
}
