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

    # La IP TIENE que declararse aqui, no solo en Terraform.
    #
    # La cloud-init de Terraform solo configura la imagen base; en cuanto
    # nixos-anywhere instala NixOS, cloud-init desaparece y la red pasa a
    # gestionarla el flake. Si esto no esta puesto, la VM arranca bien pero se
    # queda sin IPv4 y sin forma de entrar: no hay contrasena, solo claves SSH.
    ip = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "192.168.2.54/24";
      description = ''
        Direccion con mascara. Si es null se usa DHCP, que para un servidor
        que se administra por SSH es fragil: si el DHCP no responde, te quedas
        fuera de la maquina.
      '';
    };

    puertaEnlace = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "192.168.2.1";
      description = "Puerta de enlace. Obligatoria si `ip` no es null.";
    };

    dns = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ "1.1.1.1" "9.9.9.9" ];
      description = "Servidores DNS cuando la IP es fija.";
    };

    passwordConsola = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "$y$j9T$...";
      description = ''
        Hash de contrasena para root, util SOLO en la consola serie
        (`qm terminal <vmid>` en Proxmox). SSH no la acepta porque
        PasswordAuthentication esta desactivado.

        Es la red de seguridad para no quedarte fuera si la red falla. Generar
        con `mkpasswd -m yescrypt`. Ojo: acaba en el store, que es legible por
        cualquier usuario del sistema, asi que usa una distinta a las tuyas.
      '';
    };
  };
}
