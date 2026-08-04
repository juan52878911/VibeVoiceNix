# Lo unico especifico de TU maquina. Es el fichero que edita quien clone el
# repo; todo lo demas deberia servir tal cual.
{ ... }:

{
  homelab = {
    # Sin esto no se puede entrar a la VM despues de instalar.
    clavesSSH = [
      "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCjuxF3yBNYoSIhm+DqBq1L0jHDHW8SvoXp49cLWA88r76fin0KGAOq9ztoQQwU/COKEZ1yaDvl8547C2HbY5C+1qy7GtldkrogOHESeOLBcEhqrS/aGEScejzqI7RXqdP0jiw1RPk3ooaHkEcwW/KfH6RGKoT+T51poinlyspQGe4CoswWQEsMqKmGBycGTEUQcIsuq+2cNmy6+x+PEsiWpW2qKuwjtotlBHUNRCJEw5XI2L3V4ARxmE9C1f2N1shkkd4PAZ8YlUmh/eJw2wbdUPSXusxcgx1J6i49ppt3Hyj8K7IxFaktcvjbvA2XT7ZTxFAR4BchfDhwSn/yGilx ssh-key-2026-08-01"
    ];

    # Terraform crea el disco con virtio-scsi -> /dev/sda.
    # Comprobar con `lsblk` antes del primer despliegue: disko formatea sin preguntar.
    disco = "/dev/sda";

    # Tiene que coincidir con ip_cidr/gateway de terraform.tfvars.
    # Sin esto NixOS arranca en DHCP y, si el DHCP no responde, la VM queda
    # inaccesible: no hay contrasena para SSH y solo entran claves.
    ip = "192.168.2.54/24";
    puertaEnlace = "192.168.2.1";
  };

  networking.hostName = "voz";
}
