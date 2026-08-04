# Ansible lee estas salidas para saber contra que maquina lanzar nixos-anywhere.

output "vmid" {
  value       = proxmox_virtual_environment_vm.voz.vm_id
  description = "ID de la VM en Proxmox."
}

output "nombre" {
  value       = proxmox_virtual_environment_vm.voz.name
  description = "Nombre de la VM."
}

output "ip" {
  # Con IP fija sale de la variable; con DHCP hay que preguntarle al agente,
  # que solo responde una vez ha arrancado el sistema.
  value = (
    var.ip_cidr != "dhcp"
    ? split("/", var.ip_cidr)[0]
    : try(
      [
        for direccion in flatten(proxmox_virtual_environment_vm.voz.ipv4_addresses) :
        direccion if direccion != "127.0.0.1"
      ][0],
      null
    )
  )
  description = "IP de la VM. Puede ser null en el primer apply si va por DHCP y el agente aun no ha reportado."
}

output "destino_ssh" {
  value       = "root@${var.ip_cidr != "dhcp" ? split("/", var.ip_cidr)[0] : "PENDIENTE"}"
  description = "Destino que consume nixos-anywhere."
}
