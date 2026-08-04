variable "proxmox_endpoint" {
  type        = string
  description = "URL de la API de Proxmox, p.ej. https://192.168.2.100:8006/"
}

variable "proxmox_api_token" {
  type        = string
  sensitive   = true
  description = <<-EOT
    Token de API con formato usuario@realm!nombre=uuid.
    Crearlo con:
      pveum user token add root@pam terraform --privsep 0
  EOT
}

variable "proxmox_insecure" {
  type        = bool
  default     = true
  description = "true si Proxmox usa el certificado autofirmado de fabrica."
}

variable "proxmox_ssh_user" {
  type        = string
  default     = "root"
  description = "Usuario SSH del host Proxmox. El provider lo necesita para operaciones de disco."
}

variable "node" {
  type        = string
  default     = "pve"
  description = "Nombre del nodo Proxmox."
}

variable "vmid" {
  type        = number
  default     = 210
  description = "ID de la VM. Debe estar libre: en este homelab 100, 200 y 201 ya se usan."
}

variable "nombre" {
  type        = string
  default     = "voz"
  description = "Nombre de la VM."
}

# ---------------------------------------------------------------------------
# Recursos
#
# Los valores por defecto salen de medir el stack en un i7-8700T:
#   - VibeVoice hace pico de 3,9 GB -> 6 GB deja margen para el resto
#   - whisper `small` va con 6 hilos; Piper apenas consume
# ---------------------------------------------------------------------------
variable "cores" {
  type        = number
  default     = 8
  description = "Nucleos. El host tiene 12 hilos; 8 deja aire para las otras VMs."
}

variable "memoria_mb" {
  type        = number
  default     = 6144
  description = <<-EOT
    RAM en MB. Minimo realista 5120: el modelo de VibeVoice son 1,9 GB en fp32
    y el proceso llega a 3,9 GB de pico. Con menos, el sistema se va a swap.
  EOT

  validation {
    condition     = var.memoria_mb >= 4096
    error_message = "Por debajo de 4096 MB VibeVoice no llega a cargar el modelo."
  }
}

variable "disco_gb" {
  type        = number
  default     = 40
  description = <<-EOT
    Disco en GB. El cierre de Nix con torch CPU (~2 GB), el modelo (1,9 GB),
    las voces de Piper y el modelo de whisper piden holgura.
  EOT

  validation {
    condition     = var.disco_gb >= 25
    error_message = "Con menos de 25 GB no cabe el cierre de Nix mas los modelos."
  }
}

# ---------------------------------------------------------------------------
# Red y almacenamiento
# ---------------------------------------------------------------------------
variable "bridge" {
  type        = string
  default     = "vmbr0"
  description = "Puente de red de Proxmox."
}

variable "ip_cidr" {
  type        = string
  default     = "dhcp"
  description = "IP con mascara (192.168.2.54/24) o \"dhcp\"."
}

variable "gateway" {
  type        = string
  default     = null
  description = "Puerta de enlace. Se ignora si ip_cidr es dhcp."
}

variable "datastore_vm" {
  type        = string
  default     = "local-lvm"
  description = "Almacenamiento de los discos de VM."
}

variable "datastore_iso" {
  type        = string
  default     = "local"
  description = "Almacenamiento de imagenes ISO."
}

variable "imagen_url" {
  type        = string
  default     = "https://cloud.debian.org/images/cloud/bookworm/latest/debian-12-genericcloud-amd64.qcow2"
  description = <<-EOT
    Imagen de arranque. Solo tiene que traer SSH y cloud-init: nixos-anywhere
    la reemplaza entera por NixOS en el primer despliegue.
  EOT
}

variable "imagen_nombre" {
  type        = string
  default     = "debian-12-genericcloud-amd64.img"
  description = "Nombre con el que se guarda la imagen en Proxmox."
}

variable "claves_ssh" {
  type        = list(string)
  description = <<-EOT
    Claves publicas para el primer arranque. Deben coincidir con las de
    nix/host.nix, o perderas el acceso tras instalar NixOS.
  EOT
}
