# VM de voz en Proxmox.
#
# Terraform solo la crea y la arranca con un sistema minimo que tenga SSH.
# NixOS lo instala despues nixos-anywhere, que hace kexec sobre la maquina
# arrancada y la reinstala desde el flake. Ese reparto es a proposito:
# Terraform gestiona el ciclo de vida de la VM, el flake gestiona el sistema.

terraform {
  required_version = ">= 1.6"

  required_providers {
    proxmox = {
      source  = "bpg/proxmox"
      version = "~> 0.111"
    }
  }
}

provider "proxmox" {
  endpoint  = var.proxmox_endpoint
  api_token = var.proxmox_api_token
  insecure  = var.proxmox_insecure

  # nixos-anywhere necesita entrar por SSH al host para algunas operaciones
  # de disco; con el token de API solo no basta.
  ssh {
    agent    = true
    username = var.proxmox_ssh_user
  }
}

# Imagen base: cualquier cloud image con SSH sirve, porque nixos-anywhere la
# sustituye entera. Debian se usa por ser pequena y traer cloud-init.
resource "proxmox_download_file" "base" {
  content_type = "iso"
  datastore_id = var.datastore_iso
  node_name    = var.node

  url       = var.imagen_url
  file_name = var.imagen_nombre
}

# La clave publica que se inyecta por cloud-init es la que usara nixos-anywhere
# para entrar la primera vez.
resource "proxmox_virtual_environment_vm" "voz" {
  name        = var.nombre
  description = "Stack de voz del homelab (Piper + whisper.cpp + VibeVoice). Gestionada por NixOS."
  tags        = ["nixos", "voz", "terraform"]

  node_name = var.node
  vm_id     = var.vmid

  # Arranque automatico: es un servicio, no un experimento.
  on_boot = true

  agent {
    # Desactivado a proposito. Con enabled=true el provider ESPERA a que el
    # qemu-guest-agent reporte, y la cloud image de Debian no lo trae: el apply
    # se queda colgado tras crear la VM. Como la IP es fija no hace falta para
    # nada, y de todas formas nixos-anywhere reemplaza el sistema entero.
    enabled = false
  }

  # UEFI, no el SeaBIOS por defecto: nix/disko.nix crea GPT + ESP y el sistema
  # arranca con systemd-boot, que solo funciona con UEFI. Con SeaBIOS la VM
  # instala bien y luego se queda colgada sin encontrar arranque.
  bios = "ovmf"

  efi_disk {
    datastore_id = var.datastore_vm
    type         = "4m"
  }

  cpu {
    cores = var.cores
    # `host` expone AVX2 al huesped. Sin esto la inferencia pierde bastante:
    # tanto ggml como onnxruntime dependen de las extensiones vectoriales.
    type = "host"
  }

  memory {
    dedicated = var.memoria_mb
    # Sin balloon: VibeVoice hace picos de ~4 GB y el balloon los penaliza.
    floating = 0
  }

  disk {
    datastore_id = var.datastore_vm
    file_id      = proxmox_download_file.base.id
    interface    = "scsi0"
    size         = var.disco_gb
    discard      = "on"
    ssd          = true
  }

  network_device {
    bridge = var.bridge
  }

  initialization {
    datastore_id = var.datastore_vm

    ip_config {
      ipv4 {
        address = var.ip_cidr
        gateway = var.gateway
      }
    }

    user_account {
      username = "root"
      keys     = var.claves_ssh
    }
  }

  operating_system {
    type = "l26"
  }

  serial_device {}

  lifecycle {
    ignore_changes = [
      # A partir del primer nixos-anywhere el disco lo gestiona disko:
      # Terraform no debe intentar devolverlo a la imagen base.
      disk[0].file_id,
      initialization,
    ]
  }
}
