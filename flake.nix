{
  description = "Stack de voz del homelab en NixOS: Piper (TTS) + whisper.cpp (STT) + VibeVoice (laboratorio)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    disko = {
      url = "github:nix-community/disko";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    nixos-anywhere = {
      url = "github:nix-community/nixos-anywhere";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.disko.follows = "disko";
    };

    # uv2nix traduce los uv.lock a derivaciones Nix. Es lo que permite que
    # VibeVoice -que no esta en nixpkgs- sea reproducible de verdad.
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { self
    , nixpkgs
    , disko
    , nixos-anywhere
    , uv2nix
    , pyproject-nix
    , pyproject-build-systems
    , ...
    }:
    let
      # El destino es siempre x86_64-linux (la VM en Proxmox). Los sistemas de
      # desarrollo son varios porque el repo se maneja desde un Mac ARM.
      sistemaDestino = "x86_64-linux";
      sistemasDev = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];

      forEachDev = f: nixpkgs.lib.genAttrs sistemasDev (system: f system nixpkgs.legacyPackages.${system});

      # Overlay con los paquetes propios, para que esten disponibles como
      # pkgs.voz-api / pkgs.vibevoice-env dentro de los modulos NixOS.
      overlayPropio = import ./nix/overlay.nix {
        inherit uv2nix pyproject-nix pyproject-build-systems;
      };
    in
    {
      overlays.default = overlayPropio;

      # ------------------------------------------------------------------
      # El sistema completo. `nixos-anywhere --flake .#voz` instala esto.
      # ------------------------------------------------------------------
      nixosConfigurations.voz = nixpkgs.lib.nixosSystem {
        system = sistemaDestino;
        modules = [
          disko.nixosModules.disko
          { nixpkgs.overlays = [ overlayPropio ]; }
          ./nix/options.nix
          ./nix/disko.nix
          ./nix/configuration.nix
          ./nix/host.nix
          ./nix/modules
        ];
      };

      packages = forEachDev (system: pkgs:
        let pkgsCon = pkgs.extend overlayPropio;
        in {
          voz-api = pkgsCon.voz-api;
        }
        # El entorno de VibeVoice solo se construye para el destino: las ruedas
        # de torch+cpu que fija el lock son linux-x86_64.
        // nixpkgs.lib.optionalAttrs (system == sistemaDestino) {
          vibevoice-env = pkgsCon.vibevoice-env;
          default = pkgsCon.voz-api;
        });

      # ------------------------------------------------------------------
      # Imagenes Docker, para usar el stack SIN tener Nix. Se construyen con
      # Nix a partir de los mismos paquetes que la VM, asi que no hay una
      # segunda definicion que pueda divergir.
      #
      #   nix build .#imagenes.voz-api && docker load < result
      #
      # Solo para linux-x86_64: las imagenes contienen binarios de esa
      # arquitectura. Desde un Mac ARM se construyen con un constructor remoto
      # o con `--system x86_64-linux` si hay emulacion configurada.
      # ------------------------------------------------------------------
      imagenes =
        let pkgs = (nixpkgs.legacyPackages.${sistemaDestino}).extend overlayPropio;
        in import ./nix/imagenes.nix { inherit pkgs; inherit (nixpkgs) lib; };

      # ------------------------------------------------------------------
      # Entorno de trabajo: trae terraform, ansible y nixos-anywhere para no
      # tener que instalar nada en la maquina del que clone el repo.
      # ------------------------------------------------------------------
      devShells = forEachDev (system: pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            opentofu
            ansible
            uv
            jq
            curl
          ]
          # nixos-anywhere no se construye para todos los sistemas; se anade
          # solo donde exista para no romper el shell en el resto.
          ++ pkgs.lib.optional (nixos-anywhere.packages ? ${system})
            nixos-anywhere.packages.${system}.default
          ++ pkgs.lib.optional pkgs.stdenv.isLinux pkgs.nixos-rebuild;

          shellHook = ''
            echo "VibeVoiceNix — entorno de despliegue"
            echo "  tofu ...............  provisiona la VM en Proxmox"
            echo "  ansible-playbook ...  orquesta provision -> install -> update"
            echo "  nixos-anywhere .....  instala NixOS sobre la VM recien creada"
            echo
            echo "Flujo completo:  ansible-playbook ansible/playbooks/provision.yml"
          '';
        };
      });

      formatter = forEachDev (_: pkgs: pkgs.nixpkgs-fmt);
    };
}
