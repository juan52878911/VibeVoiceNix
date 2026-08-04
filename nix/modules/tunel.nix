# Tunel WireGuard hacia el edge, para llegar a la VM desde fuera de casa.
#
# La conexion de casa esta tras CGNAT: no hay IP publica entrante, asi que la
# VM abre el tunel hacia el edge y este hace de punto de encuentro. Un movil
# que se conecte al edge alcanza la VM por la IP del tunel.
#
# Lo importante de este diseno: NO expone nada a internet. La API sigue
# escuchando solo en la LAN y en la red del tunel; quien no este en el tunel no
# la ve. La alternativa (publicar voz.asccilabs.com con Caddy) daria acceso mas
# comodo a cambio de dejar el endpoint a la vista de cualquiera.
{ config, lib, pkgs, ... }:

let
  cfg = config.homelab.tunel;
in
{
  options.homelab.tunel = {
    enable = lib.mkEnableOption "tunel WireGuard hacia el edge";

    ip = lib.mkOption {
      type = lib.types.str;
      example = "10.10.10.5/24";
      description = "Direccion de esta maquina DENTRO del tunel.";
    };

    ficheroClave = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/wireguard/privada";
      description = ''
        Clave privada. NO puede vivir en el store: es legible por cualquier
        usuario del sistema. Se genera aparte con `wg genkey` y se deja aqui
        con permisos 600.
      '';
    };

    clavePublicaEdge = lib.mkOption {
      type = lib.types.str;
      description = "Clave publica del servidor WireGuard del edge.";
    };

    endpoint = lib.mkOption {
      type = lib.types.str;
      example = "149.130.186.157:51820";
      description = "IP publica y puerto del edge.";
    };

    redTunel = lib.mkOption {
      type = lib.types.str;
      default = "10.10.10.0/24";
      description = ''
        Que trafico va por el tunel. Solo la red del tunel: asi la VM sigue
        saliendo a internet por su ruta normal y no se convierte en un cliente
        VPN completo, que ademas rompería el acceso desde la LAN.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    networking.wireguard.enable = true;
    networking.wireguard.interfaces.wg0 = {
      ips = [ cfg.ip ];
      privateKeyFile = toString cfg.ficheroClave;

      peers = [{
        publicKey = cfg.clavePublicaEdge;
        allowedIPs = [ cfg.redTunel ];
        endpoint = cfg.endpoint;
        # La VM esta tras CGNAT: sin keepalive el mapeo NAT caduca y el edge
        # deja de poder iniciar conexiones hacia aqui.
        persistentKeepalive = 25;
      }];
    };

    # La API escucha en 0.0.0.0, asi que ya responde por la IP del tunel. Solo
    # hay que dejar pasar el puerto en el cortafuegos para la interfaz wg0.
    networking.firewall.interfaces.wg0.allowedTCPPorts =
      lib.mkIf config.services.voz-api.enable [ config.services.voz-api.puerto ];
  };
}
