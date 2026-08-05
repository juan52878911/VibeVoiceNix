# Imágenes Docker para quien no tiene Nix.
#
# Se construyen CON Nix pero se usan SIN él: quien las reciba solo necesita
# Docker. No hay Dockerfile ni una segunda definición del stack que mantener
# sincronizada — las imágenes salen de los mismos paquetes que la VM NixOS,
# así que no pueden divergir.
#
#     nix build .#imagenes.voz-api      # ~350 MB: Piper + whisper
#     nix build .#imagenes.voz-stream   # ~4 GB: + VibeVoice
#     docker load < result
#
# Por qué buildLayeredImage y no buildImage: reparte el cierre en capas por
# tamaño, así el modelo de 1,9 GB queda en su propia capa. Cambiar una línea
# de Python reenvía megas, no gigas.
#
# TAMAÑO: la imagen de VibeVoice ronda los 4 GB y no hay forma de evitarlo —
# son 1,9 GB de pesos más ~2 GB de torch. La de Piper + whisper sí es ligera.
{ pkgs, lib }:

let
  # Un usuario sin privilegios dentro de la imagen. Sin esto los procesos
  # corren como root, que en un contenedor no aporta nada y molesta si se
  # montan volúmenes del anfitrión.
  usuarioBase = pkgs.runCommand "usuario-base" { } ''
    mkdir -p "$out/etc"
    echo "voz:x:1000:1000:voz:/tmp:/bin/sh" > "$out/etc/passwd"
    echo "voz:x:1000:" > "$out/etc/group"
    echo "root:x:0:0:root:/root:/bin/sh" >> "$out/etc/passwd"
  '';

  # Un /tmp de verdad, escribible. No puede salir de `contents`: todo lo que
  # viene del store de Nix es de solo lectura, asi que un /tmp de ahi no vale
  # para nada. extraCommands escribe en la capa de la imagen, que si lo es.
  #
  # Sin esto el contenedor NO ARRANCA. Python resuelve su directorio temporal
  # al importar, y torch lo pide nada mas cargar:
  #
  #   FileNotFoundError: No usable temporary directory found in
  #   ['/tmp', '/var/tmp', '/usr/tmp']
  #
  # En la VM NixOS no se nota porque systemd le da su propio /tmp privado a
  # cada servicio; el fallo solo sale por Docker.
  crearTmp = ''
    mkdir -p tmp
    chmod 1777 tmp
  '';

  pesos = pkgs.vibevoicePesos;
in
{
  # ------------------------------------------------------------------
  # Piper (TTS) + whisper.cpp (STT). Es el camino de produccion y el que
  # tiene sentido distribuir: ligero y rapido en cualquier maquina.
  # ------------------------------------------------------------------
  voz-api = pkgs.dockerTools.buildLayeredImage {
    name = "voz-api";
    tag = "latest";
    # /stt normaliza el audio en un TemporaryDirectory: sin /tmp da un 500 en
    # cada transcripcion, aunque el servicio arranque tan campante.
    extraCommands = crearTmp;

    contents = [
      pkgs.voz-api
      pkgs.ffmpeg-headless  # headless: el ffmpeg completo arrastra GTK y X11
      pkgs.whisper-cpp
      pkgs.bashInteractive
      pkgs.coreutils
      pkgs.cacert
      usuarioBase
      (pkgs.vozPiperVoces.paquete [
        "es_MX-claude-high"
        "es_MX-ald-medium"
        "es_ES-davefx-medium"
      ])
    ];

    config = {
      Cmd = [ "${pkgs.voz-api}/bin/voz-api" ];
      WorkingDir = "/tmp";
      User = "1000:1000";
      ExposedPorts."8080/tcp" = { };
      Env = [
        "VOZ_VOICES_DIR=${pkgs.vozPiperVoces.paquete [
          "es_MX-claude-high" "es_MX-ald-medium" "es_ES-davefx-medium"
        ]}"
        "VOZ_DEFECTO=es_MX-claude-high"
        "VOZ_FFMPEG=${pkgs.ffmpeg-headless}/bin/ffmpeg"
        # Sin whisper propio dentro: se apunta al servicio de al lado. Meterlo
        # aqui obligaria a arrancar dos procesos en un contenedor.
        "VOZ_WHISPER_URL=http://whisper:8081"
        "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      ];
    };
  };

  # ------------------------------------------------------------------
  # whisper.cpp como servicio aparte, para que cada contenedor haga una cosa.
  # ------------------------------------------------------------------
  whisper = pkgs.dockerTools.buildLayeredImage {
    name = "voz-whisper";
    tag = "latest";
    contents = [ pkgs.whisper-cpp pkgs.bashInteractive pkgs.coreutils usuarioBase ];
    config = {
      # El modelo NO va dentro: se monta como volumen. Meterlo anadiria 466 MB
      # a la imagen y ata la version del modelo a la del contenedor.
      Cmd = [
        "${pkgs.whisper-cpp}/bin/whisper-server"
        "--model" "/modelos/ggml-small.bin"
        "--language" "es"
        "--host" "0.0.0.0"
        "--port" "8081"
      ];
      User = "1000:1000";
      ExposedPorts."8081/tcp" = { };
      Volumes."/modelos" = { };
    };
  };

  # ------------------------------------------------------------------
  # VibeVoice con streaming. Pesada (~4 GB) pero autocontenida: el modelo, el
  # tokenizador y las voces viajan dentro, asi que arranca sin red.
  # ------------------------------------------------------------------
  voz-stream = pkgs.dockerTools.buildLayeredImage {
    name = "voz-stream";
    tag = "latest";
    # Mas capas de lo normal a proposito: con el modelo de 1,9 GB y torch de
    # 2 GB, el reparto por tamano evita reenviarlo todo en cada cambio.
    maxLayers = 120;
    extraCommands = crearTmp;

    contents = [
      pkgs.vibevoice-env
      pkgs.ffmpeg-headless
      pkgs.bashInteractive
      pkgs.coreutils
      pkgs.cacert
      usuarioBase
      pesos.modelo
      pesos.voces
      pesos.inferencia
    ];

    config = {
      Cmd = [
        "${pkgs.vibevoice-env}/bin/python"
        "${pesos.inferencia}/bin/voz-stream.py"
      ];
      WorkingDir = "/tmp";
      User = "1000:1000";
      ExposedPorts."8082/tcp" = { };
      Env = [
        "VIBEVOICE_MODELO=${pesos.modelo}"
        "VIBEVOICE_VOCES=${pesos.voces}"
        "VIBEVOICE_PASOS=6"
        "VIBEVOICE_VOZ=sp-Spk1_man"
        "VOZ_STREAM_PUERTO=8082"
        # Todo dentro de la imagen: sin esto intentaria bajar el tokenizador.
        "HF_HUB_OFFLINE=1"
        # 6 hilos y anclaje: medido como el optimo para el motor PyTorch. Si
        # se cambia a OpenVINO hay que QUITAR el anclaje, que ahi cuesta un
        # 118% (89 ms/llamada sin el, 195 con el).
        "OMP_NUM_THREADS=6"
        "OMP_PLACES=cores"
        "OMP_PROC_BIND=close"
        # glibc abre una arena por hilo y no devuelve lo liberado.
        "MALLOC_ARENA_MAX=2"
        "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
      ];
    };
  };
}
