# TTS con respuesta en streaming: se oye el principio mientras se genera el resto.
#
# POST /tts/stream -> audio/wav troceado. Medido en la VM:
#
#   primer sonido    23,21 s  ->  0,20 s     (116x menos espera)
#   tiempo total     23,21 s  ->  22,42 s    (sin sobrecoste)
#
# El audio es bit a bit identico al de la generacion normal: mismo md5. No es
# una aproximacion ni una version degradada, es el mismo resultado entregado
# segun se produce.
#
# POR QUE UN SERVICIO APARTE Y NO DENTRO DE voz-api
# Este carga VibeVoice (~2,3 GB residentes); voz-api solo tiene las voces de
# Piper (~100 MB) y responde en decimas de segundo. Juntarlos haria que una
# sintesis pesada bloqueara las notas de voz rapidas, y la regla en esta VM es
# un modelo por proceso.
{ config, lib, pkgs, ... }:

let
  cfg = config.services.voz-stream;
  vv = config.services.vibevoice;
  ov = config.services.vibevoice.openvino;
  pesos = pkgs.vibevoicePesos;

  # RuntimeDirectory = "voz-stream" lo pone systemd en /run/voz-stream.
  dirCombinado = "/run/voz-stream/voces";

  esQwen = cfg.motor == "qwen3tts";
  q = cfg.qwen3tts;
  modeloQwen = if q.modeloPropio != null then q.modeloPropio else "${pkgs.qwen3TtsPesos}";

  # El entorno del shim de Qwen3 (pkgs/qwen3tts-cli/voz_stream_qwen.py).
  envQwen = {
    QWEN3TTS_BIN = "${pkgs.qwen3TtsC}/bin/qwen_tts";
    QWEN3TTS_MODELO = modeloQwen;
    QWEN3TTS_VOCES = dirCombinado;
    QWEN3TTS_CUANT = q.cuantizacion;
    QWEN3TTS_HILOS = toString q.hilos;
    QWEN3TTS_VOZ_DEFECTO = q.vozDefecto;
    QWEN3TTS_IDIOMA = q.idiomaDefecto;
    QWEN3TTS_TROZO = toString q.trozo;
    QWEN3TTS_RTF = toString q.rtfEsperado;
    VOZ_STREAM_HOST = cfg.direccion;
    VOZ_STREAM_PUERTO = toString cfg.puerto;
    MALLOC_ARENA_MAX = "2";
  };

  # Con Qwen3 no hay voces oficiales: el directorio combinado son solo las
  # propias, en los cuatro formatos que entiende el shim.
  combinarVocesQwen = pkgs.writeShellScript "voz-stream-combinar-voces-qwen" ''
    set -eu
    rm -rf ${dirCombinado}
    mkdir -p ${dirCombinado}
    n=0
    for f in ${cfg.vocesPropias}/*.bin ${cfg.vocesPropias}/*.qvoice \
             ${cfg.vocesPropias}/*.wav ${cfg.vocesPropias}/*.txt; do
      [ -e "$f" ] || continue
      ln -sf "$f" ${dirCombinado}/
      n=$((n + 1))
    done
    echo "[voces] $n ficheros de voz propios (qwen3tts)"
  '';
in
{
  options.services.voz-stream = {
    enable = lib.mkEnableOption ''
      el servicio de TTS en streaming. Necesita ~2,5 GB de RAM en regimen y
      hace un pico de ~4,6 GB al cargar (materializa el fp32 antes de
      cuantizarlo), asi que conviene mirar la memoria libre antes de activarlo
    '';

    motor = lib.mkOption {
      type = lib.types.enum [ "vibevoice" "qwen3tts" ];
      default = "vibevoice";
      description = ''
        Que modelo hay detras de :8082. El contrato HTTP es el mismo
        (POST /tts/stream, GET /voces, GET /health); cambia lo que hay debajo.

          vibevoice   VibeVoice-Realtime-0.5B con OpenVINO (lo medido: RTF
                      0,75 en la VM, sesiones KV, 61 voces oficiales).
          qwen3tts    Qwen3-TTS-0.6B-Base con el motor C: 10 idiomas y clonado
                      cruzado, sin sesiones, sin voces oficiales (solo las de
                      `vocesPropias`), y con la puerta de RTF por medir en
                      esta CPU (ver docs/comparativa-motores.md).
      '';
    };

    qwen3tts = {
      cuantizacion = lib.mkOption {
        type = lib.types.enum [ "int8" "int4" ];
        default = "int8";
        description = "Cuantizacion del talker y el code predictor en el motor C.";
      };
      hilos = lib.mkOption {
        type = lib.types.int;
        default = 0;
        description = "Hilos del motor C. 0 = todos los de la VM.";
      };
      vozDefecto = lib.mkOption {
        type = lib.types.str;
        default = "";
        description = "Voz cuando la peticion no trae `voz`. Tiene que estar en `vocesPropias`.";
      };
      idiomaDefecto = lib.mkOption {
        type = lib.types.str;
        default = "es";
        description = "Idioma cuando la peticion no trae `idioma` (es, en, pt, fr, it, de, ru, ja, ko, zh).";
      };
      trozo = lib.mkOption {
        type = lib.types.int;
        default = 160;
        description = ''
          Caracteres por trozo: Qwen3 acelera el ritmo pasados ~100-150 (issue
          #239, cerrado sin arreglo), asi que el shim corta por frase y
          sintetiza los trozos seguidos con la misma semilla. 0 = sin cortar.
        '';
      };
      rtfEsperado = lib.mkOption {
        type = lib.types.float;
        default = 1.0;
        description = "Lo que se anuncia en X-RTF-Esperado. Ponerlo al valor MEDIDO en esta VM.";
      };
      modeloPropio = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        example = "/var/lib/voz/qwen3/juan";
        description = ''
          Un checkpoint afinado (LoRA fusionada o SFT) en formato Hugging Face,
          FUERA del store por la misma razon que `vocesPropias`: es la voz de
          una persona. null = los pesos oficiales del 0.6B-Base.
        '';
      };
    };

    puerto = lib.mkOption {
      type = lib.types.port;
      default = 8082;
      description = "Puerto HTTP. El 8080 lo usa voz-api y el 8081 whisper.";
    };

    direccion = lib.mkOption {
      type = lib.types.str;
      default = "0.0.0.0";
      description = "Interfaz de escucha.";
    };

    abrirCortafuegos = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Abre el puerto en la LAN. Con el tunel activo no hace falta: la red
        de WireGuard ya llega, y asi el servicio no queda expuesto en la LAN.
      '';
    };

    ficheroToken = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = "/var/lib/voz/token.env";
      description = ''
        Fichero con `VOZ_TOKEN=...`. Lo natural es reutilizar el mismo que
        voz-api para no manejar dos credenciales.
      '';
    };

    vocesPropias = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = "/var/lib/voz/voces-propias";
      description = ''
        Directorio de la MAQUINA con prefijos `.pt` propios, que se suman a las
        61 voces oficiales. Si un `.pt` propio se llama igual que una oficial,
        gana el propio.

        NO ES UNA RUTA DEL STORE, Y ES A PROPOSITO. Un prefijo de voz ES la voz
        clonable de una persona: meterlo en el flake lo publicaria en el
        repositorio, igual que pasaria con el token. Por eso esto sigue el
        mismo patron que `ficheroToken` -- una ruta que se lee en arranque y
        que se sube a la maquina aparte:

            scp mi_voz.pt root@voz:/var/lib/voz/voces-propias/

        Los prefijos se fabrican con `scripts/clonar_voz.py`, que NO cabe en
        esta VM (ver docs/clonado-de-voz.md §7.9): se hacen en una estacion de
        trabajo y aqui llega el `.pt`, que pesa entre 2,6 y 8,4 MB.

        El servicio corre con DynamicUser, asi que el directorio y los ficheros
        tienen que ser legibles por cualquiera. La regla de tmpfiles de abajo
        crea el directorio con 0755 si no existe.
      '';
    };

    solaparDecodificador = lib.mkOption {
      type = lib.types.bool;
      default = !config.services.vibevoice.openvino.enable;
      description = ''
        Saca el decodificador acustico del camino critico y lo corre en su
        propio hilo, a la vez que el resto del bucle.

        POR DEFECTO APAGADO CUANDO EL MOTOR ES OPENVINO. Se midio en el i7-8700T
        de la VM, que es lo que quedaba PENDIENTE aqui abajo, y sale al reves
        que en el Mac: solapar HUNDE este motor.

        La razon es que con OpenVINO cada etapa ya usa la maquina entera. En
        torch el decodificador desperdiciaba nucleos y solapar los recogia;
        aqui solapar solo sobresuscribe. Con 11 hilos repartidos 6 al bucle y
        5 al decodificador sobre 6 nucleos fisicos, las dos etapas se estorban
        muchisimo mas de lo que sus hilos hacen pensar (ms por llamada, banco
        aislado con la maquina libre frente al mismo IR dentro del servicio):

                        aislado, 6 hilos   solapando   sin solapar
          backbone            13,9 ms        46,4 ms      19,5 ms
          decodificador       41,4 ms        94,6 ms      55,6 ms

        Y ademas el solapado NUNCA llega a servir de nada: la contrapresion
        medida (lo que el bucle espera al decodificador) es del 0,02 %, o sea
        que el decodificador jamas era el cuello que se pretendia esconder.

        RTF end-to-end, mismo banco de 12 clips y semilla fija, con los md5
        iguales en las dos (el audio es EL MISMO BIT A BIT, solo cambia quien
        ejecuta que):

          solapado, 11 hilos (6/5)   1,011
          solapado,  6 hilos (3/3)   1,179
          sin solapar, 6 hilos       0,988   <- el que se queda
          sin solapar, 8 hilos       1,154

        En el camino torch se deja encendido, que es donde esta medido que gana.

        LA IDEA ERA APROVECHAR LOS HILOS QUE SOBRAN, dando por hecho que el
        bucle se pasa el rato leyendo pesos y el decodificador es convolucion
        densa en computo, y que por tanto no compiten por lo mismo. Con el
        motor torch se cumple. Con OpenVINO no: los dos son computo, y ahi no
        hay nada que solapar. Ver arriba.

        POR QUE ES SEGURO: el decodificador es un SUMIDERO. En el bucle de
        Microsoft la realimentacion pasa por acoustic_connector(speech_latent);
        el audio decodificado solo se guarda y se emite, no vuelve a entrar en
        el modelo. Se mantiene un unico hilo trabajador y una cola FIFO, asi
        que el orden de las llamadas es el mismo que en la version sincrona --
        que es lo que hace que el audio salga IDENTICO BIT A BIT.

        MEDIDO en un Apple M4 (motor torch-int8, 12 s de audio, semilla 11),
        RTF con y sin solapar, mismo md5 en las 20 pasadas:

          hilos      sincrono   solapado   gana
            2          1,038      0,893    14%
            4          0,700      0,594    15%
            6          0,739      0,587    21%
            8          0,785      0,644    18%
           10          0,775      0,630    19%

        Cuesta ~40 ms mas de espera al primer sonido (0,12 -> 0,16 s), que es
        la profundidad de la tuberia.

        En el i7-8700T de la VM se esperaba una ganancia parecida o mayor,
        porque alli el decodificador pesa mas en el reparto. Medido, sale lo
        contrario; los numeros estan al principio de esta descripcion. Es el
        cuarto caso en este proyecto en que una proyeccion razonable se cae al
        medirla.
      '';
    };

    hilosDecodificador = lib.mkOption {
      type = lib.types.int;
      default = 0;
      description = ''
        Hilos para el decodificador cuando va solapado; el resto son para el
        bucle. 0 = la mitad de `services.vibevoice.hilos`.

        Solo el motor OpenVINO reparte de verdad: INFERENCE_NUM_THREADS es por
        modelo compilado. En el camino torch el pool intra-op es uno para todo
        el proceso y este numero solo sirve de documentacion.
      '';
    };

    semilla = lib.mkOption {
      type = lib.types.nullOr lib.types.int;
      default = null;
      description = ''
        Semilla del ruido de la difusion cuando el cliente no manda ninguna.

        null (el defecto) = sorteo por peticion, que es lo de siempre: el
        mismo texto da un audio distinto cada vez (correlacion 0,019 entre
        pasadas, medido). Con un numero el servicio es DETERMINISTA por
        defecto -- mismas entradas, mismo md5 -- y un cliente que quiera
        variedad manda "semilla": null en la peticion. /health la anuncia
        como semilla_defecto.

        Se deja en null a proposito: fijarla en produccion es una decision
        que tiene que salir del banco (que semilla, para que voz), no de la
        opcion existir.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = esQwen || vv.enable;
        message = ''
          services.voz-stream (motor vibevoice) necesita services.vibevoice.enable
          = true: usa su mismo modelo, sus voces y su configuracion de pasos.
        '';
      }
      {
        assertion = !esQwen || cfg.vocesPropias != null;
        message = ''
          services.voz-stream con motor qwen3tts necesita vocesPropias: el
          0.6B-Base no trae voces, solo clona las que se le den (.bin, .qvoice
          o .wav + .txt fabricados con scripts/clonar_voz_qwen.py).
        '';
      }
    ];

    # El directorio tiene que existir y ser legible por el DynamicUser del
    # servicio. Se crea vacio si no esta; los .pt se suben aparte.
    systemd.tmpfiles.rules = lib.mkIf (cfg.vocesPropias != null) [
      "d ${cfg.vocesPropias} 0755 root root -"
    ];

    systemd.services.voz-stream = {
      description = "TTS en streaming (${cfg.motor})";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];

      environment = if esQwen then envQwen else {
        VIBEVOICE_MODELO = "${pesos.modelo}";
        # Con voces propias se apunta al directorio COMBINADO que monta
        # ExecStartPre; sin ellas, directo al del store y no se monta nada.
        VIBEVOICE_VOCES =
          if cfg.vocesPropias != null then dirCombinado else "${pesos.voces}";
        VIBEVOICE_PASOS = toString vv.pasosDifusion;
        VIBEVOICE_VOZ = vv.vozDefecto;
        VOZ_STREAM_HOST = cfg.direccion;
        VOZ_STREAM_PUERTO = toString cfg.puerto;
        HF_HUB_OFFLINE = "1";
        # glibc crea una arena por hilo y no devuelve lo liberado; con 6 hilos
        # eso fragmenta cientos de MB en un servicio que ya va justo de RAM.
        # Con el decodificador solapado hay un hilo mas que asigna de verdad,
        # asi que este tope importa mas que antes, no menos.
        MALLOC_ARENA_MAX = "2";
        VIBEVOICE_MOTOR = if ov.enable then "openvino" else "torch";
        VIBEVOICE_SOLAPAR_DECODER = if cfg.solaparDecodificador then "1" else "0";
        VIBEVOICE_HILOS_DECODER = toString cfg.hilosDecodificador;
      }
      # hilos = 0 -> no se pone la variable y voz_stream.py cuenta nucleos
      # fisicos. Ponerla vacia NO vale: OpenMP mira si existe, no su valor.
      // lib.optionalAttrs (vv.hilos != 0) {
        OMP_NUM_THREADS = toString vv.hilos;
      }
      # Igual que arriba: vacia significa "sortea", asi que solo se pone si
      # hay un numero.
      // lib.optionalAttrs (cfg.semilla != null) {
        VIBEVOICE_SEMILLA = toString cfg.semilla;
      }
      // lib.optionalAttrs ov.enable {
        VIBEVOICE_OV_CODIGO = "${pkgs.vibevoiceOvCodigo}";
        VIBEVOICE_IR_LM = "${ov.directorioIR}/tts_lm_estado_${ov.precisionLM}.xml";
        VIBEVOICE_IR_CABEZA = "${ov.directorioIR}/cabeza_${ov.precisionCabeza}.xml";
        VIBEVOICE_IR_ACUSTICO =
          "${ov.directorioIR}/decoder_estado_${ov.precisionAcustico}.xml";
      }
      # El anclaje a nucleos acelera PyTorch un 3% pero RALENTIZA OpenVINO un
      # 118% (medido: 89 ms/llamada sin anclaje, 195 con el). El mismo ajuste,
      # efectos opuestos: se aplica solo cuando el motor es torch.
      // lib.optionalAttrs (vv.anclarNucleos && !ov.enable) {
        OMP_PLACES = "cores";
        OMP_PROC_BIND = "close";
      };

      serviceConfig = {
        # Enlaces, no copias: son 96 MB de voces oficiales que ya estan en el
        # store, y el servicio solo las lee.
        ExecStartPre = lib.mkIf (cfg.vocesPropias != null)
          (if esQwen then combinarVocesQwen else pkgs.writeShellScript "voz-stream-combinar-voces" ''
            set -eu
            rm -rf ${dirCombinado}
            mkdir -p ${dirCombinado}
            for f in ${pesos.voces}/*.pt; do
              ln -sf "$f" ${dirCombinado}/
            done
            propias=0
            for f in ${cfg.vocesPropias}/*.pt; do
              # el glob sin coincidencias se queda literal; -e lo descarta
              [ -e "$f" ] || continue
              ln -sf "$f" ${dirCombinado}/
              propias=$((propias + 1))
            done
            echo "[voces] $propias propias sobre $(ls ${pesos.voces}/*.pt | wc -l) oficiales"
          '');

        ExecStart =
          if esQwen
          then "${pkgs.voz-api}/bin/python ${pkgs.qwen3ttsCodigo}/bin/voz-stream-qwen.py"
          else "${pkgs.vibevoice-env}/bin/python ${pesos.inferencia}/bin/voz-stream.py";
        EnvironmentFile = lib.mkIf (cfg.ficheroToken != null) cfg.ficheroToken;

        # El arranque carga el modelo y hace una sintesis de calentamiento:
        # son ~2 minutos antes de aceptar la primera peticion.
        # Donde vive el directorio combinado. Se borra al parar el servicio,
        # que es lo que queremos: se rehace en cada arranque y nunca queda un
        # enlace apuntando a una voz que ya no esta.
        RuntimeDirectory = lib.mkIf (cfg.vocesPropias != null) "voz-stream";

        TimeoutStartSec = "10min";
        Restart = "on-failure";
        RestartSec = 15;

        DynamicUser = true;
        NoNewPrivileges = true;
        PrivateTmp = true;
        PrivateDevices = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictAddressFamilies = [ "AF_INET" "AF_INET6" "AF_UNIX" ];
        RestrictNamespaces = true;
        LockPersonality = true;
        SystemCallArchitectures = "native";
      };
    };

    # Devuelve a RAM lo que el pico de carga empujo al swap.
    #
    # Cargar el modelo pica ~4,6 GB en una maquina de 4,9, asi que al terminar
    # de arrancar el proceso se queda con ~100-340 MB fuera de RAM -- MEDIDO,
    # se ve en /proc/<pid>/status -- y ahi no son datos frios: son pesos que
    # el bucle lee en cada fotograma, y cada uno cuesta un fallo de pagina
    # mayor. `vm.swappiness=1` (ver disko.nix) reduce esto pero no lo evita:
    # el pico es real y el kernel tiene que sacar algo.
    #
    # swapoff -a lo relee entero a RAM y swapon -a lo vuelve a poner vacio.
    # Medido: RTF 1,082 -> 1,058 con el mismo audio bit a bit.
    #
    # SOLO SI CABE. Si el swap usado no entra en la memoria disponible,
    # swapoff mataria algo; en ese caso no se toca nada y se avisa. Y no se
    # marca como fallo: es un saneo oportunista, no un requisito de arranque.
    systemd.services.voz-stream-sin-swap = {
      description = "Devuelve a RAM las paginas que el arranque de voz-stream empujo al swap";
      # El motor C carga por mmap y por peticion: no hay pico de arranque.
      enable = !esQwen;
      after = [ "voz-stream.service" ];
      requires = [ "voz-stream.service" ];
      wantedBy = [ "multi-user.target" ];
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        # El pico esta en la CARGA, y voz-stream es Type=simple: systemd lo da
        # por arrancado en cuanto existe el proceso, dos minutos antes de que
        # el modelo este dentro. Sin esperar a /health esto correria justo
        # antes del pico, que es cuando no sirve de nada.
        TimeoutStartSec = "12min";
        ExecStart = pkgs.writeShellScript "voz-stream-sin-swap" ''
          set -u
          for _ in $(seq 1 120); do
            if ${pkgs.curl}/bin/curl -fsS -m 3 \
                 "http://127.0.0.1:${toString cfg.puerto}/health" >/dev/null; then
              break
            fi
            sleep 5
          done
          usado=$(${pkgs.gawk}/bin/awk '/^SwapTotal:/{t=$2} /^SwapFree:/{f=$2} END{print (t-f)}' /proc/meminfo)
          libre=$(${pkgs.gawk}/bin/awk '/^MemAvailable:/{print $2}' /proc/meminfo)
          if [ "$usado" -eq 0 ]; then
            echo "no hay nada en swap"
            exit 0
          fi
          # margen de 512 MB: MemAvailable es una estimacion, no una promesa
          if [ "$usado" -ge $((libre - 524288)) ]; then
            echo "en swap hay $((usado/1024)) MB y solo quedan $((libre/1024)) MB disponibles: no se toca"
            exit 0
          fi
          echo "devolviendo $((usado/1024)) MB del swap a RAM"
          ${pkgs.util-linux}/bin/swapoff -a && ${pkgs.util-linux}/bin/swapon -a
        '';
      };
    };

    networking.firewall.allowedTCPPorts =
      lib.mkIf cfg.abrirCortafuegos [ cfg.puerto ];

    # Por el tunel si esta activo: asi se llega desde fuera sin exponer nada.
    networking.firewall.interfaces.wg0.allowedTCPPorts =
      lib.mkIf config.homelab.tunel.enable [ cfg.puerto ];
  };
}
