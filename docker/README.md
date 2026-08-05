# Usar el stack de voz con Docker

**No hace falta Nix.** Tres comandos:

```bash
./bajar-modelo.sh                              # modelo de whisper, 466 MB
echo "VOZ_TOKEN=$(openssl rand -hex 24)" > .env
docker compose up -d --build
```

La primera vez tarda unos minutos: compila whisper.cpp y descarga las voces.
Después arranca en segundos.

Se construye para **la arquitectura de tu máquina**, así que en un Mac ARM
salen imágenes ARM y no hace falta constructor remoto ni nada parecido.

## Dos formas de tener las imágenes

Este directorio ofrece **Dockerfiles normales** (lo de arriba) y el flake
ofrece **imágenes generadas por Nix**. No es duplicación por accidente:

| | Dockerfile | Imagen de Nix |
|---|---|---|
| Requisitos | Solo Docker | Nix + máquina x86_64 |
| Arquitectura | La de tu máquina | Solo `x86_64-linux` |
| Reproducible | Las versiones sí (mismo `uv.lock`) | Bit a bit |
| Para quién | **Cualquiera** | Producción |

Ambas fijan las dependencias con **el mismo `uv.lock`**, así que las versiones
de Python no pueden divergir. Lo que cambia es la base del sistema.

Si quieres las de Nix: `nix build .#imagenes.voz-api -o voz-api.tar.gz`, y
luego `docker load < voz-api.tar.gz`.

Ya responde en `http://localhost:8080`:

```bash
TOKEN=$(grep VOZ_TOKEN .env | cut -d= -f2)

# Generar una nota de voz
curl -X POST http://localhost:8080/tts \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"texto":"El backup terminó sin errores."}' -o nota.ogg

# Entender una nota de voz
curl -X POST http://localhost:8080/stt \
  -H "Authorization: Bearer $TOKEN" -F archivo=@nota.ogg
```

### Añadir VibeVoice (opcional, pesado)

```bash
docker compose --profile pesado up -d --build
```

Página de prueba en `http://localhost:8082/`.

**Necesita ~6 GB de RAM libres.** Pide 2,5 GB en régimen pero hace un pico de
4,6 GB al arrancar, porque carga el modelo en fp32 antes de cuantizarlo. Y
tarda **unos 2 minutos** en aceptar la primera petición: carga el modelo y
hace una síntesis de calentamiento para no pagarla en tu primera llamada.

## Si vas a construir las imágenes

Necesitas Nix con flakes **y una máquina `x86_64-linux`**. Las imágenes
contienen binarios de esa arquitectura, así que desde un Mac ARM fallan con:

```
error: Cannot build '...-usuario-base.drv'
       Required system: 'x86_64-linux'
       Current system: 'aarch64-darwin'
```

No es un fallo del flake: es que el trabajo hay que hacerlo en x86_64.

### Desde una máquina x86_64-linux

```bash
nix build .#imagenes.voz-api     -o voz-api.tar.gz
nix build .#imagenes.whisper     -o voz-whisper.tar.gz
nix build .#imagenes.voz-stream  -o voz-stream.tar.gz
```

### Desde un Mac, usando otra máquina como constructor

Nix puede delegar la compilación por SSH. Lo hace el **daemon**, que corre
como `root`, así que la clave tiene que estar en el `root` del Mac — de ahí
los `sudo`. Se configura una vez:

```bash
# 1. Deja que root del Mac entre en el constructor
sudo mkdir -p /var/root/.ssh
sudo cp ~/.ssh/TU_CLAVE /var/root/.ssh/id_constructor
sudo chmod 600 /var/root/.ssh/id_constructor

# 2. Declara el constructor
sudo tee /etc/nix/machines >/dev/null <<'FIN'
ssh-ng://root@IP_DEL_CONSTRUCTOR x86_64-linux /var/root/.ssh/id_constructor 4 1 big-parallel
FIN

# 3. Acepta su clave de host (tambien como root)
sudo ssh -i /var/root/.ssh/id_constructor -o StrictHostKeyChecking=accept-new \
     root@IP_DEL_CONSTRUCTOR true

# 4. Activa la delegacion
echo "builders-use-substitutes = true" | sudo tee -a /etc/nix/nix.conf
sudo launchctl kickstart -k system/org.nixos.nix-daemon
```

Después, `nix build .#imagenes.voz-api` funciona desde el Mac: compila allí y
trae el resultado.

### O construir allí y copiar el fichero

Si es algo puntual, no merece la pena configurar nada:

```bash
ssh CONSTRUCTOR 'cd /ruta/al/repo && nix build .#imagenes.voz-api -o /tmp/voz-api.tar.gz'
scp CONSTRUCTOR:/tmp/voz-api.tar.gz .
```

No hay Dockerfile: las imágenes salen de los mismos paquetes que la VM NixOS,
así que **no pueden divergir** de lo que corre en producción.

## Qué esperar

| | RAM | RTF | Para qué |
|---|---|---|---|
| `voz-api` (Piper) | ~150 MB | **0,042** | Notas de voz. 24x tiempo real |
| `whisper` | ~600 MB | 0,671 | Entender audios |
| `voz-stream` | ~2,5 GB | 2,18 | Voz expresiva. Lento |

Medido en un Intel i7-8700T de 6 núcleos. **Piper es el que querrás para casi
todo**: responde en décimas de segundo. VibeVoice suena mejor pero tarda el
doble de lo que dura el audio.

## Decisiones que te pueden extrañar

**Tres contenedores y no uno.** Los perfiles de memoria son incomparables
(150 MB frente a 2,5 GB) y así una síntesis pesada no bloquea las notas de voz
rápidas.

**El modelo de whisper se monta, no va dentro.** Son 466 MB que atarían la
versión del modelo a la del contenedor. La imagen de VibeVoice sí lleva el
suyo dentro, porque sin él no arranca y son 1,9 GB que nadie querría montar a
mano.

**`voz-stream` pesa ~4 GB y no hay forma de evitarlo**: 1,9 GB de pesos más
~2 GB de PyTorch. Las capas están repartidas por tamaño, así que actualizar el
código reenvía megas y no gigas.

## Límites conocidos

- **Sin GPU.** Todo va en CPU. Se probó una iGPU Intel y resultó 2,5x más
  lenta; PyTorch tampoco soporta esa generación.
- **El streaming se entrecorta si el RTF pasa de 1.** A RTF 2,2 el audio se
  genera más despacio de lo que se oye. La página de prueba lo compensa
  esperando un poco antes de empezar.
- **Una síntesis a la vez** en `voz-stream`: hay un candado global porque el
  modelo ya satura todos los núcleos.
