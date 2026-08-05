# Usar el stack de voz con Docker

Para quien **no tiene Nix**. Las imágenes se construyen con Nix, pero usarlas
solo requiere Docker.

## Si te han pasado las imágenes

```bash
docker load < voz-api.tar.gz
docker load < voz-whisper.tar.gz

./bajar-modelo.sh                    # el modelo de whisper, 466 MB
echo "VOZ_TOKEN=$(openssl rand -hex 24)" > .env

docker compose up -d
```

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
docker load < voz-stream.tar.gz
docker compose --profile pesado up -d
```

Página de prueba en `http://localhost:8082/`.

**Necesita ~6 GB de RAM libres.** Pide 2,5 GB en régimen pero hace un pico de
4,6 GB al arrancar, porque carga el modelo en fp32 antes de cuantizarlo. Y
tarda **unos 2 minutos** en aceptar la primera petición: carga el modelo y
hace una síntesis de calentamiento para no pagarla en tu primera llamada.

## Si vas a construir las imágenes

Necesitas Nix con flakes, y que el destino sea `x86_64-linux`:

```bash
nix build .#imagenes.voz-api     -o voz-api.tar.gz
nix build .#imagenes.whisper     -o voz-whisper.tar.gz
nix build .#imagenes.voz-stream  -o voz-stream.tar.gz
```

No hay Dockerfile: las imágenes salen de los mismos paquetes que la VM NixOS,
así que **no pueden divergir** de lo que corre en producción.

Desde un Mac ARM hace falta un constructor remoto x86_64 — las imágenes
contienen binarios de esa arquitectura.

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
