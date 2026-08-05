# Colección de Bruno

Peticiones listas para la API de voz. Ábrela en Bruno con **Open Collection**
y apunta a esta carpeta.

## Antes de empezar

**1. Elige el entorno** (arriba a la derecha):

| Entorno | Cuándo |
|---|---|
| `homelab` | En casa, misma red |
| `tunel` | Fuera, con WireGuard activo |

**2. Pon el token.** Es una variable *secreta*, así que no se guarda en el
repo: hay que ponerla a mano una vez por entorno.

Sácalo de la VM:

```bash
ssh root@192.168.2.54 'cat /var/lib/voz/token.env'
```

En Bruno: engranaje del entorno → variable `token` → pega el valor.

## Qué hay

```
Estado/       salud y voces — sin token, para comprobar que responde
TTS/          generar audio con Piper (rápido: RTF 0,04)
STT/          transcribir audio con whisper
Streaming/    VibeVoice, en el puerto 8082
```

## El flujo que querrás probar primero

1. **Estado → Salud** — confirma que responde y qué voces hay
2. **TTS → Nota de voz (ogg)** — genera audio; guárdalo con el icono de
   descarga del panel de respuesta, extensión `.ogg`
3. **STT → Entender un audio** — súbele ese mismo fichero y comprueba que la
   transcripción coincide

Ese ciclo cierra las dos direcciones: mandar y entender audios.

## Cosas que confunden al principio

**El audio se ve como texto ilegible.** Es binario; Bruno lo muestra crudo.
Usa **Save Response** y ponle la extensión correcta.

**La primera petición a una voz tarda ~1 s más.** Las voces se cargan en
memoria al usarlas por primera vez. A partir de ahí, RTF 0,04.

**El streaming no se aprecia en Bruno.** Espera a tener la respuesta completa
antes de mostrarla, así que ves el resultado final pero no el goteo. Para eso
está la página en `{{stream}}/` o `curl -sN | ffplay`.

**El puerto 8082 es otro servicio.** `/tts` está en 8080 (Piper) y
`/tts/stream` en 8082 (VibeVoice). Son motores distintos, no dos rutas del
mismo.

## Equivalentes en curl

Por si prefieres la terminal:

```bash
API=http://192.168.2.54:8080
TOKEN=$(ssh root@192.168.2.54 'grep VOZ_TOKEN /var/lib/voz/token.env | cut -d= -f2')

# Generar
curl -X POST $API/tts \
  -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"texto":"El backup terminó sin errores."}' \
  -o nota.ogg

# Entender
curl -X POST $API/stt \
  -H "Authorization: Bearer $TOKEN" \
  -F archivo=@nota.ogg -F idioma=es
```
