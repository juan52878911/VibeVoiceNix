# El asistente con herramientas

Un asistente de casa que **habla, decide usar una herramienta, la usa y
contesta por voz**. Los datos que devuelven las herramientas son **simulados**
y viven en un fichero editable; el mecanismo —uso de herramientas nativo de la
API de Anthropic contra MiniMax— es el de verdad, así que lo único que hay que
cambiar el día que se enchufe un calendario real es de dónde salen las listas.

---

## Arrancarlo

```bash
# La VM tiene que estar sirviendo voz-stream (puerto 8082).
# El token está en /var/lib/voz/token.env de la VM.
export VOZ_TOKEN=$(ssh root@192.168.2.54 'sed s/VOZ_TOKEN=// /var/lib/voz/token.env')

pkgs/vibevoice/.venv/bin/python scripts/asistente_web.py \
    --voz-url http://192.168.2.54:8082
```

Y abrir <http://127.0.0.1:8090>. En el arranque imprime:

```
  herr: 10 herramientas SIMULADAS (4 escriben y se confirman por voz) · datos en
        .../herramientas_simuladas.json
  perf: agenda, general, servidor (activo: general) · rellenos en ...
```

Con micrófono desde otra máquina de la red hace falta HTTPS (el navegador solo
da micrófono en contexto seguro, y `localhost` es la única excepción):

```bash
pkgs/vibevoice/.venv/bin/python scripts/asistente_web.py \
    --host 0.0.0.0 --voz-url http://192.168.2.54:8082 \
    --tls-cert ~/.config/vibevoicenix/puente.crt \
    --tls-clave ~/.config/vibevoicenix/puente.key
```

Para apagar las herramientas y dejarlo como estaba: `--sin-herramientas`.

---

## Qué se le puede pedir

Los tres perfiles se eligen **en la propia página**, en el desplegable de
arriba. Cambiar de perfil cambia a la vez la voz, las coletillas pregrabadas y
el juego de herramientas: son otro asistente, no el mismo con otro sombrero.

| Perfil | Voz | Puede |
|---|---|---|
| **General** | `sp-Spk1_man` | las seis áreas: agenda, correo, servidor, notas, recordatorios y web |
| **Servidor** | `sp-Spk3_man` | estado de la máquina y notas de proyecto. **No** puede mandar correos ni tocar la agenda |
| **Agenda** | `sp-Spk5_man` | agenda, correo y recordatorios. Es el que más escribe, y por tanto donde mejor se ve la confirmación por voz |

### Leer (contesta y ya)

- «¿Qué tengo mañana?» · «¿Tengo algo el jueves?» · «¿Estoy libre a las cinco?»
- «¿Hay algo urgente en el correo?» · «¿Qué quería Elena?» · «¿Tengo correo sin leer?»
- «¿Cómo va el servidor de casa?» · «¿Se ha caído algo?» · «¿Cuánto disco queda?»
- «¿Por dónde iba lo de la mudanza del NAS?» · «¿Qué proyecto tengo abandonado?»
- «¿Qué tengo pendiente?» · «¿Se me ha pasado algo?»
- «Búscame en internet cómo se hace una paella» · «¿Qué tiempo hace?»

### Escribir (pregunta antes, siempre)

- «Ponme mañana a las siete y media una cena con Bea»
- «Mándale un correo a Elena con el desglose de coste de la VM: cómputo 41, disco 12, respaldo 7»
- «Apúntame que tengo que llamar al fontanero el lunes»
- «Borra la nota del huerto de la terraza»

Cualquiera de esas cuatro contesta con la frase exacta de lo que va a hacer y
un **¿lo hago?**. A partir de ahí:

- **«sí», «vale», «venga», «dale», «hazlo»** → lo hace, y contesta con una
  palabra: *Enviado.* / *Apuntado.* / *Borrada.*
- **«no», «déjalo», «cancela», «mejor no»** → *Vale, lo dejo.*
- **cualquier otra cosa** → se descarta la acción y esa frase se trata como una
  pregunta nueva. Un «¿y qué hora es?» no es un sí.

También hay un botón **Cancelar** en la página: una acción a medias tiene que
poder tirarse sin discutirlo por voz.

---

## Por qué escribir se confirma y leer no

whisper se equivoca. En este proyecto está medido: **9,7 % de WER medio y
11,1 % en el peor caso** con el prompt bueno. Un 10 % de palabras mal en «borra
la nota del huerto» puede ser una frase que nadie dijo. Leer un dato mal
entendido cuesta una respuesta absurda; **mandar un correo mal entendido no se
deshace**.

La garantía no vive en el prompt. Vive en el código, y es que **no hay camino**:

- `crear_evento`, `enviar_correo`, `crear_recordatorio` y `borrar_nota` **no
  ejecutan nada, nunca**. Lo único que hacen es anotar lo que se haría y
  devolver un resumen en castellano.
- La única función del programa que escribe es `Ejecutor.confirmar()`, y para
  llegar a ella hace falta que exista una acción anotada.
- El sí y el no **no pasan por el modelo**: los clasifica
  `herramientas.clasificar_respuesta()`, que es una lista cerrada de palabras y
  ni siquiera toca la red. Lo que no está en la lista no es un sí.
- Se ejecutan **los argumentos que se dijeron en voz alta**, guardados con la
  anotación, y no una reformulación posterior. Lo que se oye y lo que se hace
  salen del mismo sitio.
- Una confirmación **caduca en dos turnos** (y a los 5 minutos). Un «sí» que
  llega media conversación después es un sí a otra cosa.
- Con la **escucha continua**, el «sí» espera al veredicto de la compuerta de
  destinatario antes de tocar nada. Es el único sitio del puente donde vale la
  pena pagar sus 0,47-0,70 s en el camino crítico: en una habitación con gente,
  alguien diciéndole que sí a otra persona no puede mandar un correo. Si la
  compuerta no contesta a tiempo, no se confirma y la acción se queda esperando.

Lo peor que puede conseguir un modelo equivocado —o al que le cuelen una
instrucción dentro del texto de un correo— es que el asistente **pregunte**.

El detalle de por qué no vale el diseño evidente (un parámetro
`confirmado: true` y una comprobación) está en la cabecera de
[`scripts/herramientas.py`](../scripts/herramientas.py), con las medidas de los
tres intentos.

---

## Qué está simulado y qué es real

| | |
|---|---|
| **Simulado** | los datos: eventos, correos, estado de servicios, notas, recordatorios y resultados de búsqueda. Todo sale de `herramientas_simuladas.json` |
| **Simulado** | las escrituras: crear un evento o «enviar» un correo solo apunta la acción en la sección `escrituras` de ese mismo fichero. **De esta máquina no sale ningún correo** |
| **Real** | que el modelo decida qué herramienta usar y con qué argumentos (uso de herramientas nativo de la API de Anthropic) |
| **Real** | que el resultado vuelva al modelo y este conteste con él |
| **Real** | la voz, la sesión de TTS, los audios pregrabados, la escucha continua, la compuerta de destinatario y la interrupción |
| **Real** | la regla de confirmación entera, incluida la clasificación del sí |

### Editar los datos con el asistente en marcha

`herramientas_simuladas.json` **se relee solo** cuando cambia su fecha de
modificación. Se puede tumbar un servicio, añadir un correo o vaciar el
calendario y la siguiente pregunta ya contesta con eso. Los días van
**relativos a hoy** (`"dia": 0` es hoy, `1` mañana, `-1` ayer) para que el
ejemplo no caduque.

Para dejarlo como estaba, vaciar las cuatro listas de `escrituras`.

---

## Sustituir cada simulación por lo real

Cada herramienta lleva escrito por qué se cambia. Se ve todo junto con:

```bash
pkgs/vibevoice/.venv/bin/python scripts/herramientas.py listar
```

| Herramienta | Se sustituye por |
|---|---|
| `consultar_calendario` / `crear_evento` | CalDAV (`PUT` de un `.ics`) o la API de Google Calendar |
| `leer_correo` / `enviar_correo` | IMAP y SMTP, o el conector de Gmail. El resumen de cada mensaje lo escribe hoy un humano en el JSON; con correo real lo hace el propio modelo |
| `estado_servidor` | `systemctl` y `df` por SSH, o el `/health` de la VM más un exportador de nodo |
| `buscar_notas` / `borrar_nota` | el MCP de Obsidian, que ya está **declarado y deshabilitado** en el perfil `servidor` (`obsidian_notas`) |
| `ver_recordatorios` / `crear_recordatorio` | la app de Recordatorios por AppleScript, o una lista CalDAV `VTODO` |
| `buscar_en_web` | una API de búsqueda (Brave, Tavily, o SearXNG en la propia LAN). Es la única que de verdad va a tardar: hay que contar con 0,5-2 s más |

**El único punto de cambio es el método `_<nombre>` del `Ejecutor`.** Devuelve
un diccionario y no sabe nada del modelo ni de la voz. El catálogo, la regla de
confirmación, los sucesos de la página y las pruebas siguen igual.

Lo que **no** hay que tocar al enchufar lo real:

- `Ejecutor.confirmar()` y `clasificar_respuesta()`. Son justo la pieza que
  gana valor cuando el correo sale de verdad.
- El campo `escribe` del catálogo. Es el interruptor de la regla de seguridad y
  no se deduce del nombre a propósito: que una herramienta sea peligrosa tiene
  que estar **dicho**.

---

## Los rellenos, y los dos que faltaban

Los audios pregrabados (`scripts/perfiles.py`) pasan de cinco categorías a
siete. Las dos nuevas y los dos disparadores que faltaban:

| Categoría | Cuándo suena |
|---|---|
| `afirmacion` | a 0,70 s, con el veredicto de la compuerta |
| `pensando` | a 1,20 s, mientras el LLM redacta |
| `esperando` | a 3,50 s, si sigue tardando |
| **`consultando`** | arranca una herramienta **y el modelo no dijo nada antes de llamarla**. Cuando sí lo dice («voy a mirar tu calendario»), esa frase ya es el relleno, y es mejor: nombra lo que de verdad se está consultando |
| **`confirmando`** | el remate de una acción confirmada. **No es intercambiable**: se elige por texto, porque decir «apuntado» cuando lo que se hizo fue borrar sería mentir. Al estar pregrabado, suena en **~0,01 s** en vez de costar una locución entera |
| **`negacion`** | la respuesta se cae **sin haber sonado nada**: una herramienta que revienta, el LLM que no contesta, la sesión de voz rota antes del primer PCM |
| **`cerrando`** | la locución se corta **a mitad, con audio ya sonando**: el freno de descarrile o el plazo de silencio. Es el único caso en que añadir palabras que el LLM no dijo mejora la cosa: la alternativa es una frase que se corta en seco, que suena a avería |

`negacion` y `cerrando` son excluyentes por construcción —los separa «¿ya
sonaba algo?»— y ninguno de los dos suena si la respuesta sale bien.

Regenerar tras editar los perfiles:

```bash
VOZ_STREAM_URL=http://192.168.2.54:8082 VOZ_TOKEN=... \
  pkgs/vibevoice/.venv/bin/python scripts/perfiles.py generar
```

---

## Probarlo sin navegador y sin oír nada

```bash
VOZ_TOKEN=... pkgs/vibevoice/.venv/bin/python scripts/herramientas_extremo.py
```

Levanta un puente propio **en el 8099** —el del usuario vive en el 8090 y no se
toca— con una **copia** del fichero de datos, manda preguntas por
`POST /preguntar` y lee los mismos marcos binarios que leería el navegador. El
PCM se cuenta y se tira: no suena nada.

Comprueba, con 30 aserciones: que se llama a la herramienta que toca en cada
una de las seis áreas; que lo que se dice **no lleva JSON dentro**; que una
escritura no toca el fichero antes del sí; que una pregunta cualquiera no es un
sí; que un «sí» suelto sin nada preparado no escribe nada; y que el «no» no
borra.

También hay un ciclo suelto, sin voz ni puente:

```bash
pkgs/vibevoice/.venv/bin/python scripts/herramientas.py ciclo "¿qué tengo mañana?"
pkgs/vibevoice/.venv/bin/python scripts/herramientas.py llamar leer_correo filtro=importantes
```

---

## Tres cosas que costaron caro (y no se ven en el resultado)

**El modelo deja de llamar a las herramientas al tercer turno.** Turno 1
perfecto, turno 2 perfecto, turno 3 se inventa el estado del servidor. Y no es
la longitud del contexto: con un historial de dos chistes, igual de largo, el
turno 3 llama a la herramienta sin problema (2 de 2 con chistes, 0 de 2 con
datos). Lo que pasa es que un historial donde el asistente suelta horas y
remitentes **sin que se vea una sola llamada** le enseña que aquí se contesta
de memoria. La causa es que el historial que lleva la página es lo que se
*dijo*, y las llamadas no se dicen. Se arregla volviendo a meterlas en su sitio
al reconstruir los mensajes: `assistant(tool_use) → user(tool_result) →
assistant(texto)`. Con eso, 7 de 7.

**Lo que dice el programa no puede entrar en el hueco del modelo.** Con la
pregunta de confirmación y el «Borrada.» metidos en el historial como respuestas
suyas, el modelo aprende el estilo y a la siguiente contesta «Apuntado.» sin
llamar a nada: 10 de 12 aciertos sin ese historial, 4 de 12 con él, y dos veces
repitiendo la frase palabra por palabra. Reescribirlas en tercera persona sale
**igual de mal**, y encima el modelo se pone a contestar entre paréntesis. Lo
que copia no es el contenido, es el molde. Así que esos turnos se saltan
enteros —**en pareja con la pregunta que los provocó**, o queda una petición sin
responder y la vuelve a preparar dos turnos después— y el hecho se cuenta por
el bloque de sistema.

**Explicarle el mecanismo le da algo que contar en vez de algo que hacer.** La
instrucción decía «de preguntarle al usuario se encarga el sistema», y el
modelo empezó a decirlo en voz alta: «el sistema está a punto de crear un
recordatorio, ¿lo dejo?», sin llamar a nada. Ahora la instrucción solo dice qué
hacer.

---

## Latencias medidas

Contra la VM (i7-8700T, sin GPU), MiniMax-M3, medido por
`scripts/herramientas_extremo.py`, que cronometra desde que sale la pregunta
del cliente:

| | primer sonido (lo que se oye) | primer PCM (voz sintetizada) |
|---|---|---|
| sin herramienta | 1,22 s (1,22-1,22, n=4) | 1,44-2,77 s |
| con herramienta, las seis áreas | 1,13 s de media (0,94-1,22, n=9) | 1,53-4,00 s con la VM libre |
| remate confirmado («Apuntado.») | **0,004 s** (0,003-0,013, n=5) | no hay: es un WAV pregrabado |

En pasadas seguidas contra la VM, el **primer PCM** llega a irse a 8-37 s en
algún caso suelto: es la cola del sintetizador, que tiene un solo modelo con un
candado y todavía está terminando la locución anterior. **El primer sonido no
se mueve** —sigue en 1,2 s— porque para entonces ya está sonando la coletilla o
la frase de aviso del propio modelo. Es justo lo que compran los rellenos.

El 1,22 s del primer sonido no es casualidad: es el umbral del relleno
`pensando`, que está en 1,20 s. En la pasada en que el modelo escribió su
frase de aviso antes de ese umbral, el relleno **no llegó a sonar** y el primer
sonido fue la voz de verdad a 1,02 s. Es el mismo hueco tapado de dos maneras,
y la buena es la del modelo.

**El primer sonido no empeora por usar una herramienta**, y a veces mejora. El
motivo es que el modelo suele escribir una frase antes de llamarla —«voy a
mirar tu calendario»— y esa frase sale por el canal de texto y se narra ya,
mientras la herramienta se ejecuta y él redacta la respuesta de verdad. Es un
relleno gratis, escrito por el propio modelo y a medida de lo que se le ha
pedido.

Lo que sí crece es el **primer PCM**: una herramienta mete una vuelta más de
LLM, y encima el contexto de la segunda vuelta lleva el resultado. Se nota poco
porque para entonces ya se está hablando.

El **remate confirmado es el camino más corto de todo el sistema**: ni LLM ni
sesión de voz ni VM. El navegador tiene el WAV decodificado desde que abrió la
página, así que después de decir «sí» el «Enviado.» empieza con la respuesta
HTTP. Es justo donde más se agradece.

> Ojo al medir: `urllib` con `read(65536)` espera a tener los 65 KB o a que
> cierren, así que el primer marco no aparece hasta que hay 64 KB de PCM
> detrás. Con eso, «primer sonido» daba 6,8 s donde el puente decía 1,1 s, y la
> culpa era del medidor. Hay que usar `read1`.
