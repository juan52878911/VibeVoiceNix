#!/usr/bin/env python3
"""Normalizador de texto (es / en): lo que el modelo lee mal, escrito como se dice.

Sirve a los dos lados de la fase 2 del plan de mejora:
  - al MODELO: le llega "la sala doscientos cuatro" en vez de "la sala 204", y siglas y codigos
    deletreados ("a, uve doble, ese") en vez de "AWS", que medido (22-09) leia como "Albol S";
  - al JUEZ: el WER de numeros se inflaba porque whisper escribe "3.30" donde el texto pone "3:30",
    "$40,000" por "40,000 dollars" o "two" por "2". Normalizando la referencia Y lo oido con la
    misma funcion, solo cuentan los errores de verdad.

  from normalizar_texto import normalizar
  normalizar("El vuelo IB6843 sale a las 23:45.", "es")
  -> "El vuelo i be seis ocho cuatro tres sale a las veintitres cuarenta y cinco."
"""
import re
import sys

from num2words import num2words

LETRAS = {
    "es": {"A": "a", "B": "be", "C": "ce", "D": "de", "E": "e", "F": "efe", "G": "ge", "H": "hache", "I": "i",
           "J": "jota", "K": "ka", "L": "ele", "M": "eme", "N": "ene", "Ñ": "eñe", "O": "o", "P": "pe", "Q": "cu",
           "R": "erre", "S": "ese", "T": "te", "U": "u", "V": "uve", "W": "uve doble", "X": "equis", "Y": "ye",
           "Z": "zeta"},
    "en": {c: c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
}
# Siglas que se dicen como palabra y no se deletrean
COMO_PALABRA = {"OTAN", "NASA", "UNESCO", "OVNI", "SIDA", "PYME", "ONU", "FIFA", "OK"}
MONEDA = {"es": {"$": "dólares", "€": "euros", "USD": "dólares", "EUR": "euros"},
          "en": {"$": "dollars", "€": "euros", "USD": "dollars", "EUR": "euros"}}
PORCIENTO = {"es": "por ciento", "en": "percent"}
COMA = {"es": "coma", "en": "point"}


def _entero(n, idioma):
    return num2words(int(n), lang=idioma)


def _numero(txt, idioma):
    """'40.000' / '40,000' / '12,5' / '12.5' -> palabras, segun las convenciones del idioma."""
    miles, decimal = (".", ",") if idioma == "es" else (",", ".")
    t = txt
    if re.fullmatch(rf"\d{{1,3}}(\{miles}\d{{3}})+", t):          # separador de miles
        return _entero(t.replace(miles, ""), idioma)
    # decimal: el separador del idioma, o el otro si no forma miles ("8.1" en un texto espanol que viene del
    # ingles, o lo que escribe whisper). Antes "8.1" en espanol reventaba en int().
    for sep in (decimal, miles):
        if sep in t:
            ent, dec = t.rsplit(sep, 1)
            ent = ent.replace(miles, "").replace(decimal, "")
            if ent.isdigit() and dec.isdigit():
                return f"{_entero(ent, idioma)} {COMA[idioma]} " + " ".join(_entero(d, idioma) for d in dec)
    t = re.sub(r"\D", "", t)
    return _entero(t, idioma) if t else ""


def _deletrear(tok, idioma):
    partes = []
    for c in tok:
        if c.isdigit():
            partes.append(_entero(c, idioma))
        elif c.upper() in LETRAS[idioma]:
            partes.append(LETRAS[idioma][c.upper()])
    return " ".join(partes)


def _hora(h, m, idioma, sufijo=""):
    h, m = int(h), int(m)
    hh = _entero(h, idioma)
    if m == 0:
        s = hh + (" en punto" if idioma == "es" else " o'clock")
    elif idioma == "es":
        s = f"{hh} y {_entero(m, idioma)}" if m < 10 or m == 30 else f"{hh} {_entero(m, idioma)}"
    else:
        s = f"{hh} oh {_entero(m, idioma)}" if m < 10 else f"{hh} {_entero(m, idioma)}"
    if sufijo:
        s += " " + " ".join(sufijo.upper().replace(".", ""))
    return s


def normalizar(texto, idioma="es", siglas=True):
    """siglas=False no deletrea siglas ni codigos: en espanol la mejora fue pequena y un clip empeoro
    (F2b del plan de mejora), y voz_stream.py lo usa asi para el espanol."""
    idioma = "es" if idioma.startswith("es") else "en"
    t = texto
    # horas: 23:45, 3:30 PM, 3.30 (lo que escribe whisper)
    t = re.sub(r"\b(\d{1,2})[:.](\d{2})\s*([AaPp]\.?[Mm]\.?)?(?=\W|$)",
               lambda m: _hora(m.group(1), m.group(2), idioma, m.group(3) or ""), t)
    # moneda delante ($40,000) o sigla detras (40.000 USD)
    t = re.sub(r"([$€])\s?(\d[\d.,]*\d|\d)", lambda m: f"{m.group(2)} {MONEDA[idioma][m.group(1)]}", t)
    # porcentaje
    t = re.sub(r"(\d[\d.,]*)\s?%", lambda m: f"{m.group(1)} {PORCIENTO[idioma]}", t)
    # ordinales en ingles (8th, 15th, 1st, 22nd, 3rd)
    if idioma == "en":
        t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: num2words(int(m.group(1)), lang="en", to="ordinal"), t)
    if siglas:
        # codigos alfanumericos (IB6843, UA-208M): se deletrean
        t = re.sub(r"\b(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9-]{3,}\b",
                   lambda m: _deletrear(m.group(0).replace("-", ""), idioma), t)
        # siglas en mayusculas (AWS, API): se deletrean, salvo las que se dicen como palabra
        t = re.sub(r"\b[A-ZÑ]{2,5}\b", lambda m: m.group(0) if m.group(0) in COMO_PALABRA
                   else _deletrear(m.group(0), idioma), t)
    # numeros que queden
    t = re.sub(r"\d[\d.,]*\d|\d", lambda m: _numero(m.group(0).rstrip(".,"), idioma), t)
    return re.sub(r"\s+", " ", t).strip()


# Palabras funcionales EXCLUSIVAS de cada idioma: las compartidas ("de", "la", "que", "en", "in", "was",
# "will"...) no cuentan. Con las compartidas, un texto frances salia "espanol" y uno aleman "ingles", y el
# normalizador escribia cifras en otro idioma (medido el 23-09 con la charla doblada al frances).
_FUNCIONALES = {
    "es": {"el", "los", "del", "pero", "muy", "hay", "usted", "eso", "esto", "porque", "cuando", "donde",
           "entonces", "ahora", "nosotros", "ellos", "tiene", "puede", "tambien", "también", "algo", "mucho"},
    "en": {"the", "and", "of", "that", "you", "with", "this", "they", "have", "from", "are", "is", "it",
           "for", "not", "but", "we", "my", "be", "been", "would", "there", "their", "what", "which"},
    "fr": {"le", "les", "des", "est", "et", "pour", "pas", "avec", "vous", "nous", "je", "qui", "dans",
           "sur", "ce", "cette", "sont", "mais", "très", "au", "aux", "du", "ils", "elle"},
    "de": {"der", "die", "das", "und", "ist", "nicht", "ich", "sie", "mit", "auf", "ein", "eine", "den",
           "dem", "zu", "auch", "wir", "sind", "oder", "wie", "wenn"},
    "it": {"il", "gli", "della", "che", "non", "sono", "questo", "anche", "molto", "perché", "ma", "ci",
           "nel", "degli", "delle", "loro", "essere", "cosa"},
    "pt": {"os", "não", "você", "isso", "então", "são", "está", "ao", "pelo", "pela", "uma", "com",
           "muito", "também", "nós", "eles", "tem", "mais", "mas"},
}


def idioma_probable(texto):
    """'es' | 'en' | None. Cuenta palabras funcionales exclusivas de es, en, fr, de, it y pt; decide solo si
    gana es o en con al menos 2 palabras y el doble que cualquier otro idioma. Si no esta claro, None (y
    el normalizador no toca nada)."""
    pal = re.findall(r"[a-záéíóúñüàâçèêëîïôûœäößãõ]+", texto.lower())
    cuenta = {k: sum(p in v for p in pal) for k, v in _FUNCIONALES.items()}
    ganador = max(cuenta, key=cuenta.get)
    resto = max(v for k, v in cuenta.items() if k != ganador)
    if ganador in ("es", "en") and cuenta[ganador] >= 2 and cuenta[ganador] >= 2 * resto:
        return ganador
    return None


_HAY_ALGO = re.compile(r"\d|[$€%]|\b[A-ZÑ]{2,5}\b")


def normalizar_para_motor(texto, modo="auto"):
    """Lo que usa voz_stream.py: linea a linea (los saltos de linea y los parrafos se conservan: las
    sesiones y la forma los usan) y SOLO en las lineas con cifras, simbolos o siglas; el resto sale
    identico, byte a byte. modo: 'no', 'es', 'en' o 'auto' (idioma por palabras funcionales; si no
    esta claro, no se toca). En espanol, sin deletrear siglas."""
    if modo == "no" or not _HAY_ALGO.search(texto):
        return texto
    idioma = idioma_probable(texto) if modo == "auto" else modo
    if idioma not in ("es", "en"):
        return texto
    lineas = texto.split("\n")
    return "\n".join(normalizar(l, idioma, siglas=(idioma == "en")) if _HAY_ALGO.search(l) else l
                      for l in lineas)


if __name__ == "__main__":
    idioma = sys.argv[1] if len(sys.argv) > 1 else "es"
    for linea in sys.stdin:
        print(normalizar(linea.strip(), idioma))
