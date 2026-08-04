"""Cache en disco del audio ya generado.

Solo cachea por COINCIDENCIA EXACTA de (texto, voz, formato, velocidad). No
trocea palabras para recombinarlas: la prosodia en un modelo neuronal es de
frase entera, asi que una palabra recortada de una afirmacion suena mal dentro
de una pregunta, y en las uniones aparecen saltos de tono audibles. Ademas
Piper va a RTF 0,115, asi que lo que se ahorraria pegando trozos son decimas
de segundo a cambio de que se note el pegado.

Donde el ahorro si es grande es en VibeVoice (RTF ~4,2): cada acierto evita
unos 45 segundos de computo.
"""

import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path


class CacheAudio:
    """Cache LRU en disco, con indice en memoria.

    Es segura entre hilos, pero asume un unico proceso escribiendo: el
    servicio corre con --workers 1 justo por eso (las voces de Piper tambien
    viven en memoria del proceso).
    """

    def __init__(self, directorio: Path, limite_mb: int = 512, activa: bool = True):
        self.dir = Path(directorio)
        self.limite = limite_mb * 1024 * 1024
        self.activa = activa
        self._lock = threading.Lock()
        self.aciertos = 0
        self.fallos = 0

        if self.activa:
            self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def clave(texto: str, voz: str, formato: str, velocidad: float) -> str:
        """Hash de la peticion. La velocidad se redondea porque un float que
        difiere en el septimo decimal produce audio identico pero otra clave."""
        crudo = json.dumps(
            {
                "texto": texto.strip(),
                "voz": voz,
                "formato": formato,
                "velocidad": round(velocidad, 3),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(crudo.encode("utf-8")).hexdigest()

    def _ruta(self, clave: str, formato: str) -> Path:
        # Dos niveles de subdirectorio: un solo directorio con miles de
        # ficheros hace lentos los listados en ext4.
        return self.dir / clave[:2] / f"{clave}.{formato}"

    def leer(self, clave: str, formato: str) -> bytes | None:
        if not self.activa:
            return None
        ruta = self._ruta(clave, formato)
        try:
            datos = ruta.read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            with self._lock:
                self.fallos += 1
            return None

        # mtime = ultimo uso, que es lo que ordena el desalojo LRU.
        try:
            os.utime(ruta, None)
        except OSError:
            pass
        with self._lock:
            self.aciertos += 1
        return datos

    def guardar(self, clave: str, formato: str, datos: bytes) -> None:
        if not self.activa:
            return
        ruta = self._ruta(clave, formato)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        # Escritura atomica: si el proceso muere a medias no queda un fichero
        # truncado que luego se sirva como si fuera bueno.
        tmp = ruta.with_suffix(ruta.suffix + f".{os.getpid()}.tmp")
        try:
            tmp.write_bytes(datos)
            tmp.replace(ruta)
        except OSError:
            tmp.unlink(missing_ok=True)
            return
        self._desalojar()

    def _desalojar(self) -> None:
        """Borra lo menos usado hasta bajar del limite."""
        ficheros = []
        total = 0
        for f in self.dir.rglob("*"):
            if not f.is_file() or f.name.endswith(".tmp"):
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            ficheros.append((st.st_mtime, st.st_size, f))
            total += st.st_size

        if total <= self.limite:
            return

        ficheros.sort()  # mas antiguo primero
        for _, tam, f in ficheros:
            if total <= self.limite * 0.9:  # baja al 90% para no desalojar en cada escritura
                break
            try:
                f.unlink()
                total -= tam
            except OSError:
                pass

    def vaciar(self) -> int:
        """Borra todo. Devuelve los bytes liberados."""
        liberado = self.estadisticas()["bytes"]
        if self.activa and self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)
            self.dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.aciertos = 0
            self.fallos = 0
        return liberado

    def estadisticas(self) -> dict:
        entradas = 0
        total = 0
        if self.activa and self.dir.exists():
            for f in self.dir.rglob("*"):
                if f.is_file() and not f.name.endswith(".tmp"):
                    try:
                        total += f.stat().st_size
                        entradas += 1
                    except OSError:
                        pass
        with self._lock:
            aciertos, fallos = self.aciertos, self.fallos
        consultas = aciertos + fallos
        return {
            "activa": self.activa,
            "entradas": entradas,
            "bytes": total,
            "mb": round(total / 1048576, 2),
            "limite_mb": round(self.limite / 1048576),
            "aciertos": aciertos,
            "fallos": fallos,
            "tasa_acierto": round(aciertos / consultas, 3) if consultas else None,
        }
