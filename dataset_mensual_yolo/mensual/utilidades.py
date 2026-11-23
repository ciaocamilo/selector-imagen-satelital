"""E/S atómica, huellas, bloqueo de escritores y utilidades comunes."""
import csv
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def huella(obj, n=12):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:n]


def sha256_archivo(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def ahora_utc():
    return datetime.now(timezone.utc)


def iso_utc(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sin_url(texto):
    """Oculta enlaces temporales (llevan tokens): nunca se persisten URLs de descarga."""
    return re.sub(r"https?://\S+", "<url temporal>", str(texto))


def _reemplazar(tmp, ruta, intentos=8):
    for k in range(intentos):
        try:
            os.replace(tmp, ruta)
            return
        except PermissionError:  # OneDrive puede bloquear un instante
            if k == intentos - 1:
                raise
            time.sleep(0.5 * (k + 1))


def escribir_texto_atomico(ruta, texto):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_name(f".{ruta.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(texto)
            f.flush()
            os.fsync(f.fileno())
        _reemplazar(tmp, ruta)
    finally:
        if tmp.exists():
            tmp.unlink()
    return ruta


def escribir_json_atomico(ruta, datos):
    return escribir_texto_atomico(ruta, json.dumps(datos, ensure_ascii=False, indent=1, default=str))


def leer_json(ruta, defecto=None):
    try:
        return json.loads(Path(ruta).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return defecto


def escribir_csv_atomico(ruta, filas, columnas=None):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if columnas is None:
        columnas = []
        for f in filas:
            columnas += [k for k in f if k not in columnas]
    tmp = ruta.with_name(f".{ruta.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
            w.writeheader()
            for fila in filas:
                w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                            for k, v in fila.items()})
        _reemplazar(tmp, ruta)
    finally:
        if tmp.exists():
            tmp.unlink()
    return ruta


def leer_csv(ruta):
    with open(ruta, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mover_con_reintentos(origen, destino, intentos=8):
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    _reemplazar(origen, destino, intentos)


def borrar_arbol(ruta):
    ruta = Path(ruta)
    if not ruta.exists():
        return
    for k in range(6):
        try:
            shutil.rmtree(ruta)
            return
        except PermissionError:
            if k == 5:
                raise
            time.sleep(0.5 * (k + 1))


def enlazar_o_copiar(origen, destino, modo="fisico"):
    """Enlace físico local cuando se pide y el sistema lo admite; si no, copia."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    if modo == "fisico":
        try:
            os.link(origen, destino)
            return "enlace"
        except OSError:
            pass
    shutil.copy2(origen, destino)
    return "copia"


def _pid_vivo(pid):
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        codigo = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(codigo))
        k32.CloseHandle(h)
        return bool(ok) and codigo.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ErrorBloqueo(RuntimeError):
    pass


class Bloqueo:
    """Impide dos escritores simultáneos sobre el mismo destino; recupera bloqueos de procesos muertos."""

    def __init__(self, directorio):
        self.ruta = Path(directorio) / ".bloqueo"

    def __enter__(self):
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.ruta, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    json.dump({"pid": os.getpid(), "inicio": iso_utc(ahora_utc())}, f)
                return self
            except FileExistsError:
                datos = leer_json(self.ruta, {}) or {}
                pid = int(datos.get("pid", 0) or 0)
                if pid != os.getpid() and _pid_vivo(pid):
                    raise ErrorBloqueo(f"Otra ejecución (pid {pid}, desde {datos.get('inicio')}) escribe en "
                                       f"{self.ruta.parent}. Espera a que termine o usa otro directorio de salida.")
                self.ruta.unlink(missing_ok=True)
        raise ErrorBloqueo(f"No se pudo adquirir {self.ruta}")

    def __exit__(self, *exc):
        self.ruta.unlink(missing_ok=True)
        return False
