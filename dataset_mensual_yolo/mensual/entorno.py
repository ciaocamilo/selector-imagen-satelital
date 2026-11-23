"""Ajuste del entorno conda (PROJ/GDAL y DLL) antes de importar rasterio/pyproj (igual que la celda 2a del selector)."""
import os
import sys
import warnings
from pathlib import Path


def preparar_entorno():
    for var, subdirs in {"PROJ_DATA": ("Library/share/proj", "share/proj"),
                         "GDAL_DATA": ("Library/share/gdal", "share/gdal")}.items():
        for sub in subdirs:
            ruta = Path(sys.prefix) / sub
            if ruta.is_dir():
                os.environ[var] = str(ruta)
                if var == "PROJ_DATA":
                    os.environ["PROJ_LIB"] = str(ruta)
                break
    if os.name == "nt":
        dll = [str(Path(sys.prefix) / d) for d in ("Library/bin", ".") if (Path(sys.prefix) / d).is_dir()]
        os.environ["PATH"] = os.pathsep.join(dll + [p for p in os.environ.get("PATH", "").split(os.pathsep)
                                                    if p not in dll])
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\..*")
