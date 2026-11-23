"""CLI: python -m mensual --config configuracion.toml [opciones]."""
import argparse
import json
import sys

from . import cargar_configuracion, ejecutar, regenerar_yaml
from .gee import ErrorAutenticacion


def expandir_meses(texto):
    """'2016-01,2024-01' o rangos '2016-01..2016-12' (combinables)."""
    meses = []
    for parte in (p.strip() for p in texto.split(",") if p.strip()):
        if ".." in parte:
            a, b = parte.split("..")
            (ya, ma), (yb, mb) = map(int, a.split("-")), map(int, b.split("-"))
            while (ya, ma) <= (yb, mb):
                meses.append(f"{ya:04d}-{ma:02d}")
                ya, ma = (ya + 1, 1) if ma == 12 else (ya, ma + 1)
        else:
            meses.append(parte)
    return meses


def main(argv=None):
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(prog="python -m mensual", description=__doc__)
    ap.add_argument("--config", help="archivo TOML (por defecto, valores de Santa Isabel 2016–2026)")
    ap.add_argument("--meses", help="subconjunto: 2016-01,2024-01 o 2016-01..2016-12")
    ap.add_argument("--corte", help="corte UTC fijo (YYYY-MM-DD o ISO 8601); por defecto, ahora")
    ap.add_argument("--salida", help="directorio de salida (relativo al TOML)")
    ap.add_argument("--proyecto", help="proyecto de Earth Engine")
    ap.add_argument("--solo-l2a", action="store_true", help="sin respaldo L1C (meses previos a L2A: fuera_coleccion)")
    ap.add_argument("--sin-red", action="store_true", help="solo caché local; lo que requiera red queda reintentable")
    ap.add_argument("--no-reintentar", action="store_true", help="no reintenta meses error_reintentable")
    ap.add_argument("--no-refrescar", action="store_true", help="no vuelve a consultar el mes provisional")
    ap.add_argument("--sin-verificar-ultralytics", action="store_true")
    ap.add_argument("--regenerar-yaml", metavar="CARPETA_VERSION", help="reescribe path: de ambos data.yaml y sale")
    ap.add_argument("--regresion-piloto", nargs="?", const="", metavar="CARPETA",
                    help="regresión sin red con la escena piloto del selector y sale")
    a = ap.parse_args(argv)
    if a.regenerar_yaml:
        for r in regenerar_yaml(a.regenerar_yaml):
            print(f"YAML regenerado: {r}")
        return 0
    cambios = {"periodo__meses": expandir_meses(a.meses) if a.meses else None, "periodo__corte_utc": a.corte,
               "salida__directorio": a.salida, "gee__proyecto": a.proyecto,
               "fuentes__modo": "solo_l2a" if a.solo_l2a else None,
               "ejecucion__reintentar_pendientes": False if a.no_reintentar else None,
               "ejecucion__refrescar_provisional": False if a.no_refrescar else None,
               "salida__verificar_ultralytics": False if a.sin_verificar_ultralytics else None}
    cfg = cargar_configuracion(a.config, **cambios)
    if a.regresion_piloto is not None:
        from .piloto import imprimir_regresion, regresion_piloto
        inf = regresion_piloto(a.regresion_piloto or None, cfg)
        imprimir_regresion(inf)
        return 0 if inf["ok"] else 1
    try:
        res = ejecutar(cfg, sin_red=a.sin_red)
    except ErrorAutenticacion as exc:
        print(f"Earth Engine no disponible: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: res[k] for k in ("version_dataset", "yaml", "estado_producto", "estados_ejecucion")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
