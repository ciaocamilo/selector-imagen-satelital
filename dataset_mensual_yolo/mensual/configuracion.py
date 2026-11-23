"""Configuración TOML con valores por defecto (Santa Isabel, 2016–2026) y validación."""
import copy
import json
import tomllib
from datetime import date
from pathlib import Path

PERFILES = ("L2A", "L1C")

_ETIQUETAS_PERFIL = {
    "ndsi_min": 0.40,                 # NDSI ≥ umbral en soporte común
    "ndsi_min_negativa": 0.30,        # segundo barrido para confirmar una negativa
    "min_pixeles": 5,                 # componentes menores pasan a ambiguos
    "min_util_componente": 0.80,      # fracción de píxeles útiles (calidad 2) dentro del componente
    "hueco_relleno_max_px": 4,        # huecos con área < este valor se rellenan (y se registra); el resto son relevantes
    "agua_fraccion_min": 0.90,        # agua inequívoca: ≥ 90 % del componente con evidencia de agua y B8 < agua_b8_max
    "agua_b8_max": 0.10,
    "oscuro_b8_max": 0.10,            # mediana B8 < este valor sin corroboración de agua -> ambiguo
    "negativa_min_sector_pct": 99.0,  # visibilidad mínima en cada sector del núcleo para una negativa
    "conectividad_anillo": 8,         # anillo exterior de un píxel: 8 = incluye diagonales
    "ambiguos_bloquean": True,        # un componente ambiguo impide exportar la escena completa
}

DEFECTO = {
    "periodo": {"anio_inicio": 2016, "anio_fin": 2026, "corte_utc": "", "meses": []},
    "aoi": {"nombre": "santa_isabel", "caja_wgs84": [-75.397, 4.773, -75.348, 4.827], "crs": "EPSG:32618",
            "resolucion_m": 10},
    "nucleo": {"caja_wgs84": [-75.385, 4.775, -75.348, 4.827], "sectores": 3},
    "clase": {"id": 0, "nombre": "nieve_hielo_visible"},
    "fuentes": {"modo": "l2a_preferente", "inicio_l2a": "2017-03-28", "cs_cdf_min": 0.60},
    "preseleccion": {"resolucion_m": 20, "min_cobertura_aoi_pct": 95.0, "min_util_aoi_pct": 70.0,
                     "min_util_sector_pct": 90.0, "max_descargas_mes": 0},
    "etiquetas": {"mascara_agua_geojson": "", "L2A": dict(_ETIQUETAS_PERFIL), "L1C": dict(_ETIQUETAS_PERFIL)},
    "particiones": {"frac_train": 0.70, "frac_val": 0.15, "corte_val": "", "corte_test": "",
                    "conservar_cortes_publicados": True},
    "salida": {"directorio": "salida", "enlaces": "fisico", "verificar_ultralytics": True},
    "gee": {"proyecto": "", "tam_bloque": 512, "timeout_s": 180, "max_intentos": 4, "filas_pagina": 50},
    "ejecucion": {"reintentar_pendientes": True, "refrescar_provisional": True},
}


class ErrorConfiguracion(ValueError):
    pass


def _fusionar(base, extra, ruta=""):
    for k, v in (extra or {}).items():
        if k not in base:
            raise ErrorConfiguracion(f"Clave desconocida en la configuración: {ruta}{k}")
        if isinstance(base[k], dict) and isinstance(v, dict):
            _fusionar(base[k], v, f"{ruta}{k}.")
        else:
            base[k] = v
    return base


def _validar(c):
    p = c["periodo"]
    if not (2015 <= int(p["anio_inicio"]) <= int(p["anio_fin"]) <= 2100):
        raise ErrorConfiguracion("periodo: se requiere 2015 ≤ anio_inicio ≤ anio_fin.")
    for caja, nombre in ((c["aoi"]["caja_wgs84"], "aoi"), (c["nucleo"]["caja_wgs84"], "nucleo")):
        if len(caja) != 4 or not (caja[0] < caja[2] and caja[1] < caja[3]):
            raise ErrorConfiguracion(f"{nombre}.caja_wgs84 debe ser [oeste, sur, este, norte] con oeste<este y sur<norte.")
    if c["fuentes"]["modo"] not in ("l2a_preferente", "solo_l2a"):
        raise ErrorConfiguracion("fuentes.modo debe ser 'l2a_preferente' o 'solo_l2a'.")
    date.fromisoformat(c["fuentes"]["inicio_l2a"])
    pt = c["particiones"]
    if not (0 < pt["frac_train"] < 1 and 0 <= pt["frac_val"] < 1 and pt["frac_train"] + pt["frac_val"] <= 1):
        raise ErrorConfiguracion("particiones: fracciones inválidas.")
    if c["salida"]["enlaces"] not in ("fisico", "copia"):
        raise ErrorConfiguracion("salida.enlaces debe ser 'fisico' o 'copia'.")
    if int(c["nucleo"]["sectores"]) < 1:
        raise ErrorConfiguracion("nucleo.sectores debe ser ≥ 1.")
    for perfil in PERFILES:
        e = c["etiquetas"][perfil]
        if not -1 <= e["ndsi_min_negativa"] <= e["ndsi_min"] <= 1:
            raise ErrorConfiguracion(f"etiquetas.{perfil}: se requiere ndsi_min_negativa ≤ ndsi_min.")


class Configuracion(dict):
    """Diccionario anidado validado; `base` es el directorio respecto al que se resuelven las rutas relativas."""

    def __init__(self, datos, base, origen=None):
        super().__init__(datos)
        self.base = Path(base).resolve()
        self.origen = origen

    def ruta(self, valor):
        p = Path(valor)
        return p if p.is_absolute() else (self.base / p)

    @property
    def directorio_salida(self):
        return self.ruta(self["salida"]["directorio"])

    def efectiva(self):
        return json.loads(json.dumps(dict(self), default=str))


def cargar_configuracion(ruta=None, **cambios):
    """Lee el TOML (opcional) sobre los valores por defecto. `cambios` usa claves 'seccion__clave'."""
    datos = copy.deepcopy(DEFECTO)
    base, origen = Path.cwd(), None
    if ruta:
        ruta = Path(ruta)
        if not ruta.is_file():
            raise ErrorConfiguracion(f"No existe el archivo de configuración {ruta}")
        with open(ruta, "rb") as f:
            _fusionar(datos, tomllib.load(f))
        base, origen = ruta.resolve().parent, str(ruta.resolve())
    for clave, valor in cambios.items():
        if valor is None:
            continue
        partes = clave.split("__")
        d = datos
        for p in partes[:-1]:
            d = d[p]
        if partes[-1] not in d:
            raise ErrorConfiguracion(f"Cambio desconocido: {clave}")
        d[partes[-1]] = valor
    _validar(datos)
    return Configuracion(datos, base, origen)


def _toml_valor(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_valor(x) for x in v) + "]"
    return json.dumps(str(v), ensure_ascii=False)


def a_toml(datos, prefijo=""):
    """Serializador TOML mínimo (escalares, listas y tablas anidadas) para la copia de la configuración efectiva."""
    lineas, tablas = [], []
    for k, v in datos.items():
        if isinstance(v, dict):
            tablas.append((k, v))
        else:
            lineas.append(f"{k} = {_toml_valor(v)}")
    salida = "\n".join(lineas) + ("\n" if lineas else "")
    for k, v in tablas:
        nombre = f"{prefijo}{k}"
        salida += f"\n[{nombre}]\n" + a_toml(v, nombre + ".")
    return salida
