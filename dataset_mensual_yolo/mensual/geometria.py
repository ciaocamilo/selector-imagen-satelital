"""AOI, malla común, núcleo de cumbres y sectores (adaptado de la sección 3 del selector)."""
import json
import math
import re
from dataclasses import dataclass

import shapely
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from shapely.geometry import box
from shapely.ops import transform as transformar_geometria

from .utilidades import huella


@dataclass(frozen=True)
class Malla:
    """Cuadrícula única del AOI: misma afín, filas y columnas para todas las fechas y productos."""
    crs: str
    transform: tuple  # (a, b, c, d, e, f): x = c + a·col, y = f + e·fila
    ancho: int
    alto: int

    @property
    def afin(self):
        return Affine(*self.transform)

    @property
    def limites(self):
        a, _, x0, _, e, y0 = self.transform
        return (x0, y0 + e * self.alto, x0 + a * self.ancho, y0)

    def como_dict(self):
        return {"crs": self.crs, "transform": list(self.transform), "ancho": self.ancho,
                "alto": self.alto, "resolucion_m": self.transform[0], "limites": list(self.limites)}

    @classmethod
    def desde_dict(cls, d):
        return cls(d["crs"], tuple(float(v) for v in d["transform"][:6]), int(d["ancho"]), int(d["alto"]))


def zona_utm(lon):
    return min(int((lon + 180) // 6) + 1, 60)


def elegir_crs(geom, crs_explicito=None):
    if crs_explicito:
        crs = CRS.from_user_input(crs_explicito)
        unidad = crs.axis_info[0].unit_name if crs.axis_info else ""
        if not crs.is_projected or unidad not in ("metre", "meter"):
            raise ValueError(f"El CRS debe ser proyectado y en metros (recibido: {crs_explicito}).")
        epsg = crs.to_epsg()
        return f"EPSG:{epsg}" if epsg else crs.to_wkt()
    minx, _, maxx, _ = geom.bounds
    if zona_utm(minx) != zona_utm(maxx):
        raise ValueError("El AOI cruza zonas UTM: define un CRS métrico explícito.")
    c = geom.centroid
    return f"EPSG:{(32600 if c.y >= 0 else 32700) + zona_utm(c.x)}"


def calcular_malla(geom, crs, res):
    """Proyecta el AOI (bordes densificados) y ajusta hacia afuera a múltiplos de res."""
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    geom_proj = transformar_geometria(tr.transform, shapely.segmentize(geom, 0.0005))
    minx, miny, maxx, maxy = geom_proj.bounds
    x0, y0 = math.floor(minx / res) * res, math.floor(miny / res) * res
    x1, y1 = math.ceil(maxx / res) * res, math.ceil(maxy / res) * res
    malla = Malla(crs, (float(res), 0.0, float(x0), 0.0, -float(res), float(y1)),
                  int(round((x1 - x0) / res)), int(round((y1 - y0) / res)))
    return malla, geom_proj


def mascara_geom(malla, geom_proj):
    """True en los píxeles cuyo centro cae dentro de la geometría proyectada."""
    return geometry_mask([geom_proj], out_shape=(malla.alto, malla.ancho), transform=malla.afin, invert=True)


def geojson_wgs84(geom):
    return json.loads(shapely.to_geojson(geom))


def sectores_nucleo(caja, n):
    """Divide el núcleo en n franjas de igual extensión latitudinal, de norte a sur."""
    oeste, sur, este, norte = (float(v) for v in caja)
    paso = (norte - sur) / n
    nombres = ["norte", "centro", "sur"] if n == 3 else [f"s{k + 1}" for k in range(n)]
    return [(nombres[k], box(oeste, norte - (k + 1) * paso, este, norte - k * paso)) for k in range(n)]


@dataclass
class Contexto:
    nombre: str
    id_aoi: str
    geom: object
    geom_proj: object
    malla: Malla
    nucleo: object
    sectores: list           # [(nombre, geom_wgs84, geom_proj)]
    res_diag: int

    def mascaras(self):
        """{'aoi': bool[H,W], <sector>: bool[H,W]} sobre la malla de salida."""
        m = {"aoi": mascara_geom(self.malla, self.geom_proj)}
        for nombre, _, gp in self.sectores:
            m[nombre] = mascara_geom(self.malla, gp)
        return m

    def resumen(self):
        return {"id_aoi": self.id_aoi, "malla": self.malla.como_dict(),
                "aoi_wgs84": list(self.geom.bounds), "nucleo_wgs84": list(self.nucleo.bounds),
                "sectores": {n: list(g.bounds) for n, g, _ in self.sectores}}


def preparar_contexto(cfg):
    a, n = cfg["aoi"], cfg["nucleo"]
    geom = box(*[float(v) for v in a["caja_wgs84"]])
    crs = elegir_crs(geom, a.get("crs"))
    malla, geom_proj = calcular_malla(geom, crs, float(a["resolucion_m"]))
    nombre = re.sub(r"[^A-Za-z0-9_-]+", "_", str(a["nombre"])).strip("_") or "aoi"
    # Misma huella que el selector: AOI idéntico -> mismo id_aoi (santa_isabel_6db5e66f6c).
    id_aoi = f"{nombre}_{huella({'geom': shapely.to_wkt(geom, rounding_precision=7), 'malla': malla.como_dict()}, 10)}"
    nucleo = box(*[float(v) for v in n["caja_wgs84"]])
    if not geom.buffer(1e-12).contains(nucleo):
        raise ValueError("El núcleo de cumbres debe quedar dentro del AOI.")
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    sectores = [(nom, g, transformar_geometria(tr.transform, shapely.segmentize(g, 0.0005)))
                for nom, g in sectores_nucleo(n["caja_wgs84"], int(n["sectores"]))]
    ctx = Contexto(nombre, id_aoi, geom, geom_proj, malla, nucleo, sectores, int(cfg["preseleccion"]["resolucion_m"]))
    for nom, m in ctx.mascaras().items():
        if not m.any():
            raise ValueError(f"La región {nom} no contiene píxeles de la malla.")
    return ctx


def plan_bloques(malla, tam):
    return [(f, c, min(tam, malla.alto - f), min(tam, malla.ancho - c))
            for f in range(0, malla.alto, tam) for c in range(0, malla.ancho, tam)]


def porcentajes_regiones(calidad, soporte, mascaras):
    """Cobertura (observado) y fracción útil (calidad 2) por región, con el área total de la región como denominador."""
    observado = calidad >= 1
    util = (calidad == 2) & soporte
    salida = {}
    for nombre, m in mascaras.items():
        total = int(m.sum())
        salida[nombre] = {"total_px": total,
                          "cobertura_pct": round(100.0 * int((observado & m).sum()) / total, 4),
                          "util_pct": round(100.0 * int((util & m).sum()) / total, 4)}
    return salida


def caja_en_pixeles(malla, geom_proj):
    """Rectángulo envolvente de una geometría proyectada en coordenadas de borde de píxel (x=col, y=fila)."""
    inv = ~malla.afin
    minx, miny, maxx, maxy = geom_proj.bounds
    c0, f0 = inv * (minx, maxy)
    c1, f1 = inv * (maxx, miny)
    return float(c0), float(f0), float(c1), float(f1)
