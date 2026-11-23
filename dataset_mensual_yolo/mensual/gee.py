"""Acceso a Earth Engine: inventario mensual con sumas por región y descarga por bloques (secciones 2, 4 y 6 del selector).

Nunca abre autenticación interactiva dentro del bucle: si faltan credenciales lanza ErrorAutenticacion y el flujo
marca los meses pendientes como reintentables. Las URL temporales no se guardan ni se imprimen.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from rasterio import Affine
from rasterio.crs import CRS as CRSRasterio
from rasterio.io import MemoryFile

from .productos import BANDAS_REFL, ESCALA_REFL, NODATA, PERFILES, SCL_ADMITIDAS
from .utilidades import sin_url

PROPIEDADES_S2 = {"satelite": "SPACECRAFT_NAME", "tesela": "MGRS_TILE", "orbita": "SENSING_ORBIT_NUMBER",
                  "baseline": "PROCESSING_BASELINE", "nube_granulo_pct": "CLOUDY_PIXEL_PERCENTAGE"}
LIMITE_BYTES_PETICION = 32 * 2 ** 20
LIMITE_DIM_PETICION = 10000
BYTES_PRODUCTO = {("reflectancia", None): 5 * 4, ("qa", "L2A"): 1, ("qa", "L1C"): 4}


class ErrorRed(RuntimeError):
    """Error transitorio (red, cuota, servidor): el mes queda reintentable."""


class ErrorAutenticacion(RuntimeError):
    """Credenciales o permisos ausentes: se detiene el uso de red en esta ejecución."""


_CLAVES_AUT = ("authorize", "authenticate", "credentials", "invalid_grant", "refresherror", "unauthenticated",
               "token has been expired", "reauth", "permission", "caller does not have", "not registered",
               "has not been used", "is disabled", "forbidden")
_CLAVES_RED = ("connection", "timed out", "timeout", "name resolution", "getaddrinfo", "unreachable", "max retries",
               "sslerror", "proxyerror", "temporarily unavailable", "too many requests", "quota", "rate limit",
               "429", "500", "502", "503", "504", "internal error", "deadline", "computation timed out",
               "user memory limit", "service unavailable")


def clasificar_error(exc):
    texto = str(exc).lower()
    if any(k in texto for k in _CLAVES_AUT):
        return "autenticacion"
    if isinstance(exc, (requests.RequestException, ConnectionError, TimeoutError)) or \
            any(k in texto for k in _CLAVES_RED):
        return "red"
    return "otro"


def proyecto_de_credenciales():
    try:
        import ee
        return str(json.loads(Path(ee.oauth.get_credentials_path()).read_text()).get("project") or "").strip()
    except Exception:
        return ""


def resolver_proyecto(proyecto):
    """Prioridad: configuración > EE_PROJECT > proyecto de las credenciales locales."""
    for valor, origen in ((proyecto, "configuración"), (os.environ.get("EE_PROJECT"), "variable EE_PROJECT"),
                          (proyecto_de_credenciales(), "credenciales locales")):
        if str(valor or "").strip():
            return str(valor).strip(), origen
    return "", ""


class ProveedorGEE:
    """Interfaz usada por el flujo (los dobles de prueba implementan los mismos dos métodos):
    inventario(perfil, inicio, fin_consulta) -> [candidato] y obtener_bloque(perfil, cand, producto, malla, ventana)."""

    def __init__(self, cfg, ctx, avance=print):
        self.cfg, self.ctx, self.avance = cfg, ctx, avance
        g = cfg["gee"]
        self.timeout, self.intentos, self.filas = int(g["timeout_s"]), int(g["max_intentos"]), int(g["filas_pagina"])
        self.cs_min = float(cfg["fuentes"]["cs_cdf_min"])
        self.proyecto = None
        self._ee = None

    # --- conexión ---------------------------------------------------------------------------------------------
    def conectar(self):
        if self._ee is not None:
            return self._ee
        try:
            import ee
        except ImportError as exc:
            raise ErrorAutenticacion(f"earthengine-api no está instalado: {exc}") from None
        proyecto, origen = resolver_proyecto(self.cfg["gee"]["proyecto"])
        if not proyecto:
            raise ErrorAutenticacion("Falta el proyecto de Earth Engine: define gee.proyecto, EE_PROJECT o "
                                     "`earthengine set_project`. Autentica fuera del flujo con `earthengine authenticate`.")
        try:
            ee.Initialize(project=proyecto)
            ee.Number(1).getInfo()
        except Exception as exc:
            if clasificar_error(exc) == "red":
                raise ErrorRed(f"Sin conexión con Earth Engine: {sin_url(exc)[:300]}") from None
            raise ErrorAutenticacion(f"No se pudo inicializar Earth Engine (proyecto '{proyecto}', {origen}): "
                                     f"{sin_url(exc)[:300]}. Ejecuta `earthengine authenticate` fuera del flujo.") from None
        self._ee, self.proyecto = ee, proyecto
        self.avance(f"Earth Engine listo · proyecto '{proyecto}' ({origen}) · earthengine-api {ee.__version__}")
        return ee

    def _get_info(self, objeto, que):
        self.conectar()
        for intento in range(1, self.intentos + 1):
            try:
                return objeto.getInfo()
            except Exception as exc:
                tipo = clasificar_error(exc)
                if tipo == "autenticacion":
                    raise ErrorAutenticacion(sin_url(exc)[:300]) from None
                if tipo != "red":  # error determinista de cómputo: no se reintenta, el mes queda reintentable
                    raise ErrorRed(f"{que}: {type(exc).__name__}: {sin_url(exc)[:300]}") from None
                if intento == self.intentos:
                    raise ErrorRed(f"{que}: falló tras {self.intentos} intentos: {sin_url(exc)[:300]}") from None
                espera = min(60, 2 ** intento)
                self.avance(f"      reintento {intento}/{self.intentos - 1} de {que} en {espera} s · {sin_url(exc)[:120]}")
                time.sleep(espera)

    # --- inventario -------------------------------------------------------------------------------------------
    def _geom(self, g):
        return self._ee.Geometry.Rectangle(list(g.bounds), "EPSG:4326", False)

    def _transform_diag(self):
        t, r = self.ctx.malla.transform, float(self.ctx.res_diag)
        return [r, 0, t[2], 0, -r, t[5]]

    def _capas(self, img, perfil, cs):
        ee = self._ee
        if perfil == "L2A":
            datos = img.select(BANDAS_REFL + ["SCL"]).mask().reduce(ee.Reducer.min()).gt(0).unmask(0, False)
            admitida = img.select("SCL").remap(list(SCL_ADMITIDAS), [1] * len(SCL_ADMITIDAS), 0).unmask(0, False)
            return datos, datos.And(admitida)
        datos = img.select(BANDAS_REFL).mask().reduce(ee.Reducer.min()).gt(0).unmask(0, False)
        visible = cs.gte(self.cs_min).unmask(0, False)
        return datos, datos.And(visible)

    def _coleccion_cs(self):
        return self._ee.ImageCollection(PERFILES["L1C"]["coleccion_qa"])

    def inventario(self, perfil, inicio, fin_consulta):
        ee = self.conectar()
        ctx = self.ctx
        aoi = self._geom(ctx.geom)
        regiones = [("aoi", aoi)] + [(n, self._geom(g)) for n, g, _ in ctx.sectores]
        zonas = [ee.Image(0).paint(ee.FeatureCollection([ee.Feature(g)]), 1).rename(n) for n, g in regiones]
        col = (ee.ImageCollection(PERFILES[perfil]["coleccion"]).filterBounds(aoi)
               .filterDate(inicio.isoformat(), fin_consulta.isoformat()))
        cs_col = self._coleccion_cs()

        def anotar(img):
            if perfil == "L1C":
                c = cs_col.filter(ee.Filter.eq("system:index", img.get("system:index")))
                tiene = c.size().gt(0)
                cs = ee.Image(ee.Algorithms.If(tiene, ee.Image(c.first()).select("cs_cdf"),
                                               ee.Image.constant(-1).rename("cs_cdf").selfMask()))
            else:
                tiene, cs = ee.Number(1), None
            datos, util = self._capas(img, perfil, cs)
            area = ee.Image.pixelArea()
            bandas = []
            for (n, _), z in zip(regiones, zonas):
                az = area.multiply(z)
                bandas += [az.rename(f"{n}__total"), az.multiply(datos).rename(f"{n}__datos"),
                           az.multiply(util).rename(f"{n}__util")]
            sumas = ee.Image.cat(bandas).reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, crs=ctx.malla.crs,
                                                      crsTransform=self._transform_diag(), maxPixels=1e10)
            props = {"id": img.get("system:id"), "indice": img.get("system:index"),
                     "t": img.get("system:time_start"), "qa_disponible": tiene, "sumas": sumas}
            props.update({k: img.get(v) for k, v in PROPIEDADES_S2.items()})
            return ee.Feature(None, props)

        n = int(self._get_info(col.size(), f"inventario {perfil}"))
        if n > 2000:
            raise ErrorRed(f"La consulta mensual devuelve {n} gránulos: revisa el AOI.")
        fc = ee.FeatureCollection(col.sort("system:index").map(anotar))
        filas = []
        for desde in range(0, n, self.filas):
            pagina = self._get_info(fc.toList(self.filas, desde), f"inventario {perfil} filas {desde + 1}–")
            filas += [f["properties"] for f in pagina]
        return [self._normalizar(p) for p in filas]

    def _normalizar(self, p):
        sumas = {}
        for k, v in (p.get("sumas") or {}).items():
            region, campo = k.split("__")
            sumas.setdefault(region, {})[campo] = float(v or 0.0)
        cand = {"id": p["id"], "indice": p["indice"], "t": int(p["t"]), "qa_disponible": bool(p.get("qa_disponible")),
                "sumas": sumas}
        for k in PROPIEDADES_S2:
            v = p.get(k)
            cand[k] = int(v) if k == "orbita" and v is not None else v
        return cand

    # --- descarga ---------------------------------------------------------------------------------------------
    def _imagen_producto(self, perfil, cand, producto):
        ee = self._ee
        if producto == "reflectancia":
            r = ee.Image(cand["id"]).select(BANDAS_REFL).multiply(ESCALA_REFL).resample("bilinear")
            return r.updateMask(r.mask().gte(1)).unmask(NODATA, False).toFloat()
        if perfil == "L2A":
            return ee.Image(cand["id"]).select("SCL").unmask(0, False).toUint8()  # vecino más próximo
        c = self._coleccion_cs().filter(ee.Filter.eq("system:index", cand["indice"])).first()
        return ee.Image(c).select("cs_cdf").unmask(NODATA, False).toFloat()

    def obtener_bloque(self, perfil, cand, producto, malla, ventana):
        self.conectar()
        f0, c0, alto, ancho = ventana
        nb = 5 if producto == "reflectancia" else 1
        volumen = alto * ancho * BYTES_PRODUCTO[(producto, None if producto == "reflectancia" else perfil)]
        if volumen > LIMITE_BYTES_PETICION or max(alto, ancho) > LIMITE_DIM_PETICION:
            raise ValueError(f"Bloque {ventana} excede los límites de petición: reduce gee.tam_bloque.")
        a, _, x0, _, e, y0 = malla.transform
        t = [a, 0, x0 + c0 * a, 0, e, y0 + f0 * e]
        params = {"crs": malla.crs, "crs_transform": t, "dimensions": [ancho, alto], "format": "GEO_TIFF"}
        imagen = self._imagen_producto(perfil, cand, producto)
        url = None
        for intento in range(1, self.intentos + 1):
            try:
                if url is None:
                    url = imagen.getDownloadURL(params)
                r = requests.get(url, timeout=self.timeout)
                if r.status_code in (400, 401, 403, 404, 410):
                    url = None  # URL caducada o inválida: se renueva
                    raise ErrorRed(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code == 429 or r.status_code >= 500:
                    raise ErrorRed(f"HTTP {r.status_code}")
                r.raise_for_status()
                with MemoryFile(r.content) as mem, mem.open() as ds:
                    if ((ds.width, ds.height) != (ancho, alto) or ds.count != nb
                            or not ds.transform.almost_equals(Affine(*t), precision=1e-6)
                            or ds.crs != CRSRasterio.from_user_input(malla.crs)):
                        raise ValueError(f"El bloque {ventana} recibido no coincide con la malla solicitada.")
                    return np.asarray(ds.read())
            except ValueError:
                raise
            except Exception as exc:
                if not isinstance(exc, ErrorRed) and clasificar_error(exc) == "autenticacion":
                    raise ErrorAutenticacion(sin_url(exc)[:300]) from None
                if intento == self.intentos:
                    raise ErrorRed(f"{producto} {ventana}: falló tras {self.intentos} intentos: "
                                   f"{sin_url(exc)[:200]}") from None
                espera = min(60, 2 ** intento)
                self.avance(f"      reintento {intento}/{self.intentos - 1} en {espera} s · {sin_url(exc)[:120]}")
                time.sleep(espera)
