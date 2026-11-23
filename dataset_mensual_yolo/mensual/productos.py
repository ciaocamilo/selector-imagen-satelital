"""Descarga por bloques a la malla común, productos locales (GeoTIFF + 2 PNG) y verificación (sección 6 del selector).

Diferencias con el selector: el nivel de procesamiento es un parámetro (perfil L2A o L1C) y se conserva la banda
de calidad original en `qa.tif` (SCL para L2A, cs_cdf de Cloud Score+ para L1C) para la regla de agua.
"""
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.crs import CRS as CRSRasterio
from rasterio.windows import Window

from .geometria import plan_bloques, porcentajes_regiones
from .utilidades import (borrar_arbol, escribir_json_atomico, huella, iso_utc, leer_json, mover_con_reintentos,
                         sha256_archivo)

RECETA_VERSION = "2.0"
BANDAS_REFL = ["B2", "B3", "B4", "B8", "B11"]
RES_NATIVA_M = {"B2": 10, "B3": 10, "B4": 10, "B8": 10, "B11": 20, "SCL": 20, "cs_cdf": 10}
ESCALA_REFL = 0.0001
NODATA = -9999.0
SCL_ADMITIDAS = (4, 5, 6, 11)
SCL_EXCLUIDAS = (0, 1, 2, 3, 7, 8, 9, 10)
SCL_AGUA = 6
RGB_MAX_REFL = 0.60

PERFILES = {
    "L2A": {"coleccion": "COPERNICUS/S2_SR_HARMONIZED", "nivel": "L2A reflectancia de superficie",
            "qa": "SCL", "metodo_qa": "SCL clases útiles 4,5,6,11", "qa_dtype": "uint8", "qa_nodata": None},
    "L1C": {"coleccion": "COPERNICUS/S2_HARMONIZED", "nivel": "L1C reflectancia TOA",
            "qa": "cs_cdf", "metodo_qa": "Cloud Score+ cs_cdf ≥ umbral (vinculado por system:index)",
            "qa_dtype": "float32", "qa_nodata": NODATA, "coleccion_qa": "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"},
}


def archivos_tif(perfil):
    p = PERFILES[perfil]
    return {"reflectancia.tif": (5, "float32", NODATA), "qa.tif": (1, p["qa_dtype"], p["qa_nodata"]),
            "indices.tif": (2, "float32", NODATA), "calidad.tif": (1, "uint8", None),
            "soporte_png.tif": (1, "uint8", None)}


def receta_productos(perfil, ctx, cs_cdf_min):
    p = PERFILES[perfil]
    calidad = ({"scl_admitidas": list(SCL_ADMITIDAS), "scl_excluidas": list(SCL_EXCLUIDAS)} if perfil == "L2A"
               else {"cs_cdf_min": float(cs_cdf_min), "qa_ausente": "sin observación (calidad 0)"})
    receta = {
        "version": RECETA_VERSION, "perfil": perfil, "coleccion": p["coleccion"], "producto_qa": p["qa"],
        "id_aoi": ctx.id_aoi, "malla": ctx.malla.como_dict(), "bandas_reflectancia": BANDAS_REFL,
        "escala_reflectancia": ESCALA_REFL,
        "remuestreo": {**{b: "bilinear" for b in BANDAS_REFL}, p["qa"]: "nearest"},
        "mascara_remuestreo": "píxel válido solo si la máscara remuestreada es 1 (no se interpolan rellenos)",
        "nodata": {"reflectancia": NODATA, "indices": NODATA, "calidad": 0, "soporte_png": 0, "qa": p["qa_nodata"]},
        "indices": {"NDSI": "(B3-B11)/(B3+B11)", "NDVI": "(B8-B4)/(B8+B4)", "denominador_cero": "NoData"},
        "calidad": {**calidad, "codigos": {"0": "exterior o sin observación completa", "1": "observado y no visible",
                                           "2": "observado y visible"}},
        "png": {"rgb": "R=B4,G=B3,B=B2; round(255*clip(refl/0.60,0,1))",
                "espectral": "R=NDSI,G=NDVI: round(255*(clip(i,-1,1)+1)/2); B=B4 escala RGB",
                "redondeo": "floor(x+0.5)", "fuera_de_soporte": [0, 0, 0]},
    }
    receta["huella"] = huella(receta, 12)
    return receta


def a_uint8_reflectancia(r):
    return np.floor(np.clip(np.asarray(r, dtype=np.float64) / RGB_MAX_REFL, 0, 1) * 255 + 0.5).astype(np.uint8)


def a_uint8_indice(i):
    return np.floor((np.clip(np.asarray(i, dtype=np.float64), -1, 1) + 1) / 2 * 255 + 0.5).astype(np.uint8)


def derivar_productos(refl, qa, dentro, perfil="L2A", cs_cdf_min=0.60):
    """Índices, calidad, soporte y PNG desde las bandas alineadas en float32 (antes de cuantizar)."""
    refl = np.asarray(refl, dtype=np.float32)
    valida = dentro & np.all(np.isfinite(refl) & (refl != NODATA), axis=0)
    b2, b3, b4, b8, b11 = refl.astype(np.float64)
    den_ndsi, den_ndvi = b3 + b11, b8 + b4
    ok_ndsi, ok_ndvi = valida & (den_ndsi != 0), valida & (den_ndvi != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndsi = np.where(ok_ndsi, (b3 - b11) / den_ndsi, NODATA).astype(np.float32)
        ndvi = np.where(ok_ndvi, (b8 - b4) / den_ndvi, NODATA).astype(np.float32)
    qa = np.asarray(qa)
    if perfil == "L2A":
        observado = valida & (qa != 0)
        visible = np.isin(qa, SCL_ADMITIDAS)
    else:
        qa_ok = np.isfinite(qa) & (qa != NODATA)
        observado = valida & qa_ok
        with np.errstate(invalid="ignore"):
            visible = qa_ok & (qa >= cs_cdf_min)
    calidad = np.where(observado, np.where(visible, 2, 1), 0).astype(np.uint8)
    soporte = ok_ndsi & ok_ndvi
    rf = refl.astype(np.float64)
    rgb = np.stack([a_uint8_reflectancia(np.where(soporte, rf[k], 0)) for k in (2, 1, 0)], axis=-1)
    esp = np.stack([a_uint8_indice(np.where(soporte, ndsi, 0)), a_uint8_indice(np.where(soporte, ndvi, 0)),
                    a_uint8_reflectancia(np.where(soporte, rf[2], 0))], axis=-1)
    rgb[~soporte] = 0
    esp[~soporte] = 0
    util = soporte & (calidad == 2)
    sat = [util & (rf[k] >= RGB_MAX_REFL) for k in (2, 1, 0)]
    conteos = {"dentro": int(dentro.sum()), "observado": int(observado.sum()), "util": int((calidad == 2).sum()),
               "soporte": int(soporte.sum()), "util_soporte": int(util.sum()),
               "sat_alguno": int((sat[0] | sat[1] | sat[2]).sum()),
               "sat_r": int(sat[0].sum()), "sat_g": int(sat[1].sum()), "sat_b": int(sat[2].sum())}
    return {"indices": np.stack([ndsi, ndvi]), "calidad": calidad, "soporte": soporte.astype(np.uint8),
            "rgb": rgb, "espectral": esp, "conteos": conteos}


def porcentajes_locales(c):
    pct = lambda a, b: round(100.0 * a / b, 3) if b else None  # noqa: E731
    return {"cobertura_pct": pct(c["observado"], c["dentro"]), "util_pct": pct(c["util"], c["dentro"]),
            "soporte_png_pct": pct(c["soporte"], c["dentro"]),
            "saturacion_rgb_pct": pct(c["sat_alguno"], c["util_soporte"]),
            "saturacion_por_canal_pct": {"R": pct(c["sat_r"], c["util_soporte"]),
                                         "G": pct(c["sat_g"], c["util_soporte"]),
                                         "B": pct(c["sat_b"], c["util_soporte"])}}


def perfil_tif(malla, count, dtype, nodata):
    return dict(driver="GTiff", width=malla.ancho, height=malla.alto, count=count, dtype=dtype, crs=malla.crs,
                transform=malla.afin, nodata=nodata, compress="deflate", tiled=True, blockxsize=256, blockysize=256)


def fecha_utc(t_ms):
    return datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc)


def clave_escena(cand, perfil):
    """Fecha + tesela + system:index (igual que el selector); las escenas L1C llevan sufijo _l1c."""
    tesela = cand.get("tesela") or "NA"
    indice = re.sub(rf"_T{tesela}$", "", cand["indice"])
    return f"{fecha_utc(cand['t']):%Y%m%d}_T{tesela}_{indice}" + ("_l1c" if perfil == "L1C" else "")


def generar_productos_locales(carpeta, malla, dentro, perfil, cs_cdf_min, nombre, tam_bloque=512):
    """Desde reflectancia.tif y qa.tif escribe indices.tif, calidad.tif, soporte_png.tif y los dos PNG."""
    carpeta = Path(carpeta)
    (carpeta / "rgb").mkdir(parents=True, exist_ok=True)
    (carpeta / "espectral").mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((malla.alto, malla.ancho, 3), np.uint8)
    esp = np.zeros_like(rgb)
    total = dict.fromkeys(["dentro", "observado", "util", "soporte", "util_soporte", "sat_alguno",
                           "sat_r", "sat_g", "sat_b"], 0)
    with rasterio.open(carpeta / "reflectancia.tif") as src_r, rasterio.open(carpeta / "qa.tif") as src_q, \
         rasterio.open(carpeta / "indices.tif", "w", **perfil_tif(malla, 2, "float32", NODATA)) as d_i, \
         rasterio.open(carpeta / "calidad.tif", "w", **perfil_tif(malla, 1, "uint8", None)) as d_c, \
         rasterio.open(carpeta / "soporte_png.tif", "w", **perfil_tif(malla, 1, "uint8", None)) as d_p:
        d_i.descriptions = ("NDSI", "NDVI")
        for f0, c0, h, w in plan_bloques(malla, tam_bloque):
            win = Window(c0, f0, w, h)
            p = derivar_productos(src_r.read(window=win), src_q.read(1, window=win), dentro[f0:f0 + h, c0:c0 + w],
                                  perfil, cs_cdf_min)
            d_i.write(p["indices"], window=win)
            d_c.write(p["calidad"], 1, window=win)
            d_p.write(p["soporte"], 1, window=win)
            rgb[f0:f0 + h, c0:c0 + w] = p["rgb"]
            esp[f0:f0 + h, c0:c0 + w] = p["espectral"]
            for k, v in p["conteos"].items():
                total[k] += v
    for var, arr in (("rgb", rgb), ("espectral", esp)):
        Image.fromarray(arr).save(carpeta / var / f"{nombre}.png", format="PNG")
    return total


def verificar_escena(carpeta, malla, perfil, n_muestras=6):
    """Archivos, huellas, malla, tamaños de PNG y valores de PNG recalculados desde los rásteres."""
    carpeta = Path(carpeta)
    meta = leer_json(carpeta / "metadata.json")
    if meta is None or meta.get("estado") != "completa":
        return False, ["metadata.json ausente o incompleto"]
    problemas = []
    for rel, inf in meta["archivos"].items():
        ruta = carpeta / rel
        if not ruta.is_file():
            problemas.append(f"falta {rel}")
        elif sha256_archivo(ruta) != inf["sha256"]:
            problemas.append(f"huella distinta en {rel}")
    if problemas:
        return False, problemas
    for nombre, (count, dtype, nodata) in archivos_tif(perfil).items():
        with rasterio.open(carpeta / nombre) as ds:
            if ds.crs != CRSRasterio.from_user_input(malla.crs):
                problemas.append(f"{nombre}: CRS distinto")
            if not ds.transform.almost_equals(malla.afin, precision=1e-6):
                problemas.append(f"{nombre}: afín distinta")
            if (ds.width, ds.height, ds.count, ds.dtypes[0]) != (malla.ancho, malla.alto, count, dtype):
                problemas.append(f"{nombre}: dimensiones o tipo distintos")
            if ds.nodata != nodata:
                problemas.append(f"{nombre}: NoData {ds.nodata} ≠ {nodata}")
    pngs = {}
    for var in ("rgb", "espectral"):
        with Image.open(carpeta / var / f"{meta['nombre_escena']}.png") as im:
            if im.size != (malla.ancho, malla.alto) or im.mode != "RGB":
                problemas.append(f"{var}.png: tamaño o modo distintos")
            pngs[var] = np.asarray(im.convert("RGB"))
    if problemas:
        return False, problemas
    with rasterio.open(carpeta / "soporte_png.tif") as ds:
        soporte = ds.read(1).astype(bool)
    for var, arr in pngs.items():
        if arr[~soporte].any():
            problemas.append(f"{var}.png tiene valores ≠ 0 fuera del soporte")
    filas, cols = np.nonzero(soporte)
    if len(filas):
        rng = np.random.default_rng(0)
        elegidos = sorted({0, len(filas) // 2, len(filas) - 1, *rng.integers(0, len(filas), n_muestras).tolist()})
        with rasterio.open(carpeta / "reflectancia.tif") as dr, rasterio.open(carpeta / "indices.tif") as di:
            for k in elegidos:
                f, c = int(filas[k]), int(cols[k])
                v = dr.read(window=Window(c, f, 1, 1))[:, 0, 0]
                ndsi, ndvi = di.read(window=Window(c, f, 1, 1))[:, 0, 0]
                e_rgb = a_uint8_reflectancia([v[2], v[1], v[0]])
                e_esp = np.array([a_uint8_indice(ndsi), a_uint8_indice(ndvi), a_uint8_reflectancia(v[2])])
                if not (np.array_equal(pngs["rgb"][f, c], e_rgb) and np.array_equal(pngs["espectral"][f, c], e_esp)):
                    problemas.append(f"píxel ({f}, {c}): PNG no coincide con los rásteres")
    return not problemas, problemas


def versiones_paquetes():
    import PIL
    import shapely
    v = {"python": platform.python_version(), "numpy": np.__version__, "rasterio": rasterio.__version__,
         "pillow": PIL.__version__, "shapely": shapely.__version__}
    try:
        import ee
        v["earthengine-api"] = ee.__version__
    except Exception:
        pass
    return v


class ErrorDescarga(RuntimeError):
    """Fallo técnico (red, cuota, URL) tras agotar reintentos: el candidato queda reintentable."""


def descargar_escena(cand, perfil, ctx, receta, raiz_cache, obtener_bloque, tam_bloque=512, cs_cdf_min=0.60,
                     avance=print):
    """Descarga (o reutiliza) una escena en cache_escenas/<aoi>/<perfil>/<clave>/<receta>/ con publicación atómica."""
    malla = ctx.malla
    clave = clave_escena(cand, perfil)
    destino = Path(raiz_cache) / ctx.id_aoi / perfil / clave / receta["huella"]
    if destino.is_dir():
        ok, problemas = verificar_escena(destino, malla, perfil)
        meta = leer_json(destino / "metadata.json") or {}
        if ok and meta.get("id_asset") == cand["id"]:
            return destino, "reutilizada"
        apartada = Path(raiz_cache) / "_descartadas" / f"{clave}_{receta['huella']}_{datetime.now():%Y%m%d%H%M%S}"
        mover_con_reintentos(destino, apartada)
        avance(f"      caché inválida ({(problemas or ['asset distinto'])[0]}); apartada y se descarga de nuevo")
    tmp = destino.parent / f"_tmp_{receta['huella']}"
    if tmp.exists():
        avance("      descarga interrumpida previa: se reinicia")
        borrar_arbol(tmp)
    tmp.mkdir(parents=True)
    mascaras = ctx.mascaras()
    dentro = mascaras["aoi"]
    p = PERFILES[perfil]
    bloques = plan_bloques(malla, tam_bloque)
    with rasterio.open(tmp / "reflectancia.tif", "w", **perfil_tif(malla, 5, "float32", NODATA)) as dst_r, \
         rasterio.open(tmp / "qa.tif", "w", **perfil_tif(malla, 1, p["qa_dtype"], p["qa_nodata"])) as dst_q:
        dst_r.descriptions = tuple(BANDAS_REFL)
        dst_q.descriptions = (p["qa"],)
        for producto, dst, nb in (("reflectancia", dst_r, 5), ("qa", dst_q, 1)):
            for k, (f0, c0, h, w) in enumerate(bloques, 1):
                arr = np.asarray(obtener_bloque(perfil, cand, producto, malla, (f0, c0, h, w)))
                if arr.shape != (nb, h, w):
                    raise ValueError(f"Forma inesperada {arr.shape} en {producto} {(f0, c0, h, w)}")
                fuera = ~dentro[f0:f0 + h, c0:c0 + w]
                if producto == "reflectancia" or perfil == "L1C":
                    arr = arr.astype(np.float32)
                    arr[~np.isfinite(arr)] = NODATA
                    arr[:, fuera] = NODATA
                else:
                    arr = arr.astype(np.uint8)
                    arr[:, fuera] = 0
                dst.write(arr, window=Window(c0, f0, w, h))
            avance(f"      {producto}: {len(bloques)} bloque(s)")
    conteos = generar_productos_locales(tmp, malla, dentro, perfil, cs_cdf_min, clave, tam_bloque)
    with rasterio.open(tmp / "calidad.tif") as dc, rasterio.open(tmp / "soporte_png.tif") as ds:
        locales = porcentajes_regiones(dc.read(1), ds.read(1).astype(bool), mascaras)
    meta = construir_metadata(cand, perfil, ctx, receta, clave, conteos, locales)
    meta["archivos"] = {rel: {"sha256": sha256_archivo(tmp / rel), "bytes": (tmp / rel).stat().st_size}
                        for rel in ["reflectancia.tif", "qa.tif", "indices.tif", "calidad.tif", "soporte_png.tif",
                                    f"rgb/{clave}.png", f"espectral/{clave}.png"]}
    escribir_json_atomico(tmp / "metadata.json", meta)
    ok, problemas = verificar_escena(tmp, malla, perfil)
    if not ok:
        raise RuntimeError(f"Verificación fallida antes de publicar {clave}: {'; '.join(problemas)}")
    mover_con_reintentos(tmp, destino)
    return destino, "descargada"


def construir_metadata(cand, perfil, ctx, receta, clave, conteos, locales):
    adq = fecha_utc(cand["t"])
    p = PERFILES[perfil]
    return {
        "estado": "completa", "nombre_escena": clave, "clave": clave, "perfil": perfil, "nivel": p["nivel"],
        "id_asset": cand["id"], "system_index": cand["indice"], "coleccion": p["coleccion"],
        "metodo_qa": p["metodo_qa"], "coleccion_qa": p.get("coleccion_qa", p["coleccion"]),
        "satelite": cand.get("satelite"), "tesela": cand.get("tesela"), "orbita_relativa": cand.get("orbita"),
        "baseline": cand.get("baseline"), "nube_granulo_pct": cand.get("nube_granulo_pct"),
        "adquisicion_utc": iso_utc(adq), "grupo_temporal": f"{ctx.id_aoi}_{adq:%Y%m%d}",
        "aoi": ctx.resumen(), "malla": ctx.malla.como_dict(),
        "bandas": {"reflectancia.tif": BANDAS_REFL, "qa.tif": [p["qa"]], "indices.tif": ["NDSI", "NDVI"],
                   "calidad.tif": ["calidad"], "soporte_png.tif": ["soporte"]},
        "orden_canales": {"rgb": ["R=B4", "G=B3", "B=B2"], "espectral": ["R=NDSI", "G=NDVI", "B=B4"]},
        "resolucion_nativa_m": RES_NATIVA_M, "remuestreo": receta["remuestreo"],
        "porcentajes": {"inventario_20m": cand.get("metricas"), "local_10m": porcentajes_locales(conteos),
                        "regiones_local_10m": locales},
        "receta": {"version": RECETA_VERSION, "huella": receta["huella"], "detalle": receta},
        "versiones_paquetes": versiones_paquetes(),
        "rutas": {"reflectancia": "reflectancia.tif", "qa": "qa.tif", "indices": "indices.tif",
                  "calidad": "calidad.tif", "soporte_png": "soporte_png.tif", "rgb": f"rgb/{clave}.png",
                  "espectral": f"espectral/{clave}.png"},
        "creado_utc": iso_utc(datetime.now(timezone.utc)),
        "atribucion": f"Contiene datos modificados de Copernicus Sentinel-2 ({adq.year}) procesados mediante "
                      f"Google Earth Engine ({p['coleccion']}).",
    }


def cargar_escena(carpeta):
    """Arrays para el etiquetado: NDSI (NaN sin dato), B8, soporte, calidad, QA original (o None) y metadata."""
    carpeta = Path(carpeta)
    meta = leer_json(carpeta / "metadata.json")
    if meta is None:
        raise FileNotFoundError(f"{carpeta}: falta metadata.json")
    with rasterio.open(carpeta / "indices.tif") as ds:
        ind = ds.read()
        nd = ds.nodata
    ndsi = np.where((ind[0] == nd) | ~np.isfinite(ind[0]), np.nan, ind[0]).astype(np.float32)
    with rasterio.open(carpeta / "reflectancia.tif") as ds:
        b8 = ds.read(4)
        b8 = np.where(b8 == ds.nodata, np.nan, b8).astype(np.float32)
    with rasterio.open(carpeta / "calidad.tif") as ds:
        calidad = ds.read(1)
    with rasterio.open(carpeta / "soporte_png.tif") as ds:
        soporte = ds.read(1) == 1
    qa = None
    if (carpeta / "qa.tif").is_file():
        with rasterio.open(carpeta / "qa.tif") as ds:
            qa = ds.read(1)
    perfil = meta.get("perfil") or ("L2A" if "S2_SR" in str(meta.get("coleccion")) else "L1C")
    return {"carpeta": carpeta, "meta": meta, "perfil": perfil, "clave": meta.get("clave"), "ndsi": ndsi, "b8": b8,
            "calidad": calidad, "soporte": soporte, "qa": qa, "ancho": int(meta["malla"]["ancho"]),
            "alto": int(meta["malla"]["alto"])}


def a_json(obj):
    return json.loads(json.dumps(obj, default=str))
