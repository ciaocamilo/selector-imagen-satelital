"""Regresión sin red con la escena piloto del selector y lectura de su revisión humana publicada.

La escena piloto no guarda SCL: se reconstruye una SCL equivalente desde calidad.tif (2→4, 1→8, 0→0), que basta
para reproducir exactamente índices, calidad, soporte y los dos PNG. Por la misma razón el agua no puede
corroborarse en el piloto y los componentes oscuros quedan ambiguos.
"""
import copy
import io
import hashlib
import re
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from shapely.geometry import shape

from .etiquetas import decidir_escena, leer_txt_yolo, mascara_poligono, reglas_perfil
from .geometria import Malla, preparar_contexto
from .productos import cargar_escena, derivar_productos
from .utilidades import leer_json, sha256_archivo

RAIZ_SELECTOR = Path(__file__).resolve().parents[2]


def _escenas_completas(base):
    for c in sorted(p for p in base.glob("*") if p.is_dir()):
        meta = leer_json(c / "metadata.json")
        if meta and meta.get("estado") == "completa":
            yield c, meta


def localizar_piloto(ctx, raiz_selector=None):
    """Primera escena completa del selector para el mismo id_aoi; si el AOI del dataset es otro, la primera escena
    del selector cuya malla (mismo CRS y resolución) contiene la malla configurada."""
    datos = Path(raiz_selector or RAIZ_SELECTOR) / "datos_sentinel2"
    for c, _ in _escenas_completas(datos / ctx.id_aoi):
        return c
    x0, y1 = ctx.malla.transform[2], ctx.malla.transform[5]
    x1, y0 = x0 + ctx.malla.ancho * ctx.malla.transform[0], y1 + ctx.malla.alto * ctx.malla.transform[4]
    for base in sorted(p for p in datos.glob("*") if p.is_dir()):
        for c, meta in _escenas_completas(base):
            m = Malla.desde_dict(meta["malla"])
            mx0, my1 = m.transform[2], m.transform[5]
            mx1, my0 = mx0 + m.ancho * m.transform[0], my1 + m.alto * m.transform[4]
            if (m.crs == ctx.malla.crs and m.transform[0] == ctx.malla.transform[0]
                    and mx0 <= x0 and x1 <= mx1 and my0 <= y0 and y1 <= my1):
                return c
    return None


def configuracion_piloto(cfg, carpeta):
    """Copia de `cfg` con el AOI con el que el selector generó la escena (umbrales y núcleo se conservan)."""
    meta = leer_json(Path(carpeta) / "metadata.json")
    datos = copy.deepcopy(dict(cfg))
    datos["aoi"]["caja_wgs84"] = [round(v, 7) for v in shape(meta["aoi"]["geojson_wgs84"]).bounds]
    datos["aoi"]["nombre"] = meta["aoi"].get("nombre", datos["aoi"]["nombre"])
    return type(cfg)(datos, cfg.base, cfg.origen)


def revision_publicada(carpeta):
    """Última revisión publicada: solo carpetas numéricas de 4 dígitos (ignora _tmp_*, borrador y zip)."""
    rev = Path(carpeta) / "anotaciones" / "revisiones"
    if not rev.is_dir():
        return None
    numericas = sorted((p for p in rev.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}", p.name)),
                       key=lambda p: int(p.name))
    ignoradas = sorted(p.name for p in rev.iterdir() if not re.fullmatch(r"\d{4}", p.name))
    for r in reversed(numericas):
        a = leer_json(r / "anotacion.json")
        if a and a.get("estado") == "aprobada":
            txt = r / a["txt"]
            if sha256_archivo(txt) != a["sha256_txt"]:
                raise ValueError(f"{txt}: la huella del TXT no coincide con anotacion.json")
            return {"revision": r.name, "ruta": r, "anotacion": a, "txt": txt, "instancias_txt": leer_txt_yolo(txt),
                    "ignoradas": ignoradas}
    return None


def _sha_png(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return hashlib.sha256(b.getvalue()).hexdigest()


def regresion_piloto(carpeta=None, cfg=None):
    """Compara malla, huellas, productos derivados y conteos; aplica las reglas de etiquetado y explica cada parche.
    Se evalúa con el AOI del selector que generó la escena (puede diferir del AOI del dataset)."""
    from .configuracion import cargar_configuracion
    cfg = cfg or cargar_configuracion()
    carpeta = Path(carpeta) if carpeta else localizar_piloto(preparar_contexto(cfg))
    if carpeta is None:
        raise FileNotFoundError("No se encontró una escena piloto del selector que cubra este AOI.")
    cfg = configuracion_piloto(cfg, carpeta)
    ctx = preparar_contexto(cfg)
    meta = leer_json(carpeta / "metadata.json")
    inf = {"carpeta": str(carpeta), "clave": meta["clave"], "problemas": [], "id_aoi": ctx.id_aoi}
    p = inf["problemas"]
    malla = Malla.desde_dict(meta["malla"])
    inf["malla"] = f"{malla.ancho}×{malla.alto} {malla.crs}"
    if malla != ctx.malla:
        p.append(f"malla del piloto {malla} ≠ malla configurada {ctx.malla}")
    if meta["aoi"].get("id_aoi", ctx.id_aoi) != ctx.id_aoi:
        p.append("id_aoi distinto")
    for rel, a in meta["archivos"].items():
        if sha256_archivo(carpeta / rel) != a["sha256"]:
            p.append(f"huella distinta en {rel}")
    with rasterio.open(carpeta / "reflectancia.tif") as ds:
        refl = ds.read()
    with rasterio.open(carpeta / "calidad.tif") as ds:
        calidad = ds.read(1)
    with rasterio.open(carpeta / "indices.tif") as ds:
        indices = ds.read()
    with rasterio.open(carpeta / "soporte_png.tif") as ds:
        soporte = ds.read(1)
    pseudo_scl = np.select([calidad == 2, calidad == 1], [4, 8], 0).astype(np.uint8)
    d = derivar_productos(refl, pseudo_scl, ctx.mascaras()["aoi"], "L2A")
    comparaciones = {"indices.tif": np.array_equal(d["indices"], indices),
                     "calidad.tif": np.array_equal(d["calidad"], calidad),
                     "soporte_png.tif": np.array_equal(d["soporte"], soporte),
                     "rgb.png": _sha_png(d["rgb"]) == meta["archivos"][meta["rutas"]["rgb"]]["sha256"],
                     "espectral.png": _sha_png(d["espectral"]) == meta["archivos"][meta["rutas"]["espectral"]]["sha256"]}
    inf["productos_reproducidos"] = comparaciones
    p += [f"{k} no se reproduce" for k, ok in comparaciones.items() if not ok]
    inf["conteos_calidad"] = {int(k): int(v) for k, v in zip(*np.unique(calidad, return_counts=True))}
    inf["soporte_px"] = int(soporte.sum())

    rev = revision_publicada(carpeta)
    inf["revision"] = None if rev is None else {"revision": rev["revision"], "tipo": rev["anotacion"].get("tipo"),
                                                "poligonos": len(rev["anotacion"]["poligonos"]),
                                                "descartes": len(rev["anotacion"].get("descartes", [])),
                                                "instancias_txt": len(rev["instancias_txt"]),
                                                "ignoradas": rev["ignoradas"]}
    esc = cargar_escena(carpeta)
    esc["perfil"] = "L2A"
    reglas = reglas_perfil(cfg, "L2A")
    res = decidir_escena(esc, ctx.mascaras(), reglas)
    inf["decision_automatica"] = {"decision": res["decision"], "motivo": res["motivo"],
                                  "reglas_disparadas": res["reglas_disparadas"],
                                  "visibilidad_sectores_pct": res["visibilidad_sectores_pct"]}
    W, H = esc["ancho"], esc["alto"]
    lab = np.zeros((H, W), np.int32)
    for c in res["componentes"]:
        if c.get("vertices"):
            lab[mascara_poligono(c["vertices"], W, H) & (lab == 0)] = c["id"]
    por_id = {c["id"]: c for c in res["componentes"]}
    parches = []
    if rev:
        humanos = [("aceptado_humano", q["vertices"]) for q in rev["anotacion"]["poligonos"]] + \
                  [("descartado_humano", q["vertices"]) for q in rev["anotacion"].get("descartes", [])]
        for k, (tipo, v) in enumerate(humanos, 1):
            m = mascara_poligono(v, W, H)
            ids, n = np.unique(lab[m], return_counts=True)
            coinc = [(int(i), int(c)) for i, c in zip(ids, n) if i]
            principal = por_id[max(coinc, key=lambda t: t[1])[0]] if coinc else None
            parches.append({"parche": k, "tipo": tipo, "pixeles": int(m.sum()),
                            "componente": principal and principal["id"],
                            "decision_automatica": principal["decision"] if principal else "sin componente",
                            "motivos": principal["motivos"] if principal else [],
                            "pixeles_componente": principal and principal["pixeles"]})
    inf["parches"] = parches
    inf["componentes"] = [{k: c[k] for k in ("id", "pixeles", "caja", "ndsi_mediana", "fraccion_util",
                                             "contacto_ocultacion_px", "b8_mediana", "decision", "motivos")}
                          for c in res["componentes"]]
    inf["ok"] = not p
    return inf


def imprimir_regresion(inf, avance=print):
    avance(f"Escena piloto: {inf['clave']} · malla {inf['malla']} · AOI del selector {inf['id_aoi']}")
    avance("  Productos reproducidos: " + ", ".join(f"{k} {'ok' if v else 'FALLA'}"
                                                   for k, v in inf["productos_reproducidos"].items()))
    avance(f"  Calidad {inf['conteos_calidad']} · soporte {inf['soporte_px']} px")
    if inf["revision"]:
        r = inf["revision"]
        avance(f"  Revisión humana {r['revision']}: {r['tipo']}, {r['poligonos']} polígono(s), {r['descartes']} "
               f"descarte(s); ignoradas: {', '.join(r['ignoradas']) or '—'}")
    d = inf["decision_automatica"]
    avance(f"  Decisión automática: {d['decision']} · {d['motivo']}")
    for q in inf["parches"]:
        avance(f"   parche {q['parche']:>2} ({q['tipo']}, {q['pixeles']} px) → componente {q['componente']} "
               f"{q['decision_automatica']} {'/'.join(q['motivos']) or ''}")
    avance("  Regresión " + ("correcta" if inf["ok"] else "con problemas: " + "; ".join(inf["problemas"])))
