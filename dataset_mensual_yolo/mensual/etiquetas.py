"""Pseudoetiquetado automático NDSI sin clics (reglas versionadas) y formato YOLO Segmentation.

Reutiliza de la sección 9b del selector: validar_poligono, mascara_poligono, serializar_yolo, leer_txt_yolo y
verificar_ida_vuelta. La aceptación por clic se sustituye por reglas explícitas registradas por componente.
"""
from collections import Counter
from itertools import combinations
from pathlib import Path
import re
import uuid

import numpy as np
from PIL import Image, ImageDraw
from rasterio import Affine
from rasterio.features import geometry_mask, shapes as formas_raster
from scipy import ndimage as ndi
from shapely.geometry import LinearRing, Polygon as PoligonoShp, shape as forma_shp

from .productos import cargar_escena
from .utilidades import (borrar_arbol, escribir_json_atomico, escribir_texto_atomico, huella, leer_json,
                         mover_con_reintentos, sha256_archivo)

REGLAS_VERSION = "1.0"
ORIGEN = "automatico_ndsi"
TOLERANCIA_IDA_VUELTA_PX = 0.01
CRUZ = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
SCL_AGUA = 6
EXPORTABLES = ("positiva", "negativa")


def reglas_perfil(cfg, perfil, mascara_agua=None):
    r = dict(cfg["etiquetas"][perfil])
    r.update(version=REGLAS_VERSION, perfil=perfil, origen=ORIGEN, clase_id=int(cfg["clase"]["id"]),
             conectividad_componentes=4,
             evidencia_agua=("SCL=6 y B8<agua_b8_max" if perfil == "L2A" else
                             "máscara de lagunas verificadas y B8<agua_b8_max" if mascara_agua else
                             "sin evidencia de agua (los componentes oscuros quedan ambiguos)"),
             mascara_agua=mascara_agua)
    r["huella"] = huella(r, 12)
    return r


# --- Geometría en coordenadas de bordes de píxel (selector §9b) -----------------------------------------------
def validar_poligono(vertices, W, H):
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2:
        return ["formato de vértices inválido"]
    if not np.isfinite(v).all():
        return ["coordenadas no finitas"]
    problemas = []
    if (v[:, 0] < 0).any() or (v[:, 0] > W).any() or (v[:, 1] < 0).any() or (v[:, 1] > H).any():
        problemas.append(f"vértices fuera del lienzo [0,{W}]×[0,{H}]")
    if len({(float(x), float(y)) for x, y in v}) < 3:
        return problemas + ["menos de 3 vértices distintos"]
    if PoligonoShp(v).area <= 1e-9:
        problemas.append("área nula")
    elif not LinearRing(v).is_simple:
        problemas.append("el contorno se autointersecta")
    return problemas


def mascara_poligono(vertices, W, H):
    """Píxeles cuyo centro (c+0,5; f+0,5) cae dentro del polígono."""
    anillo = [[float(x), float(y)] for x, y in vertices]
    return geometry_mask([{"type": "Polygon", "coordinates": [anillo + [anillo[0]]]}], out_shape=(H, W),
                         transform=Affine.identity(), invert=True)


def serializar_yolo(poligonos, W, H, clase=0):
    """Una línea por instancia: clase y pares x/W y/H con 8 decimales, sin repetir el primer vértice."""
    return "".join(f"{clase} " + " ".join(f"{x / W + 0.0:.8f} {y / H + 0.0:.8f}" for x, y in v) + "\n"
                   for v in poligonos)


def leer_txt_yolo(ruta):
    """Lector independiente del serializador: [(clase, [(x, y), ...])] con valores normalizados."""
    instancias = []
    texto = Path(ruta).read_bytes().decode("utf-8")
    for n, linea in enumerate(texto.splitlines(), 1):
        campos = linea.split()
        if not campos:
            raise ValueError(f"línea {n} vacía")
        if any(not re.fullmatch(r"-?\d+\.\d{8}", c) for c in campos[1:]):
            raise ValueError(f"línea {n}: hay valores sin 8 decimales")
        valores = [float(c) for c in campos[1:]]
        if len(valores) % 2:
            raise ValueError(f"línea {n}: número impar de coordenadas")
        instancias.append((int(campos[0]), list(zip(valores[0::2], valores[1::2]))))
    return instancias


def verificar_ida_vuelta(ruta, poligonos, W, H, clase=0, tolerancia_px=TOLERANCIA_IDA_VUELTA_PX):
    instancias = leer_txt_yolo(ruta)
    if len(instancias) != len(poligonos):
        raise ValueError(f"el TXT tiene {len(instancias)} instancias y la geometría {len(poligonos)}")
    error_max = 0.0
    for i, ((c, pares), v) in enumerate(zip(instancias, poligonos), 1):
        if c != clase:
            raise ValueError(f"instancia {i}: clase {c} ≠ {clase}")
        if len(pares) != len(v):
            raise ValueError(f"instancia {i}: {len(pares)} vértices en el TXT y {len(v)} en la geometría")
        norm = np.asarray(pares, dtype=float)
        if (norm < 0).any() or (norm > 1).any():
            raise ValueError(f"instancia {i}: coordenadas fuera de [0, 1]")
        error = float(np.abs(norm * [W, H] - np.asarray(v, dtype=float)).max())
        if error > tolerancia_px:
            raise ValueError(f"instancia {i}: error {error:.4f} px > {tolerancia_px} px")
        error_max = max(error_max, error)
    return error_max


def vectorizar_componente(sub, f0, c0, hueco_relleno_max_px):
    """Contorno exterior en bordes de píxel; huecos pequeños se rellenan y los demás se informan como relevantes."""
    geoms = [forma_shp(g) for g, _ in formas_raster(sub.astype("uint8"), mask=sub, connectivity=4,
                                                    transform=Affine.translation(c0, f0))]
    if len(geoms) != 1:
        return None, {"error": f"{len(geoms)} polígonos para un componente"}
    p = geoms[0]
    huecos = [PoligonoShp(h).area for h in p.interiors]
    relevantes = [a for a in huecos if a >= hueco_relleno_max_px]
    exterior = PoligonoShp(p.exterior).simplify(0.0, preserve_topology=True)  # solo quita vértices colineales
    vertices = [[float(x), float(y)] for x, y in list(exterior.exterior.coords)[:-1]]
    return vertices, {"huecos_rellenados": len(huecos) - len(relevantes),
                      "huecos_rellenados_px": float(sum(a for a in huecos if a < hueco_relleno_max_px)),
                      "huecos_relevantes": len(relevantes), "huecos_relevantes_px": float(sum(relevantes))}


# --- Componentes y reglas ---------------------------------------------------------------------------------------
def analizar_componentes(esc, umbral, reglas, agua_aux=None, vectorizar=True):
    """Componentes 4-conexos de NDSI ≥ umbral en soporte común, con métricas y motivo de decisión."""
    ndsi, b8, soporte, calidad, qa = esc["ndsi"], esc["b8"], esc["soporte"], esc["calidad"], esc.get("qa")
    perfil = esc["perfil"]
    H, W = ndsi.shape
    with np.errstate(invalid="ignore"):
        binaria = soporte & (ndsi >= umbral)
    lab, n = ndi.label(binaria, structure=CRUZ)
    oculto = (calidad <= 1) | ~soporte
    anillo_estr = np.ones((3, 3), bool) if int(reglas.get("conectividad_anillo", 8)) == 8 else CRUZ
    if perfil == "L2A" and qa is not None:
        evidencia, fuente_agua = (qa == SCL_AGUA), "SCL=6"
    elif perfil == "L1C" and agua_aux is not None:
        evidencia, fuente_agua = agua_aux, "mascara_lagunas"
    else:
        evidencia, fuente_agua = None, None
    comps = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        f0, f1 = max(sl[0].start - 1, 0), min(sl[0].stop + 1, H)
        c0, c1 = max(sl[1].start - 1, 0), min(sl[1].stop + 1, W)
        sub = lab[f0:f1, c0:c1] == i
        npx = int(sub.sum())
        anillo = ndi.binary_dilation(sub, structure=anillo_estr) & ~sub
        contacto = int((anillo & oculto[f0:f1, c0:c1]).sum())
        borde = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == H or sl[1].stop == W
        util = float((calidad[f0:f1, c0:c1][sub] == 2).mean())
        b8c = b8[f0:f1, c0:c1][sub]
        b8_med = float(np.nanmedian(b8c)) if np.isfinite(b8c).any() else float("nan")
        frac_agua = None
        if evidencia is not None:
            with np.errstate(invalid="ignore"):
                frac_agua = float((evidencia[f0:f1, c0:c1][sub] & (b8c < reglas["agua_b8_max"])).mean())
        c = {"id": i, "pixeles": npx, "caja": [int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)],
             "ndsi_mediana": round(float(np.nanmedian(ndsi[f0:f1, c0:c1][sub])), 4),
             "fraccion_util": round(util, 4), "contacto_ocultacion_px": contacto, "toca_borde": bool(borde),
             "b8_mediana": round(b8_med, 4), "fraccion_agua": None if frac_agua is None else round(frac_agua, 4),
             "fuente_agua": fuente_agua, "motivos": [], "vertices": None}
        if frac_agua is not None and frac_agua >= reglas["agua_fraccion_min"]:
            c["decision"] = "agua"
            c["motivos"].append("agua_inequivoca")
            comps.append(c)
            continue
        m = c["motivos"]
        if npx < reglas["min_pixeles"]:
            m.append("bajo_minimo")
        if util < reglas["min_util_componente"]:
            m.append("util_insuficiente")
        if contacto:
            m.append("contacto_ocultacion")
        if borde:
            m.append("truncado_borde")
        if not np.isfinite(b8_med) or b8_med < reglas["oscuro_b8_max"]:
            m.append("oscuro_sin_corroboracion")
        if vectorizar:
            vertices, info = vectorizar_componente(sub, f0, c0, reglas["hueco_relleno_max_px"])
            c.update({k: v for k, v in info.items() if k != "error"})
            if vertices is None:
                m.append("geometria_invalida")
            else:
                problemas = validar_poligono(vertices, W, H)
                if problemas:
                    m.append("geometria_invalida")
                    c["problemas_geometria"] = problemas
                c["vertices"] = vertices
                c["area_px"] = float(PoligonoShp(vertices).area)
            if info.get("huecos_relevantes"):
                m.append("huecos_relevantes")
        c["decision"] = "ambiguo" if m else "aceptado"
        comps.append(c)
    return comps


def visibilidad_sectores(esc, mascaras):
    util = (esc["calidad"] == 2) & esc["soporte"]
    return {n: round(100.0 * int((util & m).sum()) / int(m.sum()), 4) for n, m in mascaras.items() if n != "aoi"}


def decidir_escena(esc, mascaras, reglas, agua_aux=None):
    """positiva | negativa | ambigua | negativa_incierta, con componentes, reglas disparadas y polígonos."""
    comps = analizar_componentes(esc, reglas["ndsi_min"], reglas, agua_aux)
    aceptados = [c for c in comps if c["decision"] == "aceptado"]
    ambiguos = [c for c in comps if c["decision"] == "ambiguo"]
    disparadas = Counter(m for c in comps for m in c["motivos"])
    sectores = visibilidad_sectores(esc, mascaras)
    resultado = {"componentes": comps, "aceptados": len(aceptados), "ambiguos": len(ambiguos),
                 "agua": sum(c["decision"] == "agua" for c in comps), "reglas_disparadas": dict(disparadas),
                 "visibilidad_sectores_pct": sectores, "barrido_negativa": None, "poligonos": []}
    if ambiguos and reglas["ambiguos_bloquean"]:
        resultado.update(decision="ambigua", motivo=f"{len(ambiguos)} componente(s) ambiguo(s): " + ", ".join(
            f"{k}={v}" for k, v in sorted(Counter(m for c in ambiguos for m in c["motivos"]).items())))
        return resultado
    if aceptados:
        geoms = [PoligonoShp(c["vertices"]) for c in aceptados]
        for a, b in combinations(geoms, 2):
            if a.intersection(b).area > 1e-6:
                resultado.update(decision="ambigua", motivo="solapamiento interior entre instancias")
                return resultado
        resultado.update(decision="positiva", poligonos=[c["vertices"] for c in aceptados],
                         motivo=f"{len(aceptados)} instancia(s) aceptadas" +
                         (f"; {len(ambiguos)} ambiguo(s) omitidos (ambiguos_bloquean=false)" if ambiguos else ""))
        return resultado
    if ambiguos:
        resultado.update(decision="ambigua", motivo="solo hay componentes ambiguos; no es una negativa")
        return resultado
    faltan = {n: v for n, v in sectores.items() if v < reglas["negativa_min_sector_pct"]}
    segundo = analizar_componentes(esc, reglas["ndsi_min_negativa"], reglas, agua_aux, vectorizar=False)
    plausibles = [c for c in segundo if c["decision"] != "agua" and c["pixeles"] >= reglas["min_pixeles"]]
    resultado["barrido_negativa"] = {"umbral": reglas["ndsi_min_negativa"], "componentes": len(segundo),
                                     "plausibles": len(plausibles)}
    if faltan or plausibles:
        motivos = []
        if faltan:
            motivos.append("visibilidad insuficiente en " + ", ".join(f"{n} {v:.1f} %" for n, v in faltan.items()))
        if plausibles:
            motivos.append(f"{len(plausibles)} candidato(s) plausible(s) con NDSI ≥ {reglas['ndsi_min_negativa']}")
        resultado.update(decision="negativa_incierta", motivo="; ".join(motivos))
    else:
        resultado.update(decision="negativa", motivo="escena observable sin parches (negativa automática)")
    return resultado


# --- Escritura de la etiqueta en caché --------------------------------------------------------------------------
def figura_evidencia(esc, resultado, ruta):
    rgb_ruta = Path(esc["carpeta"]) / esc["meta"]["rutas"]["rgb"]
    with Image.open(rgb_ruta) as im:
        im = im.convert("RGB")
    d = ImageDraw.Draw(im)
    colores = {"aceptado": (0, 255, 0), "ambiguo": (255, 0, 0), "agua": (0, 128, 255)}
    for c in resultado["componentes"]:
        if c.get("vertices"):
            d.polygon([tuple(v) for v in c["vertices"]], outline=colores[c["decision"]])
        else:
            x0, y0, x1, y1 = c["caja"]
            d.rectangle([x0, y0, x1, y1], outline=colores[c["decision"]])
    im.save(ruta, format="PNG")


def etiquetar_escena(carpeta_escena, mascaras, reglas, agua_aux=None):
    """Calcula (o reutiliza) la decisión de una escena para una huella de reglas; escribe TXT verificado."""
    carpeta_escena = Path(carpeta_escena)
    destino = carpeta_escena / "etiquetas" / reglas["huella"]
    previo = leer_json(destino / "decision.json")
    if previo and previo.get("reglas_huella") == reglas["huella"]:
        txt = destino / previo["txt"] if previo.get("txt") else None
        if txt is None or (txt.is_file() and sha256_archivo(txt) == previo["sha256_txt"]):
            return destino, previo
    esc = cargar_escena(carpeta_escena)
    res = decidir_escena(esc, mascaras, reglas, agua_aux)
    W, H, clave = esc["ancho"], esc["alto"], esc["clave"]
    tmp = destino.parent / f"_tmp_{reglas['huella']}_{uuid.uuid4().hex[:6]}"
    tmp.mkdir(parents=True)
    try:
        decision = {"clave": clave, "perfil": esc["perfil"], "decision": res["decision"], "motivo": res["motivo"],
                    "origen": ORIGEN, "revision_humana": False, "reglas_version": REGLAS_VERSION,
                    "reglas_huella": reglas["huella"], "reglas": reglas, "ancho": W, "alto": H,
                    "instancias": len(res["poligonos"]),
                    "area_pseudoetiquetada_px": float(sum(PoligonoShp(v).area for v in res["poligonos"])),
                    **{k: res[k] for k in ("aceptados", "ambiguos", "agua", "reglas_disparadas",
                                           "visibilidad_sectores_pct", "barrido_negativa", "componentes")},
                    "txt": None, "sha256_txt": None,
                    "nota": "La aprobación automática indica cumplimiento de reglas; no es verificación glaciológica."}
        decision["area_pseudoetiquetada_m2"] = decision["area_pseudoetiquetada_px"] * abs(
            esc["meta"]["malla"]["transform"][0] * esc["meta"]["malla"]["transform"][4])
        if res["decision"] in EXPORTABLES:
            ruta_txt = tmp / f"{clave}.txt"
            escribir_texto_atomico(ruta_txt, serializar_yolo(res["poligonos"], W, H, reglas["clase_id"]))
            decision["error_ida_vuelta_px"] = verificar_ida_vuelta(ruta_txt, res["poligonos"], W, H,
                                                                   reglas["clase_id"])
            decision.update(txt=f"{clave}.txt", sha256_txt=sha256_archivo(ruta_txt))
        figura_evidencia(esc, res, tmp / "evidencia.png")
        escribir_json_atomico(tmp / "decision.json", decision)
        borrar_arbol(destino)
        mover_con_reintentos(tmp, destino)
    except BaseException:
        borrar_arbol(tmp)
        raise
    return destino, decision
