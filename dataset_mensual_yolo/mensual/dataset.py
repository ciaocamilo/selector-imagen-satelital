"""Particiones cronológicas y empaquetado de dos datasets YOLO Segmentation (rgb y espectral) con splits idénticos."""
import math
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import yaml
from PIL import Image

from .configuracion import a_toml
from .etiquetas import leer_txt_yolo
from .utilidades import (borrar_arbol, enlazar_o_copiar, escribir_csv_atomico, escribir_json_atomico,
                         escribir_texto_atomico, huella, iso_utc, ahora_utc, leer_json, mover_con_reintentos,
                         sha256_archivo)

VARIANTES = ("rgb", "espectral")
SPLITS = ("train", "val", "test")
FORMATO_DATASET = "1.0"
COLUMNAS_MANIFIESTO = ["mes", "split", "nombre", "perfil", "clave_escena", "id_asset", "adquisicion_utc",
                       "grupo_temporal", "decision", "instancias", "area_pseudoetiquetada_m2", "origen",
                       "revision_humana", "provisional", "seleccion_incompleta", "imagen_rgb", "imagen_espectral",
                       "etiqueta", "sha256_rgb", "sha256_espectral", "sha256_txt", "receta_huella", "reglas_huella"]


# --- Particiones ------------------------------------------------------------------------------------------------
def calcular_cortes(meses_plan, cfg, previos=None):
    """Cortes por mes calendario planificado: train < corte_val ≤ val < corte_test ≤ test."""
    pt = cfg["particiones"]
    n = len(meses_plan)
    n_train = math.floor(round(float(pt["frac_train"]) * n, 9))
    n_val = math.floor(round(float(pt["frac_val"]) * n, 9))
    calc = {"corte_val": meses_plan[n_train] if n_train < n else None,
            "corte_test": meses_plan[n_train + n_val] if n_train + n_val < n else None}
    origen = "fracciones"
    if pt.get("corte_val") or pt.get("corte_test"):
        calc = {"corte_val": pt.get("corte_val") or calc["corte_val"],
                "corte_test": pt.get("corte_test") or calc["corte_test"]}
        origen = "cortes_explicitos"
    elif pt.get("conservar_cortes_publicados") and previos and previos.get("meses_plan") == [meses_plan[0], meses_plan[-1], n]:
        calc = {"corte_val": previos["corte_val"], "corte_test": previos["corte_test"]}
        origen = "conservados_de_version_publicada"
    if calc["corte_val"] and calc["corte_test"] and calc["corte_test"] < calc["corte_val"]:
        raise ValueError("particiones: corte_test debe ser posterior a corte_val.")
    cortes = {**calc, "origen": origen, "meses_plan": [meses_plan[0], meses_plan[-1], n]}
    cortes["meses_por_split"] = dict(Counter(split_de_mes(m, cortes) for m in meses_plan))
    cambio = bool(previos) and (previos.get("corte_val"), previos.get("corte_test")) != (calc["corte_val"],
                                                                                           calc["corte_test"])
    cortes["cambio_respecto_a_publicada"] = cambio
    return cortes


def split_de_mes(mes, cortes):
    if cortes["corte_val"] is None or mes < cortes["corte_val"]:
        return "train"
    if cortes["corte_test"] is None or mes < cortes["corte_test"]:
        return "val"
    return "test"


def estado_producto(conteos):
    tr, va, te = (conteos[s] for s in SPLITS)
    if tr["imagenes"] == 0 or va["imagenes"] == 0 or tr["positivas"] == 0:
        return "no_entrenable"
    if te["imagenes"] == 0:
        return "entrenable_sin_prueba"
    return "entrenable"


# --- YAML -------------------------------------------------------------------------------------------------------
def texto_yaml(raiz_variante, nombres, variante, version):
    datos = {"path": Path(raiz_variante).resolve().as_posix(), "train": "images/train", "val": "images/val",
             "test": "images/test", "names": {int(k): v for k, v in nombres.items()}}
    cabecera = (f"# Dataset YOLO Segmentation · variante {variante} · versión {version}\n"
                f"# Pseudoetiquetas automáticas (NDSI); no es verificación glaciológica.\n"
                f"# Si mueves la carpeta: python -m mensual --regenerar-yaml <carpeta_version>\n")
    return cabecera + yaml.safe_dump(datos, sort_keys=False, allow_unicode=True)


def regenerar_yaml(carpeta_version):
    """Reescribe `path:` de ambos data.yaml con la ubicación actual de la carpeta de la versión."""
    carpeta_version = Path(carpeta_version).resolve()
    resumen = leer_json(carpeta_version / "resumen.json") or {}
    nombres = resumen.get("clases") or {0: "nieve_hielo_visible"}
    rutas = []
    for v in VARIANTES:
        if (carpeta_version / v).is_dir():
            rutas.append(escribir_texto_atomico(carpeta_version / v / "data.yaml",
                                                texto_yaml(carpeta_version / v, nombres, v, carpeta_version.name)))
    return rutas


# --- Verificaciones -----------------------------------------------------------------------------------------------
def verificar_paquete(raiz, filas, ancho, alto, clases_validas):
    """Pares completos, sin huérfanos ni duplicados, TXT idénticos entre variantes, fechas sin cruzar splits."""
    problemas = []
    raiz = Path(raiz)
    nombres = Counter(f["nombre"] for f in filas)
    problemas += [f"nombre duplicado: {n}" for n, k in nombres.items() if k > 1]
    grupos = defaultdict(set)
    for f in filas:
        grupos[f["grupo_temporal"]].add(f["split"])
    problemas += [f"la fecha {g} aparece en varios splits {sorted(s)}" for g, s in grupos.items() if len(s) > 1]
    esperados = {(f["split"], f["nombre"]) for f in filas}
    for v in VARIANTES:
        for s in SPLITS:
            imgs = {p.stem for p in (raiz / v / "images" / s).glob("*") if p.is_file()}
            txts = {p.stem for p in (raiz / v / "labels" / s).glob("*.txt")}
            esperado = {n for sp, n in esperados if sp == s}
            if imgs != esperado:
                problemas.append(f"{v}/{s}: imágenes distintas del manifiesto ({len(imgs ^ esperado)} diferencias)")
            if txts != imgs:
                problemas.append(f"{v}/{s}: huérfanos imagen/etiqueta ({sorted(txts ^ imgs)[:3]})")
    for f in filas:
        rutas_txt = [raiz / v / "labels" / f["split"] / f"{f['nombre']}.txt" for v in VARIANTES]
        if not all(r.is_file() for r in rutas_txt):
            continue
        huellas = {sha256_archivo(r) for r in rutas_txt}
        if huellas != {f["sha256_txt"]}:
            problemas.append(f"{f['nombre']}: el TXT difiere entre variantes o del origen")
        instancias = leer_txt_yolo(rutas_txt[0])
        if len(instancias) != int(f["instancias"]):
            problemas.append(f"{f['nombre']}: {len(instancias)} instancias en TXT y {f['instancias']} en el manifiesto")
        if any(c not in clases_validas for c, _ in instancias):
            problemas.append(f"{f['nombre']}: clase fuera de names")
        if (f["decision"] == "negativa") != (len(instancias) == 0):
            problemas.append(f"{f['nombre']}: decisión {f['decision']} incoherente con el TXT")
        for v in VARIANTES:
            with Image.open(raiz / v / "images" / f["split"] / f"{f['nombre']}.png") as im:
                if im.size != (ancho, alto) or im.mode != "RGB":
                    problemas.append(f"{v}/{f['nombre']}: tamaño o modo distintos")
    return problemas


@contextmanager
def _sin_descarga_de_fuentes():
    """check_det_dataset intenta descargar una fuente tipográfica; se evita para trabajar sin red."""
    import ultralytics.data.utils as u
    original = u.check_font
    u.check_font = lambda *a, **k: None
    try:
        yield
    finally:
        u.check_font = original


def verificar_ultralytics(yaml_ruta, conteos):
    """Carga cada split no vacío con el cargador de segmentación de Ultralytics y arma un lote."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG

    informe = {}
    with _sin_descarga_de_fuentes():
        data = check_det_dataset(str(yaml_ruta))
    cfg = get_cfg(DEFAULT_CFG, {"task": "segment", "imgsz": 640, "workers": 0, "cache": False, "plots": False})
    raiz = Path(data["path"])
    try:
        for s in SPLITS:
            if conteos[s]["imagenes"] == 0:
                informe[s] = "vacío"
                continue
            ds = build_yolo_dataset(cfg, data[s], 2, data, mode="val", rect=False, stride=32)
            n_inst = sum(len(lb.get("segments") or []) for lb in ds.labels)
            if len(ds) != conteos[s]["imagenes"] or n_inst != conteos[s]["instancias"]:
                raise ValueError(f"Ultralytics/{s}: {len(ds)} imágenes y {n_inst} instancias; esperadas "
                                 f"{conteos[s]['imagenes']} y {conteos[s]['instancias']}")
            lote = ds.collate_fn([ds[i] for i in range(min(2, len(ds)))])
            if "masks" not in lote:
                raise ValueError(f"Ultralytics/{s}: el lote no contiene máscaras de segmentación")
            informe[s] = {"imagenes": len(ds), "instancias": n_inst, "lote_img": list(lote["img"].shape),
                          "lote_masks": list(lote["masks"].shape)}
    finally:
        for c in (raiz / "labels").glob("*.cache"):
            c.unlink(missing_ok=True)
    return informe


# --- Empaquetado --------------------------------------------------------------------------------------------------
def _siguiente_version(raiz_datasets):
    nums = [int(p.name[1:]) for p in raiz_datasets.glob("v[0-9][0-9][0-9][0-9]") if p.is_dir()]
    return f"v{max(nums, default=0) + 1:04d}"


def filas_dataset(estados, cortes, raiz_salida):
    filas = []
    for mes, e in sorted(estados.items()):
        if e.get("estado") != "exportada":
            continue
        esc, et = e["escena"], e["etiqueta"]
        carpeta = Path(raiz_salida) / esc["carpeta"]
        carpeta_et = Path(raiz_salida) / et["carpeta"]
        meta = leer_json(carpeta / "metadata.json")
        nombre = f"{mes}_{esc['clave']}"
        filas.append({
            "mes": mes, "split": split_de_mes(mes, cortes), "nombre": nombre, "perfil": esc["perfil"],
            "clave_escena": esc["clave"], "id_asset": esc["id_asset"], "adquisicion_utc": esc["adquisicion_utc"],
            "grupo_temporal": meta["grupo_temporal"], "decision": et["decision"], "instancias": et["instancias"],
            "area_pseudoetiquetada_m2": et.get("area_pseudoetiquetada_m2"), "origen": et.get("origen"),
            "revision_humana": False, "provisional": bool(e.get("provisional")),
            "seleccion_incompleta": bool(e.get("seleccion_incompleta")),
            "_src_rgb": carpeta / meta["rutas"]["rgb"], "_src_espectral": carpeta / meta["rutas"]["espectral"],
            "_src_txt": carpeta_et / et["txt"],
            "sha256_rgb": meta["archivos"][meta["rutas"]["rgb"]]["sha256"],
            "sha256_espectral": meta["archivos"][meta["rutas"]["espectral"]]["sha256"],
            "sha256_txt": et["sha256_txt"], "receta_huella": meta["receta"]["huella"],
            "reglas_huella": et["reglas_huella"], "_ancho": meta["malla"]["ancho"], "_alto": meta["malla"]["alto"]})
    return filas


def empaquetar(cfg, ctx, meses_plan, estados, raiz_salida, id_ejecucion, avance=print):
    """Construye (si cambió el contenido) una nueva versión vNNNN con rgb/ y espectral/; publicación atómica."""
    raiz_salida = Path(raiz_salida)
    raiz_ds = raiz_salida / "datasets"
    raiz_ds.mkdir(parents=True, exist_ok=True)
    for viejo in raiz_ds.glob("_tmp_*"):
        borrar_arbol(viejo)  # empaquetados interrumpidos: nunca se publican
    ultima = leer_json(raiz_ds / "ultima.json")
    previa = leer_json(raiz_ds / ultima["version"] / "resumen.json") if ultima else None
    cortes = calcular_cortes(meses_plan, cfg, (previa or {}).get("cortes"))
    filas = filas_dataset(estados, cortes, raiz_salida)
    nombres = {int(cfg["clase"]["id"]): cfg["clase"]["nombre"]}
    contenido = huella({"filas": [(f["mes"], f["split"], f["nombre"], f["sha256_rgb"], f["sha256_espectral"],
                                   f["sha256_txt"], f["reglas_huella"], f["receta_huella"]) for f in filas],
                        "cortes": [cortes["corte_val"], cortes["corte_test"]], "nombres": nombres,
                        "formato": FORMATO_DATASET}, 16)
    if ultima and ultima.get("huella_contenido") == contenido and (raiz_ds / ultima["version"]).is_dir():
        version = ultima["version"]
        avance(f"Dataset sin cambios de contenido: se mantiene {version}.")
        return {"version": version, "nueva": False, "carpeta": raiz_ds / version,
                "yaml": {v: raiz_ds / version / v / "data.yaml" for v in VARIANTES},
                "resumen": leer_json(raiz_ds / version / "resumen.json")}

    version = _siguiente_version(raiz_ds)
    tmp = raiz_ds / f"_tmp_{version}_{uuid.uuid4().hex[:6]}"
    final = raiz_ds / version
    try:
        modos = Counter()
        for v in VARIANTES:
            for s in SPLITS:
                (tmp / v / "images" / s).mkdir(parents=True, exist_ok=True)
                (tmp / v / "labels" / s).mkdir(parents=True, exist_ok=True)
        for f in filas:
            for v in VARIANTES:
                modos[enlazar_o_copiar(f[f"_src_{v}"], tmp / v / "images" / f["split"] / f"{f['nombre']}.png",
                                       cfg["salida"]["enlaces"])] += 1
                # Las etiquetas se copian (no se enlazan) para que editar una no altere la otra variante.
                enlazar_o_copiar(f["_src_txt"], tmp / v / "labels" / f["split"] / f"{f['nombre']}.txt", "copia")
            f["imagen_rgb"] = f"rgb/images/{f['split']}/{f['nombre']}.png"
            f["imagen_espectral"] = f"espectral/images/{f['split']}/{f['nombre']}.png"
            f["etiqueta"] = f"<variante>/labels/{f['split']}/{f['nombre']}.txt"
        ancho, alto = ctx.malla.ancho, ctx.malla.alto
        problemas = verificar_paquete(tmp, filas, ancho, alto, set(nombres))
        if problemas:
            raise RuntimeError("Verificación del paquete fallida: " + "; ".join(problemas[:10]))
        conteos = {s: {"imagenes": 0, "positivas": 0, "negativas": 0, "instancias": 0, "meses": []} for s in SPLITS}
        for f in filas:
            c = conteos[f["split"]]
            c["imagenes"] += 1
            c["positivas" if f["decision"] == "positiva" else "negativas"] += 1
            c["instancias"] += int(f["instancias"])
            c["meses"].append(f["mes"])
        estado = estado_producto(conteos)
        verif_ul = {}
        for v in VARIANTES:
            yaml_tmp = tmp / v / "data.yaml"
            escribir_texto_atomico(yaml_tmp, texto_yaml(tmp / v, nombres, v, version))
            if cfg["salida"]["verificar_ultralytics"] and filas:
                verif_ul[v] = verificar_ultralytics(yaml_tmp, conteos)
            escribir_texto_atomico(yaml_tmp, texto_yaml(final / v, nombres, v, version))
        escribir_csv_atomico(tmp / "manifiesto.csv", [{k: f[k] for k in COLUMNAS_MANIFIESTO} for f in filas],
                             COLUMNAS_MANIFIESTO)
        escribir_texto_atomico(tmp / "configuracion.toml",
                               f"# Configuración efectiva de la ejecución {id_ejecucion}\n" + a_toml(cfg.efectiva()))
        meses_estado = Counter(e.get("estado") for e in estados.values())
        resumen = {
            "formato": FORMATO_DATASET, "version": version, "creado_utc": iso_utc(ahora_utc()),
            "id_ejecucion": id_ejecucion, "id_aoi": ctx.id_aoi, "huella_contenido": contenido,
            "clases": nombres, "variantes": list(VARIANTES), "estado_producto": {v: estado for v in VARIANTES},
            "cortes": cortes, "conteos": conteos, "meses_planificados": len(meses_plan),
            "meses_por_estado": dict(meses_estado),
            "perfiles": dict(Counter(f["perfil"] for f in filas)),
            "provisionales": [f["mes"] for f in filas if f["provisional"]],
            "seleccion_incompleta": [f["mes"] for f in filas if f["seleccion_incompleta"]],
            "verificacion": {"paquete": "ok", "ultralytics": verif_ul or "omitida", "enlaces": dict(modos)},
            "version_anterior": ultima["version"] if ultima else None,
            "aviso": "Pseudoetiquetas automáticas por reglas NDSI sin revisión humana; no son verificación glaciológica.",
        }
        escribir_json_atomico(tmp / "resumen.json", resumen)
        mover_con_reintentos(tmp, final)
    except BaseException:
        borrar_arbol(tmp)
        raise
    escribir_json_atomico(raiz_ds / "ultima.json", {"version": version, "huella_contenido": contenido,
                                                    "publicado_utc": resumen["creado_utc"]})
    avance(f"Dataset publicado: {final} · estado {estado} · " +
           ", ".join(f"{s} {conteos[s]['imagenes']} img/{conteos[s]['instancias']} inst" for s in SPLITS))
    return {"version": version, "nueva": True, "carpeta": final,
            "yaml": {v: final / v / "data.yaml" for v in VARIANTES}, "resumen": resumen}
