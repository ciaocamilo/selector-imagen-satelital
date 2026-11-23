"""Flujo único mensual: el mismo `ejecutar` lo llaman la CLI (`python -m mensual`) y el cuaderno."""
import json
import uuid
from collections import Counter
from pathlib import Path

from .calendario import construir_calendario, describir_mes, filtrar_meses, interpretar_corte
from .configuracion import Configuracion, a_toml, cargar_configuracion
from .dataset import empaquetar, split_de_mes, calcular_cortes
from .etiquetas import EXPORTABLES, REGLAS_VERSION, etiquetar_escena, reglas_perfil
from .geometria import mascara_geom, preparar_contexto
from .gee import ErrorAutenticacion, ErrorRed, ProveedorGEE
from .productos import RECETA_VERSION, clave_escena, descargar_escena, fecha_utc, receta_productos
from .seleccion import (huella_preseleccion, motivos_locales, perfiles_para_mes, preseleccionar, resolver_estado,
                        umbrales)
from .utilidades import (Bloqueo, ahora_utc, escribir_csv_atomico, escribir_json_atomico, escribir_texto_atomico,
                         huella, iso_utc, leer_json, sha256_archivo, sin_url)

INVENTARIO_VERSION = "1.0"
FINALES = ("sin_adquisicion", "fuera_coleccion", "sin_qa", "sin_escena_viable", "etiqueta_ambigua", "exportada")
COLUMNAS_MESES = ["mes", "estado", "motivo", "estado_temporal", "provisional", "corte_utc", "ruta_fuentes", "perfil",
                  "inventariados", "elegibles", "intentados", "clave_escena", "id_asset", "adquisicion_utc",
                  "decision", "instancias", "seleccion_incompleta", "conteo_motivos", "split", "vigente",
                  "procesado_en_ejecucion", "reutilizado", "procesado_utc"]
COLUMNAS_CANDIDATOS = ["mes", "perfil", "orden", "id", "adquisicion_utc", "qa_disponible", "cobertura_aoi_pct",
                       "util_aoi_pct", "util_sectores_pct", "peor_sector_pct", "nube_granulo_pct", "preseleccionado",
                       "motivos", "resultado", "detalle"]


class _SinRed:
    def inventario(self, *a, **k):
        raise ErrorRed("ejecución sin red y sin inventario en caché")

    def obtener_bloque(self, *a, **k):
        raise ErrorRed("ejecución sin red y escena ausente de la caché")


def cargar_mascara_agua(cfg, ctx):
    """Máscara opcional de lagunas verificadas (GeoJSON WGS84) para L1C; versionada por SHA-256 del archivo."""
    ruta = cfg["etiquetas"]["mascara_agua_geojson"]
    if not ruta:
        return None, None
    ruta = cfg.ruta(ruta)
    import shapely
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform as transformar
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    feats = datos.get("features", [datos])
    geoms = [shape(f.get("geometry", f)) for f in feats]
    tr = Transformer.from_crs("EPSG:4326", ctx.malla.crs, always_xy=True)
    union = transformar(tr.transform, shapely.union_all(geoms))
    return mascara_geom(ctx.malla, union), {"archivo": str(ruta), "sha256": sha256_archivo(ruta)}


class Flujo:
    def __init__(self, cfg, proveedor=None, sin_red=False, avance=print):
        self.cfg, self.avance = cfg, avance
        self.ctx = preparar_contexto(cfg)
        self.raiz = cfg.directorio_salida
        self.mascaras = self.ctx.mascaras()
        self.sectores = [n for n, _, _ in self.ctx.sectores]
        self.u = umbrales(cfg)
        self.cs = float(cfg["fuentes"]["cs_cdf_min"])
        self.agua_aux, info_agua = cargar_mascara_agua(cfg, self.ctx)
        self.reglas = {p: reglas_perfil(cfg, p, info_agua if p == "L1C" else None) for p in ("L2A", "L1C")}
        self.recetas = {p: receta_productos(p, self.ctx, self.cs) for p in ("L2A", "L1C")}
        self.h_presel = huella_preseleccion(cfg, self.ctx)
        self.h_mes = huella({"presel": self.h_presel, "reglas": {p: r["huella"] for p, r in self.reglas.items()},
                             "recetas": {p: r["huella"] for p, r in self.recetas.items()},
                             "fuentes": cfg["fuentes"]}, 12)
        self.proveedor = _SinRed() if sin_red else proveedor
        self.sin_red = sin_red
        self.red_detenida = None
        self.dir_estado = self.raiz / "estado" / self.ctx.id_aoi
        self.cache = self.raiz / "cache_escenas"

    # --- proveedor perezoso -----------------------------------------------------------------------------------
    def _prov(self):
        if self.red_detenida:
            return _SinRed()
        if self.proveedor is None:
            self.proveedor = ProveedorGEE(self.cfg, self.ctx, self.avance)
        return self.proveedor

    def _rel(self, ruta):
        return Path(ruta).resolve().relative_to(self.raiz.resolve()).as_posix()

    # --- inventario en caché ----------------------------------------------------------------------------------
    def huella_inventario(self, perfil):
        return huella({"v": INVENTARIO_VERSION, "perfil": perfil, "receta_coleccion": self.recetas[perfil]["coleccion"],
                       "aoi": self.ctx.resumen(), "res_diag": self.ctx.res_diag,
                       "cs_cdf_min": self.cs if perfil == "L1C" else None}, 12)

    def inventario(self, perfil, mes):
        ruta = self.dir_estado / "inventarios" / perfil / f"{mes.id}.json"
        h = self.huella_inventario(perfil)
        previo = leer_json(ruta)
        if previo and previo.get("huella") == h:
            vigente = previo.get("cerrado") or not self.cfg["ejecucion"]["refrescar_provisional"]
            if vigente or self.sin_red or self.red_detenida:
                return previo["candidatos"], "caché"
        cands = self._prov().inventario(perfil, mes.inicio, mes.fin_consulta)
        escribir_json_atomico(ruta, {"huella": h, "perfil": perfil, "mes": mes.id, "consultado_utc": iso_utc(ahora_utc()),
                                     "fin_consulta_utc": iso_utc(mes.fin_consulta),
                                     "cerrado": mes.fin_consulta >= mes.fin, "candidatos": cands})
        return cands, "consultado"

    # --- un mes -----------------------------------------------------------------------------------------------
    def reutilizable(self, prev, mes):
        if not prev or prev.get("huella_mes") != self.h_mes or prev.get("estado") not in FINALES:
            return False
        e = self.cfg["ejecucion"]
        if prev.get("provisional") and (mes.estado_temporal == "cerrado" or e["refrescar_provisional"]):
            return False
        if prev.get("seleccion_incompleta") and e["reintentar_pendientes"]:
            return False
        if prev["estado"] == "exportada":
            et = prev["etiqueta"]
            txt = self.raiz / et["carpeta"] / et["txt"]
            if not ((self.raiz / prev["escena"]["carpeta"] / "metadata.json").is_file() and txt.is_file()):
                return False
        return True

    def procesar_mes(self, mes, id_ejecucion):
        ruta_estado = self.dir_estado / "meses" / f"{mes.id}.json"
        prev = leer_json(ruta_estado)
        base = {**describir_mes(mes), "huella_mes": self.h_mes, "id_ejecucion": id_ejecucion,
                "procesado_utc": iso_utc(ahora_utc()), "seleccion_incompleta": False, "candidatos": [],
                "escena": None, "etiqueta": None, "perfil": None, "ruta_fuentes": None}
        if mes.estado_temporal == "futuro":
            return {**base, "estado": "futuro", "motivo": "mes posterior al corte: sin consultas a Earth Engine"}, False
        if prev and prev.get("estado") == "error_reintentable" and not self.cfg["ejecucion"]["reintentar_pendientes"]:
            return prev, True
        if self.reutilizable(prev, mes):
            return prev, True
        perfiles, nota = perfiles_para_mes(mes, self.cfg)
        base["ruta_fuentes"] = nota
        if not perfiles:
            return {**base, "estado": "fuera_coleccion", "motivo": nota}, False
        try:
            cands, perfil, origen_inv = [], None, None
            for p in perfiles:
                inv, origen_inv = self.inventario(p, mes)
                perfil = p
                cands = [c for c in inv if c.get("sumas") and c["sumas"]["aoi"].get("datos", 0) > 0]
                if cands:
                    break
                if p == "L2A" and len(perfiles) > 1:
                    self.avance(f"   {mes.id}: inventario L2A sin adquisiciones locales → respaldo L1C + Cloud Score+")
                    base["ruta_fuentes"] = nota + " · L2A vacío → L1C"
        except ErrorAutenticacion as exc:
            self.red_detenida = f"sin_autenticacion: {sin_url(exc)[:200]}"
            self.avance(f"   Earth Engine no disponible ({self.red_detenida}); el resto de meses usa solo la caché.")
            return {**base, "estado": "error_reintentable", "motivo": self.red_detenida}, False
        except ErrorRed as exc:
            return {**base, "estado": "error_reintentable", "motivo": f"inventario: {sin_url(exc)[:300]}"}, False
        base["perfil"] = perfil
        base["inventario"] = origen_inv
        if not cands:
            estado, motivo, _ = resolver_estado([], None, False)
            return {**base, "estado": estado, "motivo": motivo}, False
        aptos, todos = preseleccionar(cands, self.sectores, self.u, mes.centro_ms)
        maximo = int(self.cfg["preseleccion"]["max_descargas_mes"]) or None
        exportado, carpeta_exp, decision_exp, destino_et = None, None, None, None
        evaluadas = {}  # (tesela, instante) → id: reprocesados del mismo datatake (frecuentes en L1C 2016)
        for c in aptos:
            if maximo is not None and sum(1 for x in aptos if x.get("resultado")) >= maximo:
                c["resultado"], c["detalle"] = "no_intentado", "límite max_descargas_mes"
                continue
            adquisicion = (c.get("tesela"), int(c["t"]) // 1000)
            if adquisicion in evaluadas:
                c["resultado"], c["detalle"] = "duplicado_adquisicion", f"misma adquisición ya evaluada: {evaluadas[adquisicion]}"
                continue
            try:
                carpeta, como = descargar_escena(c, perfil, self.ctx, self.recetas[perfil], self.cache,
                                                 self._prov().obtener_bloque, int(self.cfg["gee"]["tam_bloque"]),
                                                 self.cs, self.avance)
            except ErrorAutenticacion as exc:
                self.red_detenida = f"sin_autenticacion: {sin_url(exc)[:200]}"
                c["resultado"], c["detalle"] = "error_tecnico", self.red_detenida
                continue
            except Exception as exc:  # red, cuota, verificación: el candidato queda reintentable
                c["resultado"], c["detalle"] = "error_tecnico", f"{type(exc).__name__}: {sin_url(exc)[:300]}"
                continue
            c["descarga"] = como
            meta = leer_json(carpeta / "metadata.json")
            regiones = meta["porcentajes"]["regiones_local_10m"]
            c["regiones_local_10m"] = regiones
            mot = motivos_locales(regiones, self.u, self.sectores)
            if mot:
                c["resultado"], c["detalle"] = "calidad_local", ",".join(mot)
                evaluadas[adquisicion] = c["id"]
                continue
            try:
                destino_et, dec = etiquetar_escena(carpeta, self.mascaras, self.reglas[perfil],
                                                   self.agua_aux if perfil == "L1C" else None)
            except Exception as exc:
                c["resultado"], c["detalle"] = "error_tecnico", f"etiquetado: {type(exc).__name__}: {exc}"
                continue
            c["decision"] = dec["decision"]
            if dec["decision"] in EXPORTABLES:
                c["resultado"], c["detalle"] = dec["decision"], dec["motivo"]
                exportado, carpeta_exp, decision_exp = c, carpeta, dec
                break
            c["resultado"] = "etiqueta_ambigua" if dec["decision"] == "ambigua" else "negativa_incierta"
            c["detalle"] = dec["motivo"]
            evaluadas[adquisicion] = c["id"]
        estado, motivo, incompleta = resolver_estado(todos, exportado, True)
        reg = {**base, "estado": estado, "motivo": motivo, "seleccion_incompleta": incompleta,
               "provisional": mes.provisional, "candidatos": [self._fila_cand(mes, perfil, c) for c in todos]}
        if exportado is not None:
            meta = leer_json(carpeta_exp / "metadata.json")
            reg["escena"] = {"carpeta": self._rel(carpeta_exp), "clave": meta["clave"], "perfil": perfil,
                             "id_asset": meta["id_asset"], "adquisicion_utc": meta["adquisicion_utc"]}
            reg["etiqueta"] = {"carpeta": self._rel(destino_et), "decision": decision_exp["decision"],
                               "txt": decision_exp["txt"], "sha256_txt": decision_exp["sha256_txt"],
                               "instancias": decision_exp["instancias"], "origen": decision_exp["origen"],
                               "area_pseudoetiquetada_m2": decision_exp["area_pseudoetiquetada_m2"],
                               "reglas_huella": decision_exp["reglas_huella"], "motivo": decision_exp["motivo"]}
        return reg, False

    @staticmethod
    def _fila_cand(mes, perfil, c):
        m = c.get("metricas") or {}
        return {"mes": mes.id, "perfil": perfil, "orden": c.get("orden"), "id": c["id"],
                "adquisicion_utc": iso_utc(fecha_utc(c["t"])), "qa_disponible": c.get("qa_disponible"),
                "cobertura_aoi_pct": m.get("cobertura_aoi_pct"), "util_aoi_pct": m.get("util_aoi_pct"),
                "util_sectores_pct": m.get("util_sectores_pct"), "peor_sector_pct": m.get("peor_sector_pct"),
                "nube_granulo_pct": c.get("nube_granulo_pct"), "preseleccionado": c.get("preseleccionado"),
                "motivos": c.get("motivos") or [], "resultado": c.get("resultado"), "detalle": c.get("detalle"),
                "clave_escena": clave_escena(c, perfil), "decision": c.get("decision"), "t": c["t"]}


def _fila_mes(mes, reg, cortes, corte, h_mes, procesado, reutilizado):
    """Una fila por mes planificado; los meses no procesados en esta ejecución muestran su último estado guardado."""
    if reg is None:
        estado = "futuro" if mes.estado_temporal == "futuro" else "no_procesado"
        reg = {**describir_mes(mes), "estado": estado, "huella_mes": h_mes,
               "motivo": "mes posterior al corte" if estado == "futuro" else "sin estado guardado en esta salida"}
    esc, et = reg.get("escena") or {}, reg.get("etiqueta") or {}
    cands = reg.get("candidatos") or []
    intentados = [c for c in cands if c.get("resultado") not in (None, "no_intentado")]
    motivos = Counter(m for c in cands for m in (c.get("motivos") or []))
    motivos.update(c["resultado"] for c in intentados)
    return {"mes": mes.id, "estado": reg["estado"], "motivo": reg["motivo"],
            "estado_temporal": mes.estado_temporal, "provisional": bool(reg.get("provisional")),
            "corte_utc": iso_utc(corte), "ruta_fuentes": reg.get("ruta_fuentes"), "perfil": reg.get("perfil"),
            "inventariados": len(cands), "elegibles": sum(bool(c.get("preseleccionado")) for c in cands),
            "intentados": len(intentados), "clave_escena": esc.get("clave"), "id_asset": esc.get("id_asset"),
            "adquisicion_utc": esc.get("adquisicion_utc"), "decision": et.get("decision"),
            "instancias": et.get("instancias"), "seleccion_incompleta": bool(reg.get("seleccion_incompleta")),
            "conteo_motivos": dict(motivos), "split": split_de_mes(mes.id, cortes),
            "vigente": reg.get("huella_mes") == h_mes, "procesado_en_ejecucion": procesado,
            "reutilizado": reutilizado, "procesado_utc": reg.get("procesado_utc")}


def _linea_progreso(k, n, reg, reutilizado):
    esc, et = reg.get("escena") or {}, reg.get("etiqueta") or {}
    partes = [f"[{k:>3}/{n}] {reg['mes']}", reg.get("perfil") or "—", reg["estado"]]
    if et:
        partes.append(f"{et['decision']} · {et['instancias']} inst · {esc.get('clave')}")
    else:
        partes.append(str(reg["motivo"])[:110])
    if reg.get("seleccion_incompleta"):
        partes.append("selección incompleta")
    if reutilizado:
        partes.append("(reutilizado)")
    return " · ".join(partes)


def ejecutar(config=None, *, proveedor=None, sin_red=False, avance=print, **cambios):
    """Ejecuta el flujo completo. `config`: ruta TOML, Configuracion o None (valores por defecto).
    `cambios`: claves 'seccion__clave' (p. ej. periodo__meses=["2024-01"], salida__directorio="salida_prueba")."""
    cfg = config if isinstance(config, Configuracion) else cargar_configuracion(config, **cambios)
    corte = interpretar_corte(cfg["periodo"]["corte_utc"])
    calendario = construir_calendario(cfg["periodo"]["anio_inicio"], cfg["periodo"]["anio_fin"], corte)
    seleccion = filtrar_meses(calendario, cfg["periodo"]["meses"])
    flujo = Flujo(cfg, proveedor, sin_red, avance)
    raiz = flujo.raiz
    raiz.mkdir(parents=True, exist_ok=True)
    with Bloqueo(raiz):
        id_ej = f"{ahora_utc():%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"
        dir_ej = raiz / "ejecuciones" / id_ej
        escribir_texto_atomico(dir_ej / "configuracion.toml", a_toml(cfg.efectiva()))
        plan = [m.id for m in calendario]
        cortes = calcular_cortes(plan, cfg)
        estados_tmp = Counter(m.estado_temporal for m in calendario)
        avance(f"Ejecución {id_ej} · AOI {flujo.ctx.id_aoi} · malla {flujo.ctx.malla.ancho}×{flujo.ctx.malla.alto} · "
               f"corte {iso_utc(corte)} · {len(calendario)} meses planificados ({dict(estados_tmp)}) · "
               f"{len(seleccion)} a procesar · fuentes {cfg['fuentes']['modo']} · salida {raiz}")
        filas_meses, filas_cand, procesados = [], [], {}
        for k, mes in enumerate(seleccion, 1):
            try:
                reg, reutilizado = flujo.procesar_mes(mes, id_ej)
            except Exception as exc:  # nunca se detiene el bucle por un mes
                reg = {**describir_mes(mes), "estado": "error_reintentable", "huella_mes": flujo.h_mes,
                       "motivo": f"{type(exc).__name__}: {sin_url(exc)[:300]}", "candidatos": [],
                       "id_ejecucion": id_ej, "procesado_utc": iso_utc(ahora_utc())}
                reutilizado = False
            if not reutilizado:
                escribir_json_atomico(flujo.dir_estado / "meses" / f"{mes.id}.json", reg)
            procesados[mes.id] = reutilizado
            filas_meses.append(_fila_mes(mes, reg, cortes, corte, flujo.h_mes, True, reutilizado))
            filas_cand += [{k2: v for k2, v in c.items() if k2 in COLUMNAS_CANDIDATOS} for c in reg.get("candidatos") or []]
            escribir_csv_atomico(dir_ej / "meses.csv", filas_meses, COLUMNAS_MESES)
            escribir_csv_atomico(dir_ej / "candidatos.csv", filas_cand, COLUMNAS_CANDIDATOS)
            avance(_linea_progreso(k, len(seleccion), reg, reutilizado))
        # El dataset reúne todos los meses del calendario con estado vigente (misma huella de reglas y recetas).
        estados, desactualizados, filas_plan = {}, [], []
        for m in calendario:
            e = leer_json(flujo.dir_estado / "meses" / f"{m.id}.json")
            filas_plan.append(_fila_mes(m, e, cortes, corte, flujo.h_mes, m.id in procesados, procesados.get(m.id, False)))
            if e is None:
                continue
            if e.get("huella_mes") != flujo.h_mes:
                desactualizados.append(m.id)
                continue
            estados[m.id] = e
        escribir_csv_atomico(dir_ej / "meses.csv", filas_plan, COLUMNAS_MESES)
        if desactualizados:
            avance(f"Aviso: {len(desactualizados)} mes(es) procesados con otras reglas/recetas quedan fuera del "
                   f"dataset hasta reprocesarlos: {', '.join(desactualizados[:12])}{' …' if len(desactualizados) > 12 else ''}")
        ds = empaquetar(cfg, flujo.ctx, plan, estados, raiz, id_ej, avance)
        conteo = Counter(f["estado"] for f in filas_meses)
        resumen = {"id_ejecucion": id_ej, "corte_utc": iso_utc(corte), "id_aoi": flujo.ctx.id_aoi,
                   "meses_procesados": len(seleccion), "estados_ejecucion": dict(conteo),
                   "meses_dataset": len(estados), "meses_desactualizados": desactualizados,
                   "red": "sin red" if sin_red else (flujo.red_detenida or "disponible"),
                   "version_dataset": ds["version"], "dataset_nuevo": ds["nueva"],
                   "carpeta_dataset": str(ds["carpeta"]), "yaml": {k: str(v) for k, v in ds["yaml"].items()},
                   "estado_producto": (ds["resumen"] or {}).get("estado_producto"),
                   "huellas": {"mes": flujo.h_mes, "preseleccion": flujo.h_presel,
                               "reglas": {p: r["huella"] for p, r in flujo.reglas.items()},
                               "recetas": {p: r["huella"] for p, r in flujo.recetas.items()},
                               "reglas_version": REGLAS_VERSION, "receta_version": RECETA_VERSION},
                   "meses_csv": str(dir_ej / "meses.csv"), "candidatos_csv": str(dir_ej / "candidatos.csv")}
        escribir_json_atomico(dir_ej / "resumen_ejecucion.json", resumen)
    avance("Resumen: " + ", ".join(f"{k} {v}" for k, v in sorted(conteo.items())) +
           f" · dataset {ds['version']} ({'nuevo' if ds['nueva'] else 'sin cambios'}) · "
           f"estado {resumen['estado_producto']}")
    for v, r in ds["yaml"].items():
        avance(f"  YAML {v}: {r}")
    return resumen
