"""Preselección por visibilidad en las cumbres, ranking estable y estado final del mes (funciones puras).

Cada candidato de inventario trae áreas (m²) por región sumadas a 20 m en GEE:
  sumas = {region: {"total": área de la región, "datos": área con las 5 bandas, "util": área visible}}
Los denominadores son siempre el área total de la región (no el área observada).
"""
from datetime import date, datetime, timezone

from .utilidades import huella

ESTADOS = ("futuro", "sin_adquisicion", "fuera_coleccion", "sin_qa", "sin_escena_viable", "etiqueta_ambigua",
           "error_reintentable", "exportada")
MOTIVOS_CANDIDATO = ("sin_qa", "cobertura_insuficiente", "nubes_aoi", "nubes_nucleo", "calidad_local",
                     "etiqueta_ambigua", "negativa_incierta", "error_tecnico")


def umbrales(cfg):
    p = cfg["preseleccion"]
    return {"min_cobertura_aoi_pct": float(p["min_cobertura_aoi_pct"]), "min_util_aoi_pct": float(p["min_util_aoi_pct"]),
            "min_util_sector_pct": float(p["min_util_sector_pct"])}


def huella_preseleccion(cfg, ctx):
    return huella({"umbrales": umbrales(cfg), "res": int(cfg["preseleccion"]["resolucion_m"]),
                   "sectores": ctx.resumen()["sectores"], "max_descargas": int(cfg["preseleccion"]["max_descargas_mes"])},
                  12)


def _pct(a, b):
    return round(100.0 * float(a) / float(b), 4) if b else 0.0


def metricas(cand, sectores):
    """Porcentajes por región a partir de las sumas del inventario."""
    s = cand["sumas"]
    m = {"cobertura_aoi_pct": _pct(s["aoi"]["datos"], s["aoi"]["total"]),
         "util_aoi_pct": _pct(s["aoi"]["util"], s["aoi"]["total"]),
         "util_sectores_pct": {n: _pct(s[n]["util"], s[n]["total"]) for n in sectores}}
    m["peor_sector_pct"] = min(m["util_sectores_pct"].values()) if sectores else m["util_aoi_pct"]
    return m


def motivos_preseleccion(cand, m, u):
    """Lista de códigos de rechazo (vacía si pasa) con los mismos mínimos que la verificación local."""
    if not cand.get("qa_disponible", True):
        return ["sin_qa"]
    motivos = []
    if m["cobertura_aoi_pct"] < u["min_cobertura_aoi_pct"]:
        motivos.append("cobertura_insuficiente")
    if m["util_aoi_pct"] < u["min_util_aoi_pct"]:
        motivos.append("nubes_aoi")
    if m["peor_sector_pct"] < u["min_util_sector_pct"]:
        motivos.append("nubes_nucleo")
    return motivos


def clave_ranking(cand, m, centro_ms):
    """Mayor útil en el peor sector, mayor útil en el AOI, cercanía al centro del mes, id ascendente.
    Nunca se usa la cantidad de nieve."""
    return (-round(m["peor_sector_pct"], 4), -round(m["util_aoi_pct"], 4), abs(int(cand["t"]) - int(centro_ms)),
            str(cand["id"]))


def preseleccionar(candidatos, sectores, u, centro_ms):
    """Devuelve (ordenados_aptos, todos) con métricas, motivos y orden registrados en cada candidato."""
    todos = []
    for c in candidatos:
        c = dict(c)
        c["metricas"] = metricas(c, sectores) if c.get("sumas") else None
        c["motivos"] = motivos_preseleccion(c, c["metricas"], u) if c["metricas"] else ["sin_qa"]
        c["preseleccionado"] = not c["motivos"]
        todos.append(c)
    aptos = sorted((c for c in todos if c["preseleccionado"]),
                   key=lambda c: clave_ranking(c, c["metricas"], centro_ms))
    for k, c in enumerate(aptos, 1):
        c["orden"] = k
    return aptos, todos


def motivos_locales(regiones, u, sectores):
    """Mismos mínimos sobre la malla de 10 m (porcentajes_regiones); devuelve [] si pasa."""
    motivos = []
    if regiones["aoi"]["cobertura_pct"] < u["min_cobertura_aoi_pct"]:
        motivos.append("cobertura_insuficiente")
    if regiones["aoi"]["util_pct"] < u["min_util_aoi_pct"]:
        motivos.append("nubes_aoi")
    if sectores and min(regiones[n]["util_pct"] for n in sectores) < u["min_util_sector_pct"]:
        motivos.append("nubes_nucleo")
    return motivos


def perfiles_para_mes(mes, cfg):
    """Ruta de fuentes del mes: lista de perfiles a consultar en orden y nota de enrutamiento.
    l2a_preferente: L2A si el mes alcanza su inicio; L1C solo antes de ese inicio o si el inventario L2A
    se consultó con éxito y no encontró adquisiciones locales (la nubosidad L2A no activa el respaldo)."""
    inicio_l2a = datetime.combine(date.fromisoformat(cfg["fuentes"]["inicio_l2a"]), datetime.min.time(),
                                  tzinfo=timezone.utc)
    if mes.fin <= inicio_l2a:
        if cfg["fuentes"]["modo"] == "solo_l2a":
            return [], "mes anterior al inicio de L2A y modo solo_l2a"
        return ["L1C"], "mes anterior al inicio de L2A: L1C + Cloud Score+"
    if cfg["fuentes"]["modo"] == "solo_l2a":
        return ["L2A"], "solo L2A"
    return ["L2A", "L1C"], "L2A preferente; L1C solo si el inventario L2A está vacío"


def resolver_estado(intentos, exportado, hubo_inventario, perfil_error=None):
    """Estado final y motivo principal a partir de los intentos por candidato.

    Prioridad: exportada > error_reintentable > etiqueta_ambigua > sin_qa (si ninguno pasó la preselección
    y alguno no tenía QA) > sin_escena_viable. `seleccion_incompleta` marca que un candidato mejor falló por error.
    """
    if perfil_error:
        return "error_reintentable", perfil_error, False
    if exportado is not None:
        superiores = [i for i in intentos if i.get("orden") and i["orden"] < exportado.get("orden", 0)]
        incompleta = any(i.get("resultado") == "error_tecnico" for i in superiores)
        return "exportada", exportado["resultado"], incompleta
    if not hubo_inventario:
        return "sin_adquisicion", "sin adquisiciones con huella sobre el AOI", False
    resultados = [i.get("resultado") for i in intentos]
    if "error_tecnico" in resultados:
        return "error_reintentable", "error técnico en la descarga o verificación de un candidato", False
    if any(r in ("etiqueta_ambigua", "negativa_incierta") for r in resultados):
        return "etiqueta_ambigua", "los candidatos visibles no producen una etiqueta automática segura", False
    preseleccionados = [i for i in intentos if i.get("preseleccionado")]
    if not preseleccionados and any("sin_qa" in (i.get("motivos") or []) for i in intentos):
        return "sin_qa", "hay adquisiciones sin máscara de calidad y ninguna otra pasa la preselección", False
    return "sin_escena_viable", "ninguna adquisición cumple visibilidad en AOI y cumbres", False
