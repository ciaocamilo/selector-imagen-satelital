from datetime import timedelta

from mensual.calendario import construir_calendario, filtrar_meses, interpretar_corte
from mensual.flujo import ejecutar
from mensual.utilidades import leer_csv

from conftest import CORTE, cfg_prueba


def test_calendario_132_meses_unicos_y_bisiestos():
    cal = construir_calendario(2016, 2026, interpretar_corte(CORTE))
    assert len(cal) == 132 and len({m.id for m in cal}) == 132
    assert cal[0].id == "2016-01" and cal[-1].id == "2026-12"
    feb = {m.id: m for m in cal if m.id.endswith("-02")}
    for anio in (2016, 2020, 2024):
        assert feb[f"{anio}-02"].fin - feb[f"{anio}-02"].inicio == timedelta(days=29)
    assert feb["2017-02"].fin - feb["2017-02"].inicio == timedelta(days=28)
    for a, b in zip(cal, cal[1:]):  # contiguos: fin exclusivo = inicio del siguiente
        assert a.fin == b.inicio


def test_mes_en_curso_provisional_y_futuros():
    cal = {m.id: m for m in construir_calendario(2016, 2026, interpretar_corte(CORTE))}
    assert cal["2026-11"].estado_temporal == "abierto" and cal["2026-11"].provisional
    assert cal["2026-11"].fin_consulta == interpretar_corte(CORTE)
    assert cal["2026-12"].estado_temporal == "futuro"
    assert cal["2026-10"].estado_temporal == "cerrado" and cal["2026-10"].fin_consulta == cal["2026-10"].fin
    assert [m.id for m in filtrar_meses(list(cal.values()), ["2024-01", "2016-02"])] == ["2016-02", "2024-01"]


def test_meses_futuros_no_consultan_gee(tmp_path, falso):
    res = ejecutar(cfg_prueba(tmp_path, periodo__meses=["2026-12"]), proveedor=falso, avance=lambda *_: None)
    assert falso.llamadas_inv == [] and falso.llamadas_bloque == 0
    assert res["estados_ejecucion"] == {"futuro": 1}
    filas = leer_csv(res["meses_csv"])  # una fila por mes planificado aunque solo se procese uno
    assert len(filas) == 132 and len({f["mes"] for f in filas}) == 132
    por_mes = {f["mes"]: f for f in filas}
    assert por_mes["2026-12"]["estado"] == "futuro" and por_mes["2026-12"]["procesado_en_ejecucion"] == "True"
    assert por_mes["2016-01"]["estado"] == "no_procesado" and por_mes["2026-11"]["provisional"] == "True"
