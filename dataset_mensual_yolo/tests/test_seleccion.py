from mensual.seleccion import clave_ranking, metricas, preseleccionar, resolver_estado

SECT = ["norte", "centro", "sur"]
U = {"min_cobertura_aoi_pct": 95.0, "min_util_aoi_pct": 70.0, "min_util_sector_pct": 90.0}


def cand(id_, t, aoi_util, sectores, cobertura=100.0, qa=True):
    s = {"aoi": {"total": 100.0, "datos": cobertura, "util": aoi_util}}
    s.update({n: {"total": 100.0, "datos": 100.0, "util": v} for n, v in zip(SECT, sectores)})
    return {"id": id_, "t": t, "qa_disponible": qa, "sumas": s}


def test_nube_en_cumbres_pierde_aunque_el_aoi_este_mas_despejado():
    a = cand("A", 10, 99.0, [91, 99, 99])
    b = cand("B", 10, 80.0, [97, 97, 97])
    aptos, _ = preseleccionar([a, b], SECT, U, 0)
    assert [c["id"] for c in aptos] == ["B", "A"]


def test_rechazos_con_motivos():
    _, todos = preseleccionar([cand("N", 0, 95, [50, 99, 99]), cand("C", 0, 60, [99, 99, 99], cobertura=90),
                               cand("Q", 0, 99, [99, 99, 99], qa=False)], SECT, U, 0)
    m = {c["id"]: c["motivos"] for c in todos}
    assert m == {"N": ["nubes_nucleo"], "C": ["cobertura_insuficiente", "nubes_aoi"], "Q": ["sin_qa"]}


def test_empates_estables_por_tiempo_y_luego_id():
    centro = 1000
    cs = [cand("Z", centro + 50, 90, [95, 95, 95]), cand("B", centro - 10, 90, [95, 95, 95]),
          cand("A", centro + 10, 90, [95, 95, 95])]
    for orden in (cs, cs[::-1]):
        aptos, _ = preseleccionar(orden, SECT, U, centro)
        assert [c["id"] for c in aptos] == ["A", "B", "Z"]
    m = metricas(cs[0], SECT)
    assert clave_ranking(cs[0], m, centro)[0] == -95.0


def test_prioridad_del_motivo_principal():
    assert resolver_estado([], None, False)[0] == "sin_adquisicion"
    intentos = [{"orden": 1, "preseleccionado": True, "resultado": "error_tecnico"},
                {"orden": 2, "preseleccionado": True, "resultado": "etiqueta_ambigua"}]
    assert resolver_estado(intentos, None, True)[0] == "error_reintentable"
    assert resolver_estado(intentos[1:], None, True)[0] == "etiqueta_ambigua"
    assert resolver_estado([{"motivos": ["sin_qa"]}, {"motivos": ["nubes_aoi"]}], None, True)[0] == "sin_qa"
    assert resolver_estado([{"motivos": ["nubes_aoi"]}], None, True)[0] == "sin_escena_viable"
    exp = {"orden": 2, "resultado": "positiva"}
    estado, _, incompleta = resolver_estado([intentos[0], exp], exp, True)
    assert estado == "exportada" and incompleta
