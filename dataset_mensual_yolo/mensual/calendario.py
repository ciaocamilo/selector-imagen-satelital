"""Calendario mensual UTC (inicio inclusivo, primer día del mes siguiente exclusivo) y corte de ejecución."""
from dataclasses import dataclass
from datetime import datetime, timezone

from .utilidades import ahora_utc, iso_utc


@dataclass(frozen=True)
class Mes:
    id: str               # "YYYY-MM"
    inicio: datetime      # inclusivo, UTC
    fin: datetime         # exclusivo, UTC (primer día del mes siguiente)
    estado_temporal: str  # "cerrado" | "abierto" | "futuro"
    fin_consulta: datetime  # min(fin, corte): solo adquisiciones ya disponibles

    @property
    def anio(self):
        return self.inicio.year

    @property
    def provisional(self):
        return self.estado_temporal == "abierto"

    @property
    def centro_ms(self):
        """Centro temporal del mes calendario completo (ms desde época), usado en el ranking."""
        return int((self.inicio.timestamp() + (self.fin.timestamp() - self.inicio.timestamp()) / 2) * 1000)


def interpretar_corte(corte=None):
    """None/"" = fecha y hora real; acepta 'YYYY-MM-DD' o ISO 8601 (sin zona = UTC)."""
    if corte is None or str(corte).strip() == "":
        return ahora_utc().replace(microsecond=0)
    if isinstance(corte, datetime):
        dt = corte
    else:
        texto = str(corte).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(texto)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _siguiente(anio, mes):
    return (anio + 1, 1) if mes == 12 else (anio, mes + 1)


def construir_calendario(anio_inicio, anio_fin, corte):
    meses = []
    for anio in range(int(anio_inicio), int(anio_fin) + 1):
        for m in range(1, 13):
            ini = datetime(anio, m, 1, tzinfo=timezone.utc)
            fin = datetime(*_siguiente(anio, m), 1, tzinfo=timezone.utc)
            if corte < ini:
                estado = "futuro"
            elif corte < fin:
                estado = "abierto"
            else:
                estado = "cerrado"
            meses.append(Mes(f"{anio:04d}-{m:02d}", ini, fin, estado, min(fin, max(ini, corte))))
    return meses


def filtrar_meses(calendario, seleccion):
    if not seleccion:
        return list(calendario)
    pedidos = {str(s).strip() for s in seleccion}
    desconocidos = pedidos - {m.id for m in calendario}
    if desconocidos:
        raise ValueError(f"Meses fuera del calendario configurado: {sorted(desconocidos)}")
    return [m for m in calendario if m.id in pedidos]


def describir_mes(m):
    return {"mes": m.id, "inicio_utc": iso_utc(m.inicio), "fin_exclusivo_utc": iso_utc(m.fin),
            "estado_temporal": m.estado_temporal, "provisional": m.provisional}
