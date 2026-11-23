"""Dataset mensual automático YOLO (Santa Isabel, 2016–2026): selección por visibilidad en cumbres,
pseudoetiquetado NDSI sin clics y dos datasets YOLO Segmentation (rgb y espectral) con splits idénticos."""
from .entorno import preparar_entorno

preparar_entorno()  # PROJ/GDAL del entorno conda antes de importar rasterio o pyproj

from .configuracion import cargar_configuracion  # noqa: E402
from .dataset import regenerar_yaml  # noqa: E402
from .flujo import ejecutar  # noqa: E402

__all__ = ["ejecutar", "cargar_configuracion", "regenerar_yaml"]
