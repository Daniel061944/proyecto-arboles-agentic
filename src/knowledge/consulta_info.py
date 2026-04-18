"""
src/knowledge/consulta_info.py — Agente de Conocimiento
Lee info.json y devuelve la información botánica de una especie.
Puede usarse como módulo o ejecutarse directamente desde consola.

Uso:
    python src/knowledge/consulta_info.py --especie ceiba
"""

import argparse
import json
import os

# ── Ruta al archivo de conocimiento ──────────────────────────────────────────
INFO_PATH = os.path.join(os.path.dirname(__file__), "info.json")


def cargar_info(info_path: str = INFO_PATH) -> dict:
    """Carga y devuelve el diccionario completo de info.json."""
    if not os.path.isfile(info_path):
        raise FileNotFoundError(f"No se encontró '{info_path}'.")
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def consultar(especie: str, info_path: str = INFO_PATH) -> dict:
    """
    Devuelve los datos botánicos de una especie.

    Args:
        especie   : nombre de la especie (igual a la clave en info.json).
        info_path : ruta opcional al archivo JSON.

    Returns:
        Diccionario con los campos botánicos, o {} si no se encuentra.
    """
    info = cargar_info(info_path)
    # Buscar con el nombre exacto o en minúsculas
    return info.get(especie) or info.get(especie.lower()) or {}


def listar_especies(info_path: str = INFO_PATH) -> list[str]:
    """Devuelve la lista de especies registradas en info.json."""
    return list(cargar_info(info_path).keys())


def formatear_ficha(especie: str, datos: dict) -> str:
    """Genera un texto de ficha botánica bien formateado."""
    if not datos:
        return f"No se encontró información botánica para '{especie}'."

    campos = {
        "nombre_cientifico": "Nombre científico",
        "nombre_vulgar":     "Nombre vulgar",
        "familia":           "Familia",
        "utilidad":          "Utilidad",
        "estado":            "Estado de conservación",
        "propagacion":       "Propagación",
        "descripcion":       "Descripción",
    }

    lineas = [f"🌿 FICHA BOTÁNICA — {especie.upper()}", "─" * 45]
    for clave, etiqueta in campos.items():
        valor = datos.get(clave)
        if valor:
            lineas.append(f"  {etiqueta}: {valor}")
    lineas.append("─" * 45)
    return "\n".join(lineas)


# ── Ejecución directa ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consulta información botánica de una especie")
    parser.add_argument("--especie", type=str, required=True,
                        help="Nombre de la especie (ej: ceiba, guayacan, nogal)")
    parser.add_argument("--listar", action="store_true",
                        help="Listar todas las especies disponibles")
    args = parser.parse_args()

    if args.listar:
        especies = listar_especies()
        print("Especies registradas en info.json:")
        for e in especies:
            print(f"  • {e}")
    else:
        datos = consultar(args.especie)
        print(formatear_ficha(args.especie, datos))
