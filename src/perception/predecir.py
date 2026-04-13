"""
predecir.py — Agente de Interacción (consola)
Carga el modelo entrenado, clasifica una imagen y muestra
la información botánica desde info.json.

Proyecto: Arboretum y Palmetum UNAL Medellín

Uso:
    python predecir.py                          # pide la ruta por consola
    python predecir.py --imagen ruta/foto.jpg   # ruta directa
    python predecir.py --imagen foto.jpg --top 3  # muestra top-3 especies
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

from modelo import SimpleCNN

# ─────────────────────────────────────────────
# Rutas por defecto
# ─────────────────────────────────────────────
MODELO_PATH = "modelo_arboles.pth"
INFO_PATH   = "info.json"
IMG_SIZE    = 256


def cargar_modelo(modelo_path: str, device: torch.device):
    """Carga el modelo guardado y devuelve (modelo, clases)."""
    if not os.path.isfile(modelo_path):
        raise FileNotFoundError(
            f"No se encontró '{modelo_path}'. "
            "Ejecuta primero: python entrenar.py"
        )

    checkpoint = torch.load(modelo_path, map_location=device)
    clases     = checkpoint["clases"]
    img_size   = checkpoint.get("img_size", IMG_SIZE)
    num_clases = checkpoint["num_clases"]

    modelo = SimpleCNN(num_clases).to(device)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.eval()

    return modelo, clases, img_size


def preprocesar_imagen(ruta: str, img_size: int) -> torch.Tensor:
    """Carga y transforma una imagen para pasarla al modelo."""
    media   = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(media, std_dev),
    ])

    imagen = Image.open(ruta).convert("RGB")
    return tf(imagen).unsqueeze(0)  # añadir dimensión de batch


def cargar_info(info_path: str) -> dict:
    """Carga el archivo info.json con los datos botánicos de cada especie."""
    if not os.path.isfile(info_path):
        print(f"⚠️  No se encontró '{info_path}'. Se mostrará solo la clasificación.")
        return {}
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def mostrar_resultado(especie: str, confianza: float, info: dict, top_k: list):
    """Imprime el resultado en consola de forma clara."""
    separador = "─" * 50

    print(f"\n{separador}")
    print(f"🌳  ESPECIE IDENTIFICADA: {especie.upper()}")
    print(f"    Confianza: {confianza * 100:.1f}%")
    print(separador)

    # Información botánica desde info.json
    datos = info.get(especie, info.get(especie.lower(), {}))
    if datos:
        print("📋  INFORMACIÓN BOTÁNICA:")
        campos = {
            "nombre_cientifico": "Nombre científico",
            "nombre_vulgar":     "Nombre vulgar",
            "familia":           "Familia",
            "utilidad":          "Utilidad",
            "estado":            "Estado de conservación",
            "propagacion":       "Propagación",
            "descripcion":       "Descripción",
        }
        for clave, etiqueta in campos.items():
            valor = datos.get(clave)
            if valor:
                print(f"   • {etiqueta}: {valor}")
    else:
        print("ℹ️   No se encontró información botánica para esta especie en info.json.")

    # Top-k alternativas
    if len(top_k) > 1:
        print(f"\n📊  TOP {len(top_k)} PREDICCIONES:")
        for i, (nombre, prob) in enumerate(top_k, 1):
            marcador = "👉" if i == 1 else "  "
            print(f"   {marcador} {i}. {nombre:20s}  {prob * 100:.1f}%")

    print(separador + "\n")


def predecir(ruta_imagen: str, modelo, clases: list, img_size: int,
             info: dict, device: torch.device, top_k: int = 1):
    """Realiza la predicción y muestra el resultado."""
    if not os.path.isfile(ruta_imagen):
        print(f"❌ No se encontró el archivo: {ruta_imagen}")
        return

    tensor = preprocesar_imagen(ruta_imagen, img_size).to(device)

    with torch.no_grad():
        logits = modelo(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    # Ordenar por probabilidad descendente
    top_indices = probs.argsort(descending=True)[:top_k]
    top_k_list  = [(clases[i], probs[i].item()) for i in top_indices]

    especie_pred, confianza = top_k_list[0]
    mostrar_resultado(especie_pred, confianza, info, top_k_list)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n🌿 Sistema de Clasificación de Árboles — Arboretum UNAL Medellín")
    print("   Cargando modelo …")

    modelo, clases, img_size = cargar_modelo(MODELO_PATH, device)
    info = cargar_info(INFO_PATH)

    print(f"✅ Modelo cargado. Especies disponibles: {clases}\n")

    # Obtener ruta de la imagen
    if args.imagen:
        ruta_imagen = args.imagen
    else:
        ruta_imagen = input("📷 Ingresa la ruta de la imagen a clasificar: ").strip()
        if not ruta_imagen:
            print("❌ No se ingresó ninguna ruta. Saliendo.")
            sys.exit(1)

    predecir(ruta_imagen, modelo, clases, img_size, info, device, top_k=args.top)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clasifica una imagen de árbol del Arboretum UNAL")
    parser.add_argument("--imagen", type=str, default=None,  help="Ruta a la imagen (.jpg, .png …)")
    parser.add_argument("--top",    type=int, default=1,     help="Mostrar top-N predicciones (default: 1)")
    args = parser.parse_args()
    main(args)
