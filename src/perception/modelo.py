"""
modelo.py — Agente de Percepción
Define la arquitectura SimpleCNN para clasificar especies de árboles.
Proyecto: Arboretum y Palmetum UNAL Medellín
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Red neuronal convolucional simple para clasificar imágenes de árboles.

    Arquitectura:
        - 3 bloques Conv → BatchNorm → ReLU → MaxPool
        - 2 capas fully connected con Dropout
        - Salida con num_classes neuronas (una por especie)
    """

    def __init__(self, num_classes: int):
        super(SimpleCNN, self).__init__()

        # --- Bloque convolucional 1 ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # RGB → 32 filtros
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # 128x128
        )

        # --- Bloque convolucional 2 ---
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # 64x64
        )

        # --- Bloque convolucional 3 ---
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # 32x32
        )

        # --- Clasificador fully connected ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x


def crear_modelo(num_classes: int, device: torch.device = None) -> SimpleCNN:
    """
    Crea e inicializa el modelo en el dispositivo indicado.

    Args:
        num_classes: Número de especies a clasificar.
        device: 'cuda' o 'cpu'. Si es None, se detecta automáticamente.

    Returns:
        Modelo SimpleCNN listo para entrenar o hacer inferencia.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = SimpleCNN(num_classes=num_classes).to(device)
    return modelo


if __name__ == "__main__":
    # Prueba rápida de la arquitectura
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = crear_modelo(num_classes=3, device=device)
    print(modelo)

    # Pasar un batch de prueba: 4 imágenes RGB de 256x256
    dummy = torch.randn(4, 3, 256, 256).to(device)
    salida = modelo(dummy)
    print(f"\nEntrada: {dummy.shape}  →  Salida: {salida.shape}")
    print("✅ Arquitectura correcta.")
