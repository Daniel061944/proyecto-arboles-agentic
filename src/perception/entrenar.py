import argparse
import json
import os
import time
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import modelo
importlib.reload(modelo)
from modelo import crear_modelo


# Ruta del dataset en Google Drive
DATASET_DIR = "/content/drive/MyDrive/dataset"

MODELO_SALIDA = "modelo_arboles.pth"
CLASES_JSON = "clases.json"
IMG_SIZE = 224
VAL_SPLIT = 0.2


def get_transforms(img_size):
    media = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(media, std_dev),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(media, std_dev),
    ])

    return train_tf, val_tf


def contar_imagenes_por_clase(dataset_dir):
    print("\n📊 Imágenes por clase:")
    for clase in sorted(os.listdir(dataset_dir)):
        ruta = os.path.join(dataset_dir, clase)
        if os.path.isdir(ruta):
            n = len([
                f for f in os.listdir(ruta)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            print(f"   {clase}: {n}")


def cargar_datasets(dataset_dir, img_size, val_split, batch_size):
    train_tf, val_tf = get_transforms(img_size)

    dataset_train_full = datasets.ImageFolder(root=dataset_dir, transform=train_tf)
    dataset_val_full = datasets.ImageFolder(root=dataset_dir, transform=val_tf)

    clases = dataset_train_full.classes
    n_total = len(dataset_train_full)

    if n_total == 0:
        raise ValueError("El dataset está vacío.")

    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    indices = torch.randperm(
        n_total,
        generator=torch.Generator().manual_seed(42)
    ).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_set = Subset(dataset_train_full, train_indices)
    val_set = Subset(dataset_val_full, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, clases


def entrenar_una_epoca(modelo, loader, criterio, optimizador, device):
    modelo.train()
    total_loss = 0.0
    correctos = 0
    total = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for imagenes, etiquetas in loader:
        imagenes = imagenes.to(device, non_blocking=True)
        etiquetas = etiquetas.to(device, non_blocking=True)

        optimizador.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            salidas = modelo(imagenes)
            loss = criterio(salidas, etiquetas)

        scaler.scale(loss).backward()
        scaler.step(optimizador)
        scaler.update()

        total_loss += loss.item() * imagenes.size(0)
        preds = salidas.argmax(dim=1)
        correctos += (preds == etiquetas).sum().item()
        total += imagenes.size(0)

    return total_loss / total, correctos / total


def validar(modelo, loader, criterio, device):
    modelo.eval()
    total_loss = 0.0
    correctos = 0
    total = 0

    with torch.no_grad():
        for imagenes, etiquetas in loader:
            imagenes = imagenes.to(device, non_blocking=True)
            etiquetas = etiquetas.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                salidas = modelo(imagenes)
                loss = criterio(salidas, etiquetas)

            total_loss += loss.item() * imagenes.size(0)
            preds = salidas.argmax(dim=1)
            correctos += (preds == etiquetas).sum().item()
            total += imagenes.size(0)

    return total_loss / total, correctos / total


def main(args):
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Dispositivo: {device}")

    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"No existe la carpeta '{DATASET_DIR}'. "
            "Monta Google Drive antes con drive.mount('/content/drive')."
        )

    contar_imagenes_por_clase(DATASET_DIR)

    print(f"\n📂 Cargando dataset desde '{DATASET_DIR}'...")

    train_loader, val_loader, clases = cargar_datasets(
        DATASET_DIR,
        IMG_SIZE,
        VAL_SPLIT,
        args.batch
    )

    num_clases = len(clases)

    print(f"\n {num_clases} especies encontradas:")
    for i, clase in enumerate(clases):
        print(f"   {i}: {clase}")

    print(f"\nTrain: {len(train_loader.dataset)} imágenes")
    print(f"Val:   {len(val_loader.dataset)} imágenes")

    with open(CLASES_JSON, "w", encoding="utf-8") as f:
        json.dump({str(i): c for i, c in enumerate(clases)}, f, ensure_ascii=False, indent=2)

    print(f"\n Mapeo de clases guardado en '{CLASES_JSON}'")

    modelo = crear_modelo(num_clases, device)

    criterio = nn.CrossEntropyLoss()
    optimizador = optim.Adam(
        modelo.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizador,
        patience=3,
        factor=0.5
    )

    print(f"\n Entrenando por {args.epocas} épocas...\n")

    mejor_val_acc = 0.0

    for epoca in range(1, args.epocas + 1):
        t0 = time.time()

        train_loss, train_acc = entrenar_una_epoca(
            modelo, train_loader, criterio, optimizador, device
        )

        val_loss, val_acc = validar(
            modelo, val_loader, criterio, device
        )

        scheduler.step(val_loss)

        duracion = time.time() - t0

        print(
            f"Época {epoca:02d}/{args.epocas} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
            f" {duracion:.1f}s"
        )

        if val_acc > mejor_val_acc:
            mejor_val_acc = val_acc

            torch.save({
                "epoch": epoca,
                "model_state_dict": modelo.state_dict(),
                "clases": clases,
                "num_clases": num_clases,
                "img_size": IMG_SIZE,
                "val_acc": val_acc,
            }, MODELO_SALIDA)

            print(f"    Mejor modelo guardado con val_acc={val_acc:.4f}")

    print(f"\n Entrenamiento completo.")
    print(f"Mejor val_acc: {mejor_val_acc:.4f}")
    print(f"Modelo guardado en '{MODELO_SALIDA}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=64)

    args = parser.parse_args([])  # útil en Colab/Jupyter
    main(args)