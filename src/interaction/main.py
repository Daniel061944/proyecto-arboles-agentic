"""
src/interaction/main.py — Punto de entrada principal
Permite elegir entre interfaz de consola o interfaz gráfica (Tkinter).

Uso:
    python src/interaction/main.py              # menú de selección
    python src/interaction/main.py --modo consola
    python src/interaction/main.py --modo grafico
    python src/interaction/main.py --modo consola --imagen ruta/foto.jpg
"""

import argparse
import os
import sys


def main(args):
    modo = args.modo

    if not modo:
        print("\n🌳 Arboretum y Palmetum — UNAL Medellín")
        print("─" * 42)
        print("¿Qué interfaz deseas usar?\n")
        print("  1. Consola  (terminal)")
        print("  2. Gráfica  (ventana Tkinter)")
        print("  0. Salir")
        print()
        opcion = input("Selecciona (0-2): ").strip()
        mapa = {"1": "consola", "2": "grafico", "0": "salir"}
        modo = mapa.get(opcion, "salir")

    if modo == "salir":
        sys.exit(0)

    elif modo == "consola":
        sys.path.insert(0, os.path.dirname(__file__))
        from interfaz_consola import ejecutar
        ejecutar(ruta_imagen=args.imagen, top_k=args.top)

    elif modo == "grafico":
        try:
            from interfaz_tk import AppArboretum
            app = AppArboretum()
            app.mainloop()
        except ImportError as e:
            print(f"❌ No se pudo abrir la interfaz gráfica: {e}")
            print("   Asegúrate de instalar Pillow: pip install pillow")
            sys.exit(1)

    else:
        print(f"❌ Modo desconocido: '{modo}'. Usa --modo consola o --modo grafico.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sistema de Identificación de Árboles — Arboretum UNAL"
    )
    parser.add_argument(
        "--modo", choices=["consola", "grafico"], default=None,
        help="Interfaz a usar: 'consola' o 'grafico'"
    )
    parser.add_argument("--imagen", type=str, default=None,
                        help="Ruta de imagen (solo para modo consola)")
    parser.add_argument("--top",    type=int, default=1,
                        help="Top-N predicciones (solo para modo consola)")
    main(parser.parse_args())
