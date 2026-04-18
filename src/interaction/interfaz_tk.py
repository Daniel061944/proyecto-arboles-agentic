"""
src/interaction/interfaz_tk.py — Agente de Interacción (GUI)
Interfaz gráfica con Tkinter que muestra la foto del árbol
y la información botánica en una ventana.

Uso:
    python src/interaction/interfaz_tk.py
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

# ── Importar agentes ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "perception"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "knowledge"))

from predecir      import clasificar
from consulta_info import consultar

# ── Colores y fuentes ─────────────────────────────────────────────────────────
COLOR_FONDO    = "#F5F5F0"
COLOR_HEADER   = "#2D5016"
COLOR_ACENTO   = "#5A8F2B"
COLOR_TEXTO    = "#2C2C2C"
COLOR_TARJETA  = "#FFFFFF"
COLOR_BORDE    = "#D4E8B8"
FUENTE_TITULO  = ("Helvetica", 16, "bold")
FUENTE_LABEL   = ("Helvetica", 11, "bold")
FUENTE_VALOR   = ("Helvetica", 11)
FUENTE_BOTON   = ("Helvetica", 11, "bold")
IMG_PREVIEW    = (300, 300)


class AppArboretum(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Arboretum y Palmetum — UNAL Medellín")
        self.configure(bg=COLOR_FONDO)
        self.resizable(True, True)
        self.minsize(700, 560)
        self._construir_ui()

    # ── Construcción de la interfaz ───────────────────────────────────────────

    def _construir_ui(self):
        # Header
        header = tk.Frame(self, bg=COLOR_HEADER, pady=12)
        header.pack(fill="x")
        tk.Label(header, text="🌳 Arboretum y Palmetum — UNAL Medellín",
                 bg=COLOR_HEADER, fg="white", font=FUENTE_TITULO).pack()
        tk.Label(header, text="Sistema de Identificación de Especies Arbóreas",
                 bg=COLOR_HEADER, fg="#A8D57A", font=("Helvetica", 10)).pack()

        # Contenido principal
        contenido = tk.Frame(self, bg=COLOR_FONDO, padx=16, pady=12)
        contenido.pack(fill="both", expand=True)

        # Columna izquierda — imagen
        izq = tk.Frame(contenido, bg=COLOR_FONDO)
        izq.pack(side="left", fill="y", padx=(0, 12))

        self.canvas_img = tk.Label(
            izq, bg="#E8EDE0", width=IMG_PREVIEW[0], height=IMG_PREVIEW[1],
            text="Selecciona una imagen\npara identificar",
            font=("Helvetica", 10), fg="#666666",
            relief="groove", bd=2,
        )
        self.canvas_img.pack()

        tk.Button(
            izq, text="📷  Seleccionar imagen",
            command=self._seleccionar_imagen,
            bg=COLOR_ACENTO, fg="white", font=FUENTE_BOTON,
            relief="flat", padx=12, pady=8, cursor="hand2",
        ).pack(fill="x", pady=(10, 4))

        self.btn_analizar = tk.Button(
            izq, text="🔍  Identificar especie",
            command=self._analizar,
            bg=COLOR_HEADER, fg="white", font=FUENTE_BOTON,
            relief="flat", padx=12, pady=8, cursor="hand2",
            state="disabled",
        )
        self.btn_analizar.pack(fill="x")

        # Columna derecha — resultado
        der = tk.Frame(contenido, bg=COLOR_FONDO)
        der.pack(side="left", fill="both", expand=True)

        # Especie + confianza
        self.lbl_especie = tk.Label(
            der, text="—", bg=COLOR_FONDO,
            font=("Helvetica", 18, "bold"), fg=COLOR_HEADER,
        )
        self.lbl_especie.pack(anchor="w")

        self.lbl_confianza = tk.Label(
            der, text="", bg=COLOR_FONDO,
            font=("Helvetica", 10), fg="#666666",
        )
        self.lbl_confianza.pack(anchor="w", pady=(0, 8))

        # Barra de progreso de confianza
        self.progress = ttk.Progressbar(der, length=340, mode="determinate")
        self.progress.pack(anchor="w", pady=(0, 12))

        # Tarjeta con info botánica
        self.frame_info = tk.Frame(der, bg=COLOR_TARJETA, bd=1, relief="solid",
                                   highlightbackground=COLOR_BORDE)
        self.frame_info.pack(fill="both", expand=True)

        self.campos_labels = {}
        campos = [
            ("nombre_cientifico", "Nombre científico"),
            ("nombre_vulgar",     "Nombre vulgar"),
            ("familia",           "Familia"),
            ("utilidad",          "Utilidad"),
            ("estado",            "Estado de conservación"),
            ("propagacion",       "Propagación"),
            ("descripcion",       "Descripción"),
        ]
        for clave, etiqueta in campos:
            fila = tk.Frame(self.frame_info, bg=COLOR_TARJETA, pady=3, padx=10)
            fila.pack(fill="x")
            tk.Label(fila, text=f"{etiqueta}:",
                     bg=COLOR_TARJETA, font=FUENTE_LABEL,
                     fg=COLOR_HEADER, anchor="w", width=22).pack(side="left")
            lbl = tk.Label(fila, text="—", bg=COLOR_TARJETA,
                           font=FUENTE_VALOR, fg=COLOR_TEXTO,
                           anchor="w", wraplength=280, justify="left")
            lbl.pack(side="left", fill="x", expand=True)
            self.campos_labels[clave] = lbl

        # Barra de estado
        self.lbl_estado = tk.Label(
            self, text="Listo.", bg=COLOR_HEADER, fg="white",
            font=("Helvetica", 9), anchor="w", padx=10,
        )
        self.lbl_estado.pack(fill="x", side="bottom")

        self.ruta_imagen = None

    # ── Acciones ──────────────────────────────────────────────────────────────

    def _seleccionar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen de árbol",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos", "*.*")],
        )
        if not ruta:
            return

        self.ruta_imagen = ruta
        self._mostrar_preview(ruta)
        self.btn_analizar.configure(state="normal")
        self.lbl_estado.configure(text=f"Imagen cargada: {os.path.basename(ruta)}")

    def _mostrar_preview(self, ruta: str):
        img = Image.open(ruta).convert("RGB")
        img.thumbnail(IMG_PREVIEW, Image.LANCZOS)
        foto = ImageTk.PhotoImage(img)
        self.canvas_img.configure(image=foto, text="")
        self.canvas_img.image = foto  # mantener referencia

    def _analizar(self):
        if not self.ruta_imagen:
            return

        self.lbl_estado.configure(text="🔍 Clasificando …")
        self.update()

        try:
            resultados = clasificar(self.ruta_imagen, top_k=1)
        except FileNotFoundError as e:
            messagebox.showerror("Error de modelo", str(e))
            self.lbl_estado.configure(text="❌ Error al cargar el modelo.")
            return
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        especie, confianza = resultados[0]
        datos = consultar(especie)

        # Actualizar encabezado
        self.lbl_especie.configure(text=especie.upper())
        self.lbl_confianza.configure(text=f"Confianza: {confianza * 100:.1f}%")
        self.progress["value"] = confianza * 100

        # Actualizar campos botánicos
        for clave, lbl in self.campos_labels.items():
            valor = datos.get(clave, "—") if datos else "—"
            lbl.configure(text=valor or "—")

        self.lbl_estado.configure(
            text=f"✅ Identificado: {especie} ({confianza*100:.1f}% de confianza)"
        )


if __name__ == "__main__":
    app = AppArboretum()
    app.mainloop()
