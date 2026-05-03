import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk, ImageOps

FILAS = 3
COLUMNAS = 3
TAMANO_FINAL = (512, 512)

class RecortadorGridApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Recortador automático de imágenes 3x3")
        self.root.geometry("950x800")

        self.imagen_original = None
        self.recortes = []
        self.preview_imgs = []

        self.carpeta_salida = Path("imagenes_recortadas")
        self.carpeta_salida.mkdir(exist_ok=True)

        tk.Label(
            root,
            text="Recortador de dataset 3x3",
            font=("Arial", 18, "bold")
        ).pack(pady=10)

        frame_nombre = tk.Frame(root)
        frame_nombre.pack(pady=5)

        tk.Label(frame_nombre, text="Nombre base:").grid(row=0, column=0, padx=5)

        self.entry_nombre = tk.Entry(frame_nombre, width=30)
        self.entry_nombre.grid(row=0, column=1, padx=5)
        self.entry_nombre.insert(0, "manoCara")

        frame_botones = tk.Frame(root)
        frame_botones.pack(pady=10)

        tk.Button(
            frame_botones,
            text="Seleccionar imagen",
            command=self.seleccionar_imagen,
            width=22
        ).grid(row=0, column=0, padx=10)

        tk.Button(
            frame_botones,
            text="Elegir carpeta destino",
            command=self.elegir_carpeta,
            width=22
        ).grid(row=0, column=1, padx=10)

        tk.Button(
            frame_botones,
            text="Confirmar y guardar",
            command=self.guardar_recortes,
            width=22
        ).grid(row=0, column=2, padx=10)

        self.lbl_carpeta = tk.Label(
            root,
            text=f"Carpeta destino: {self.carpeta_salida.resolve()}",
            wraplength=850
        )
        self.lbl_carpeta.pack(pady=5)

        self.frame_preview = tk.Frame(root)
        self.frame_preview.pack(pady=20)

    def seleccionar_imagen(self):
        ruta = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.webp"),
                ("Todos los archivos", "*.*")
            ]
        )

        if not ruta:
            return

        try:
            self.imagen_original = Image.open(ruta).convert("RGB")
            self.generar_recortes()
            self.mostrar_preview()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{e}")

    def elegir_carpeta(self):
        carpeta = filedialog.askdirectory(title="Selecciona carpeta de destino")
        if carpeta:
            self.carpeta_salida = Path(carpeta)
            self.carpeta_salida.mkdir(exist_ok=True)
            self.lbl_carpeta.config(
                text=f"Carpeta destino: {self.carpeta_salida.resolve()}"
            )

    def generar_recortes(self):
        self.recortes.clear()

        ancho, alto = self.imagen_original.size
        ancho_celda = ancho // COLUMNAS
        alto_celda = alto // FILAS

        for fila in range(FILAS):
            for columna in range(COLUMNAS):
                izquierda = columna * ancho_celda
                arriba = fila * alto_celda
                derecha = izquierda + ancho_celda
                abajo = arriba + alto_celda

                recorte = self.imagen_original.crop(
                    (izquierda, arriba, derecha, abajo)
                )

                recorte_final = ImageOps.pad(
                    recorte,
                    TAMANO_FINAL,
                    method=Image.LANCZOS,
                    color=(255, 255, 255),
                    centering=(0.5, 0.5)
                )

                self.recortes.append(recorte_final)

    def mostrar_preview(self):
        for widget in self.frame_preview.winfo_children():
            widget.destroy()

        self.preview_imgs.clear()

        for i, img in enumerate(self.recortes):
            preview = img.resize((160, 160), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(preview)
            self.preview_imgs.append(img_tk)

            contenedor = tk.Frame(self.frame_preview)
            contenedor.grid(row=i // 3, column=i % 3, padx=10, pady=10)

            lbl_img = tk.Label(contenedor, image=img_tk)
            lbl_img.pack()

            lbl_texto = tk.Label(
                contenedor,
                text=f"Recorte {i + 1}",
                font=("Arial", 10)
            )
            lbl_texto.pack()

    def obtener_siguiente_numero(self, nombre_base):
        existentes = list(self.carpeta_salida.glob(f"{nombre_base}_*.jpg"))

        numeros = []
        for archivo in existentes:
            try:
                numero = archivo.stem.replace(f"{nombre_base}_", "")
                numeros.append(int(numero))
            except ValueError:
                pass

        if numeros:
            return max(numeros) + 1
        return 1

    def guardar_recortes(self):
        if not self.recortes:
            messagebox.showwarning(
                "Sin imagen",
                "Primero selecciona una imagen para recortar."
            )
            return

        nombre_base = self.entry_nombre.get().strip()

        if not nombre_base:
            messagebox.showwarning(
                "Nombre vacío",
                "Escribe un nombre base para las imágenes."
            )
            return

        self.carpeta_salida.mkdir(exist_ok=True)

        inicio = self.obtener_siguiente_numero(nombre_base)

        for i, recorte in enumerate(self.recortes, start=inicio):
            nombre = f"{nombre_base}_{i:03d}.jpg"
            ruta_salida = self.carpeta_salida / nombre

            recorte.save(
                ruta_salida,
                format="JPEG",
                quality=95,
                optimize=True
            )

        messagebox.showinfo(
            "Guardado exitoso",
            f"Se guardaron {len(self.recortes)} imágenes como:\n"
            f"{nombre_base}_{inicio:03d}.jpg a {nombre_base}_{inicio + len(self.recortes) - 1:03d}.jpg\n\n"
            f"Carpeta:\n{self.carpeta_salida.resolve()}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = RecortadorGridApp(root)
    root.mainloop()