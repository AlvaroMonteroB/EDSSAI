import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
from random import randint
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageTk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
# Para solucionar problemas de compatibilidad entre sistemas operativos
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def ploteo():
    plt.show()
# Función para cargar la imagen

def cargar_img(path):
    """
    Carga una imagen desde la ruta especificada, la muestra en una ventana de tkinter y 
    espera a que el usuario la cierre antes de continuar con el flujo del programa.
    
    Parámetros:
        path (str): Ruta de la imagen a cargar.
    
    Retorna:
        img_pil (PIL.Image): Imagen cargada.
    """
    
    # Cargar la imagen utilizando PIL
    img_pil = Image.open(path)

    # Crear la ventana de tkinter
    root = tk.Tk()
    root.title("Imagen Cargada")

    # Convertir la imagen para tkinter
    img_tk = ImageTk.PhotoImage(img_pil)

    # Crear y mostrar la imagen en un Label
    label = tk.Label(root, image=img_tk)
    label.img = img_tk  # Mantener la referencia para evitar problemas de garbage collection
    label.pack()

    # Botón para cerrar la ventana
    btn_cerrar = tk.Button(root, text="Cerrar", command=root.destroy)
    btn_cerrar.pack()

    # Configurar para que al cerrar la ventana siga el flujo
    root.protocol("WM_DELETE_WINDOW", root.destroy)

    # Ejecutar el bucle de eventos de tkinter
    root.mainloop()

    print("La ventana se ha cerrado, el flujo continúa.")

    return img_pil


    
    


def cargar_modelo(modo):
    if modo == "Face":
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_FaceV2.pt', force_reload=True)
    else:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='bestPupilV7.pt', force_reload=True)
    return model

def evaluar_rostro(img):
    modo = "Face"
    model = cargar_modelo(modo)
    # Configurar el modelo para correr en CPU o GPU (si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # Realizar la inferencia
    results = model(img)
    return results

def evaluar_pupila(img):
    modo = "Pupil"
    model = cargar_modelo(modo)
    # Configurar el modelo para correr en CPU o GPU (si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # Realizar la inferencia
    results = model(img)
   
    detections = results.xyxy[0].cpu().numpy()  # Convertir a numpy array
    print("Bounding boxes detectadas:")

    for i, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
        print(f"Detección {i + 1}:")
        print(f"  - Esquina superior izquierda: ({x1:.2f}, {y1:.2f})")
        print(f"  - Esquina inferior derecha: ({x2:.2f}, {y2:.2f})")
        print(f"  - Confianza: {conf:.2f}")
        print(f"  - Clase: {int(cls)}")
    return results

def extraer_bounding_boxes(results, img):
    # Extraer bounding boxes
    detections = results.xyxy[0].cpu().numpy()  # Convertir a numpy array
    print("Bounding boxes detectadas:")
    cropped_images = []

    for i, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
        print(f"Detección {i + 1}:")
        print(f"  - Esquina superior izquierda: ({x1:.2f}, {y1:.2f})")
        print(f"  - Esquina inferior derecha: ({x2:.2f}, {y2:.2f})")
        print(f"  - Confianza: {conf:.2f}")
        print(f"  - Clase: {int(cls)}")

        # Recortar la imagen usando las coordenadas de la bounding box
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_images.append(cropped_img)

    # Si 3+ imágenes, solo conservar la primera y la última
    if len(cropped_images) > 2:
        cropped_images = [cropped_images[0], cropped_images[2]]

    return cropped_images





# Función para graficar el seno dentro de una ventana de tkinter
def graficar_seno_tkinter():
    # Crear la ventana principal de tkinter
    root = tk.Tk()
    root.title("Gráfica de sin(x)")

    # Crear los datos para la gráfica
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    # Crear la figura y el eje
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, label="sin(x)", color='blue')
    ax.set_title("Gráfica de sin(x)")
    ax.set_xlabel("x")
    ax.set_ylabel("sin(x)")
    ax.legend()
    ax.grid()

    # Insertar la gráfica de matplotlib en tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)  # Crear el canvas
    canvas.draw()  # Dibujar la gráfica
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Empaquetar el canvas en la ventana

    # Crear un botón de cierre
    button_quit = ttk.Button(root, text="Cerrar", command=root.quit)
    button_quit.pack(side=tk.BOTTOM, fill=tk.X)

    # Iniciar el bucle de eventos de tkinter
    root.protocol("WM_DELETE_WINDOW", root.quit)  # Asegura que se cierre correctamente al hacer clic en la X
    root.mainloop()

    # Código que continúa después de cerrar la ventana
    print("La ventana se ha cerrado, el flujo continúa.")



# Función para plotear las imágenes de los ojos
def plotear_ojos(cropped_images):
    # Verificar y mostrar el tipo de datos antes de plotear
    print("Tipo de datos de cropped_images:", type(cropped_images))
    if cropped_images:
        print("Tipo de datos de cada imagen en cropped_images:", type(cropped_images[0]))
        print("Dimensiones de cada imagen:", np.array(cropped_images[0]).shape)

    print("Ploteando")
    if not cropped_images:
        print("No se detectaron objetos en la imagen.")
        return
    
    num_images = len(cropped_images)
    
    # Limpiar la figura actual y asegurarse de que la memoria esté libre
    fig, axes = plt.subplots(1, num_images, figsize=(10, 5))

    # Si hay solo una imagen, axes no será un arreglo, lo convertimos en lista para iterar
    if num_images == 1:
        axes = [axes]  # Convertir a lista para iterar
    
    for i, cropped in enumerate(cropped_images):
        img_np = np.array(cropped)  # Convertir PIL a NumPy
        axes[i].imshow(img_np)
        axes[i].set_title(f'Ojo {i+1}')
        axes[i].axis("off")

    ploteo()  # Mostrar las imágene



# Función para mostrar la primera imagen recortada en una ventana de tkinter
def mostrar_imagen_recortada(cropped_images):
    """
    Muestra todas las imágenes recortadas de la lista en una ventana de tkinter.
    
    Parámetros:
        cropped_images (list): Lista de imágenes recortadas.
    """
    # Verificar que hay imágenes en la lista
    if not cropped_images:
        print("No hay imágenes recortadas para mostrar.")
        return

    # Crear la ventana principal de tkinter
    root = tk.Toplevel()
    root.title("Imágenes Recortadas")

    # Lista para mantener referencias de las imágenes
    img_refs = []

    # Mostrar las imágenes en una cuadrícula (máximo 2 por fila)
    for i, cropped_image in enumerate(cropped_images):
        img_tk = ImageTk.PhotoImage(cropped_image)
        img_refs.append(img_tk)  # Guardar referencia

        # Crear un Label y asignar la imagen
        label = tk.Label(root, image=img_tk)
        label.grid(row=i // 2, column=i % 2, padx=10, pady=10)

    # Botón para cerrar la ventana
    button_quit = ttk.Button(root, text="Cerrar", command=root.destroy)
    button_quit.grid(row=(len(cropped_images) // 2) + 1, column=0, columnspan=2, pady=10)

    # Manejo del cierre con la X
    root.protocol("WM_DELETE_WINDOW", root.destroy)

    # Iniciar el bucle de eventos de tkinter
    root.mainloop()

    # Continuar con el flujo después de que la ventana se cierre
    print("La ventana se ha cerrado, el flujo continúa.")

 



def procesar_pupilas(resultado, img):
    detections = resultado.xyxy[0].cpu().numpy()  # Convertir a numpy array

    if len(detections) == 0:
        print("No se detectaron pupilas.")
        return None

    # Tomar la primera detección
    x1, y1, x2, y2, conf, cls = detections[1]

    # Obtener el tamaño de la imagen original
    width, height = img.size  

    margin = 5
    x1, x2 = max(0, min(x1 + margin, width)), max(0, min(x2 - margin, width))
    y1, y2 = max(0, min(y1 + margin, height)), max(0, min(y2 - margin, height))

    # Validar que las coordenadas sean correctas (evitar x2 < x1 y y2 < y1)
    if x2 <= x1 or y2 <= y1:
        print("Error: Bounding box inválido después del ajuste.")
        return None

    # Recortar la imagen con el bounding box ajustado
    cropped_img = img.crop((x1, y1, x2, y2))

    # Convertir la imagen recortada a escala de grises
    cropped_img_gray = cropped_img.convert("L")

    # Mostrar la imagen recortada en escala de grises
    plt.imshow(cropped_img_gray, cmap='gray')
    plt.title("Pupila Detectada")
    plt.axis("off")
    plt.show()

    return cropped_img_gray, cropped_img

def dibujar_diferencia(img, resultado, class_colors, scale_factor=4):
    draw = ImageDraw.Draw(img)
    
    # Obtener detecciones
    detections = resultado.xyxy[0].cpu().numpy()

    for x1, y1, x2, y2, _, cls in detections:
        # Obtener el centro de la bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Obtener color para la clase
        color = class_colors[int(cls)]

        # Dibujar un punto en el centro de la detección
        draw.ellipse((center_x - 1, center_y - 1, center_x + 1, center_y + 1), fill=color)

        # Agregar el texto de la clase debajo del punto
        class_text = f"Clase {int(cls)}"
        draw.text((center_x + 5, center_y + 5), class_text, fill=color)
    
    # Aumentar el tamaño de la imagen antes de mostrarla
    width, height = img.size
    img_resized = img.resize((width * scale_factor, height * scale_factor), Image.Resampling.LANCZOS)

    # Mostrar la imagen con Matplotlib
    plt.imshow(img_resized)
    plt.title("Pupila Detectada")
    plt.axis("off")
    plt.show()

class_colors = {
    0: "red",
    1: "blue",
    2: "green",
    3: "yellow",
    4: "purple"
}
img_path = 'yp1.jpg' 

img = cargar_img(img_path)



resultado_1er_Modelo = evaluar_rostro(img)

cropped_images = extraer_bounding_boxes(resultado_1er_Modelo, img)
graficar_seno_tkinter()

# Llamada a la función para mostrar la imagen recortada
mostrar_imagen_recortada(cropped_images)  
"""
plotear_ojos(cropped_images)

pupila, pupila_color = procesar_pupilas(resultado_1er_Modelo, img)
resultado_2do_Modelo = evaluar_pupila(pupila)

dibujar_diferencia(pupila_color, resultado_2do_Modelo, class_colors)
"""