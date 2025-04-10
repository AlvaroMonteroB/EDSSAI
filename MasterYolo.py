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
import subprocess
import cv2
import RPi.GPIO as GPIO
import time
import math

# Para solucionar problemas de compatibilidad entre sistemas operativos
#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath

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
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_FaceV2.pt', force_reload=False)
    else:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='EDSSAI150.pt', force_reload=False)
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

def capture_video(frame_list):
    VIDEO_PIPE = "libcamera-vid -t 0 --inline --flush --width 640 --height 480 --framerate 30 --codec mjpeg -o -"
    cap_process = subprocess.Popen(VIDEO_PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    print("Presiona cualquier tecla para tomar una foto. Presiona 'q' para salir.")

    buffer = bytearray()
    save_path = "capture.jpg"  # Ruta por defecto, puedes cambiarla

    LED_PINS = [17, 27, 22, 5, 6, 13, 19, 26, 21]  # Ajusta estos pines según tu conexión física

    GPIO.setmode(GPIO.BCM)  # Usar numeración BCM
    GPIO.setup(LED_PINS, GPIO.OUT)

    # Apagar todos los LEDs al inicio
    for pin in LED_PINS:
        GPIO.output(pin, GPIO.LOW)
    
    
    GPIO.output(LED_PINS[0], GPIO.HIGH)
    led_index = 0
    while True:
        # Leer los datos del flujo MJPEG en pequeños fragmentos
        buffer.extend(cap_process.stdout.read(4096))
        start = buffer.find(b'\xff\xd8')  # Inicio de imagen JPEG
        end = buffer.find(b'\xff\xd9')  # Fin de imagen JPEG
        
        if start != -1 and end != -1 and start < end:
            jpg_data = buffer[start:end+2]  # Extraer datos JPEG
            buffer = buffer[end+2:]  # Limpiar buffer
            
            # Decodificar la imagen
            frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if frame is not None:
                cv2.imshow("Raspberry Pi Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # Cualquier tecla presionada
            if key == ord('q'):
                break
            GPIO.output(LED_PINS[led_index], GPIO.LOW)
            time.sleep(0.1)  # Pequeño retraso antes de capturar la imagen
            frame_list.append(Image.fromarray(frame))
            led_index += 1
            if led_index < len(LED_PINS):
                GPIO.output(LED_PINS[led_index], GPIO.HIGH)
                
        if len(frame_list)==9:
            break
    cap_process.terminate()
    cv2.destroyAllWindows()

    

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

 



def procesar_pupilas(cropped_images):
    if not cropped_images:
        print("No se detectaron pupilas.")
        return None
    root = tk._default_root  # Obtener la ventana principal de Tkinter si existe
    if root is None:  
        root = tk.Tk()  # Crear ventana principal solo si no existe
        root.withdraw()  # Ocultar la ventana principal para que no sea visible
    # Crear una ventana secundaria con Toplevel para mostrar las imágenes procesadas
    top = tk.Toplevel()
    top.title("Pupilas Procesadas")

    # Lista para mantener referencias de las imágenes
    img_refs = []
    processed_images = []  # Lista para almacenar las imágenes procesadas

    # Procesar cada imagen recortada
    for i, cropped_img in enumerate(cropped_images):
        # Convertir la imagen recortada a formato OpenCV
        cropped_img_cv = np.array(cropped_img)
        cropped_img_cv = cv2.cvtColor(cropped_img_cv, cv2.COLOR_RGB2BGR)

        # Aplicar desenfoque gaussiano
        imagen_desenfocada = cv2.GaussianBlur(cropped_img_cv, (3, 3), 1.5)

        # Convertir a espacio de color YCrCb para mejorar la luminancia
        imagen_ycrcb = cv2.cvtColor(imagen_desenfocada, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(imagen_ycrcb)
        Y_ecualizado = cv2.equalizeHist(Y)
        # Aplicar CLAHE para mejorar el contraste
        clahe = cv2.createCLAHE(clipLimit=2.25, tileGridSize=(8, 8))
        Y_clahe = clahe.apply(Y_ecualizado)
        imagen_contrastada = cv2.merge([Y_clahe, Cr, Cb])
        imagen_final = cv2.cvtColor(imagen_contrastada, cv2.COLOR_YCrCb2BGR)

        # Convertir la imagen a escala de grises
        imagen_gris = cv2.cvtColor(imagen_final, cv2.COLOR_BGR2GRAY)
        processed_images.append(imagen_gris)  # Agregar imagen procesada a la lista

    # Devolver las imágenes procesadas para usar en la función evaluar_pupila
    return processed_images

def extraer_centros(results):
    """
    Extrae los centros de las bounding boxes detectadas.

    Parámetros:
        results: Resultado de la inferencia de YOLOv5.

    Retorna:
        Lista de tuplas con coordenadas (cx, cy) de los centros de cada bounding box detectada.
    """
    detections = results.xyxy[0].cpu().numpy()  # Convertir detecciones a numpy array
    centros = []

    for x1, y1, x2, y2, conf, cls in detections:
        cx = (x1 + x2) / 2  # Calcular centro X
        cy = (y1 + y2) / 2  # Calcular centro Y
        centros.append((cx, cy))

    return centros

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


def calcular_distancia_y_angulo(centros):
    """
    Calcula la distancia y el ángulo entre los dos primeros centros detectados.

    Parámetros:
        centros: Lista de tuplas (cx, cy) representando los centros detectados.

    Retorna:
        (ángulo, distancia): Tupla con el ángulo en grados y la distancia euclidiana.
    """
    if len(centros) < 2:
        return None, None  # No hay suficientes puntos para calcular

    # Extraer los primeros dos puntos
    (x1, y1), (x2, y2) = centros[:2]

    # Calcular diferencia de coordenadas
    dx, dy = x2 - x1, y2 - y1

    # Calcular ángulo en grados (atan2 da el ángulo en radianes, lo convertimos a grados)
    angulo = math.degrees(math.atan2(dy, dx))
    # Calcular distancia euclidiana
    distancia = math.sqrt(dx**2 + dy**2)

    return round(angulo, 2), round(distancia, 2)

def mostrar_resultados(processed_images, lista_centros):
    """
    Muestra las imágenes procesadas con los centros detectados y los valores calculados en una ventana Tkinter.

    Parámetros:
        processed_images: Lista de imágenes en formato OpenCV (NumPy array).
        lista_centros: Lista de listas, donde cada elemento contiene los centros (cx, cy) 
                       de las bounding boxes detectadas en la imagen correspondiente.
    """
    # Crear ventana principal si no existe
    if not tk._default_root:
        root = tk.Tk()
        root.withdraw()

    # Crear una ventana secundaria
    top = tk.Toplevel()
    top.title("Pupilas Procesadas")

    img_refs = []  # Lista para almacenar referencias de imágenes

    for i, (img, centros) in enumerate(zip(processed_images, lista_centros)):
        # Convertir la imagen a color para dibujar
        img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Dibujar los puntos detectados
        for (cx, cy) in centros:
            cv2.circle(img_colored, (round(cx), round(cy)), radius=2, color=(0, 0, 255), thickness=-1)  # Rojo

        # Dibujar línea si hay al menos dos centros
        if len(centros) >= 2:
            cv2.line(img_colored, 
                     (round(centros[0][0]), round(centros[0][1])), 
                     (round(centros[1][0]), round(centros[1][1])), 
                     (255, 255, 0), 2)  # Azul

        # Calcular ángulo y distancia (si hay al menos 2 centros)
        angulo, distancia = calcular_distancia_y_angulo(centros)

        # Convertir imagen para Tkinter
        img_pil = Image.fromarray(cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(img_pil)
        img_refs.append(img_tk)

        # Mostrar imagen en la ventana
        frame = tk.Frame(top)
        frame.grid(row=0, column=i, padx=10, pady=10)

        label_img = tk.Label(frame, image=img_tk)
        label_img.pack()

        # Mostrar ángulo y distancia debajo de la imagen
        if angulo is not None and distancia is not None:
            label_text = f"Ángulo: {angulo:.2f}°\nDistancia: {distancia:.2f} px"
        else:
            label_text = "No hay suficientes puntos"

        label_info = tk.Label(frame, text=label_text, font=("Arial", 10))
        label_info.pack()

        # Crear un canvas para dibujar la flecha representando el ángulo
        if angulo is not None:
            canvas = tk.Canvas(frame, width=100, height=50)
            canvas.pack()

            # Calcular coordenadas de la flecha
            angle_radians = math.radians(angulo)
            arrow_length = 30  

            start_x, start_y = 50, 40  
            end_x = start_x + arrow_length * math.cos(angle_radians)
            end_y = start_y + arrow_length * math.sin(angle_radians)

            canvas.create_line(start_x, start_y, end_x, end_y, arrow=tk.LAST, width=2)

    # Botón para cerrar
    button_quit = ttk.Button(top, text="Cerrar", command=top.destroy)
    button_quit.grid(row=1, column=0, columnspan=len(processed_images), pady=10)

    top.protocol("WM_DELETE_WINDOW", top.destroy)  
    top.wait_window()

class_colors = {
    0: "red",
    1: "blue",
    2: "green",
    3: "yellow",
    4: "purple"
}


frame_list=[]
#
def dummy_capture_video(frame_list):
    """
    Función dummy que simula la captura de video cargando 9 veces la misma imagen.
    
    Parámetros:
        frame_list (list): Lista donde se almacenarán las imágenes cargadas.
    """
    path = "zNwoKg0f.jpg"  # Cambia esto a la ruta de tu imagen local

    try:
        img = Image.open(path)
        for _ in range(9):
            frame_list.append(img.copy())  # Usar copy para evitar referencias compartidas
        print("Carga dummy completada: 9 imágenes añadidas.")
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")

capture_video(frame_list)#Captura de video
result_list=[]
for frame in frame_list:
    resultado_1er_Modelo = evaluar_rostro(frame)
    cropped_image=extraer_bounding_boxes(resultado_1er_Modelo,frame)
    #mostrar_imagen_recortada(cropped_image)

    # Proceso del segundo model
    processed_images = procesar_pupilas(cropped_image)
    resultado_2do_Modelo = [evaluar_pupila(processed_images[1]), evaluar_pupila(processed_images[0])]
    centros_detectados = [extraer_centros(resultado_2do_Modelo[0]), extraer_centros(resultado_2do_Modelo[1])]
    mostrar_resultados([processed_images[1], processed_images[0]], centros_detectados)


