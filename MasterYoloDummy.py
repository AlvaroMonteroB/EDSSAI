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
import time
import math
import copy
import RPi.GPIO as GPIO

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
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best_FaceV2.pt', force_reload=False)
    else:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/EDSSAI150.pt', force_reload=False)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # img = img.to(device) # Asegurar que img esté en el device

    # Realizar la inferencia
    with torch.no_grad():
        results = model(img) # Objeto original de resultados

    # --- Crear copia profunda para modificar ---
    results_aux = copy.deepcopy(results)
    print("\nTipo de results_aux:", type(results_aux))

    # --- Seleccionar las Top 2 detecciones por confianza ---
    top_n = 2 # Queremos las 2 mejores
    try:
        # Acceder a las detecciones del results ORIGINAL (no de la copia aún)
        # para obtener todas las detecciones antes de seleccionar
        if hasattr(results, 'xyxy') and results.xyxy and len(results.xyxy[0]) > 0:
            detections_np = results.xyxy[0].cpu().numpy() # Obtener como NumPy array [N, 6]
            num_original = detections_np.shape[0]
            print(f"Número original de detecciones: {num_original}")

            if num_original > 0:
                # Ordenar el array NumPy por la columna de confianza (índice 4) en orden DESCENDENTE
                # argsort devuelve los índices que ordenarían el array
                # [::-1] invierte el orden para que sea descendente (mayor confianza primero)
                sorted_indices = np.argsort(detections_np[:, 4])[::-1]

                # Seleccionar los índices de las top N detecciones
                # El slicing [:top_n] maneja automáticamente si hay menos de N detecciones
                top_n_indices = sorted_indices[:top_n]

                # Usar los índices para obtener las filas correspondientes del array original
                top_n_detections_np = detections_np[top_n_indices]
                print(f"Seleccionadas las Top {len(top_n_detections_np)} detecciones por confianza.")

                # Convertir el array NumPy de las top N de vuelta a un tensor de PyTorch
                top_n_detections_tensor = torch.from_numpy(top_n_detections_np).to(device)

                # *** Reemplazar el tensor en la COPIA results_aux con el tensor Top N ***
                results_aux.xyxy[0] = top_n_detections_tensor

            else:
                # Si no hubo detecciones originales, asegurarse que el tensor en results_aux esté vacío
                print("No había detecciones originales.")
                # (La copia profunda ya debería tener un tensor vacío si results.xyxy[0] estaba vacío)
                results_aux.xyxy[0] = torch.empty((0, 6), device=device) # Asegurar forma correcta si es posible

        else:
            print("No se encontraron detecciones en results.xyxy[0].")
            # Asegurar que results_aux también refleje esto
            if hasattr(results_aux, 'xyxy') and isinstance(results_aux.xyxy, list):
                 results_aux.xyxy[0] = torch.empty((0, 6), device=device) # Tensor vacío

    except AttributeError:
        print("Error: El objeto 'results' no tiene el atributo '.xyxy' esperado.")
    except IndexError:
        print("Error: No se pudo acceder a results.xyxy[0]. ¿La lista está vacía?")
    except Exception as e:
        print(f"Ocurrió un error inesperado durante la selección Top N: {e}")
        import traceback
        traceback.print_exc() # Útil para depurar

    # --- Imprimir las detecciones Top N que quedaron en results_aux ---
    print(f"\nDetecciones Top {top_n} restantes en results_aux (después de la selección):")
    try:
        if hasattr(results_aux, 'xyxy') and results_aux.xyxy:
            # Acceder al tensor DENTRO de results_aux (que ya debe contener solo las top N)
            final_detections_np = results_aux.xyxy[0].cpu().numpy()
            if len(final_detections_np) > 0:
                # Ordenar por confianza para imprimir (opcional, ya deberían estar ordenadas)
                final_detections_np = final_detections_np[np.argsort(final_detections_np[:, 4])[::-1]]
                for i, (x1, y1, x2, y2, conf, cls) in enumerate(final_detections_np):
                    print(f"  Detección {i + 1} (Top {len(final_detections_np)}):")
                    print(f"    - Coords: ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f})")
                    print(f"    - Confianza: {conf:.2f}") # Esta debería ser alta
                    print(f"    - Clase: {int(cls)}")
            else:
                print("  No quedaron detecciones después de la selección (o no había originales).")
        else:
             print("  El atributo .xyxy no existe o está vacío en results_aux.")
    except Exception as e:
        print(f"Error imprimiendo detecciones finales: {e}")

    # Retornar la COPIA MODIFICADA que ahora contiene solo las Top N detecciones
    return results_aux

def extraer_bounding_boxes(results, img):
    """
    Extrae las regiones de interés (bounding boxes) de una imagen NumPy.

    Args:
        results: Objeto de resultados del modelo YOLOv5.
        img (np.ndarray): Imagen de entrada en formato NumPy array (se asume BGR).

    Returns:
        list: Lista de imágenes recortadas (NumPy arrays).
              Contiene las detecciones 1ra y 3ra si hay 3+, las 2 si hay 2,
              la única si hay 1, o vacía si no hay.
              ¡ESTA LÓGICA DE SELECCIÓN PUEDE NECESITAR REVISIÓN!
    """
    # Verificar tipo de imagen de entrada
    if not isinstance(img, np.ndarray):
        # Podrías intentar convertir desde PIL aquí si fuera necesario
        print(f"[ERROR] extraer_bounding_boxes: La imagen de entrada no es un NumPy array (tipo: {type(img)})")
        return []

    # Extraer bounding boxes
    try:
        # results.xyxy[0] contiene las detecciones [x1, y1, x2, y2, conf, cls] para la imagen 0
        detections = results.xyxy[0].cpu().numpy()
    except Exception as e:
        print(f"[ERROR] extraer_bounding_boxes: No se pudieron extraer detecciones de 'results'. Error: {e}")
        return []

    print(f"[DEBUG] extraer_bounding_boxes: {len(detections)} detecciones encontradas.")
    cropped_items = []
    img_height, img_width = img.shape[:2] # Dimensiones para clipping

    for i, detection_data in enumerate(detections):
        if len(detection_data) < 6:
             print(f"  - Advertencia: Detección {i+1} tiene formato inesperado. Saltando.")
             continue
        x1, y1, x2, y2, conf, cls = detection_data

        print(f"[DEBUG] Detección {i+1}: Box [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], Conf: {conf:.2f}, Cls: {int(cls)}")

        # Convertir a enteros y asegurar límites (Clipping)
        x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
        x2_int, y2_int = min(img_width, int(x2)), min(img_height, int(y2))

        # Validar coordenadas
        if x1_int >= x2_int or y1_int >= y2_int:
            print(f"  - Advertencia: Coordenadas inválidas tras clipping para detección {i+1}. Saltando.")
            continue

        # Recortar usando NumPy slicing
        try:
            cropped_img_np = img[y1_int:y2_int, x1_int:x2_int]
            if cropped_img_np.size == 0:
                print(f"  - Advertencia: Recorte vacío para detección {i+1}. Saltando.")
                continue
            cropped_items.append(cropped_img_np) # Añadir el array NumPy recortado
            print(f"  - Recorte {i+1} añadido, shape: {cropped_img_np.shape}")
        except Exception as e:
            print(f"  - ERROR al recortar detección {i+1} con coords [{y1_int}:{y2_int}, {x1_int}:{x2_int}]: {e}")


    # --- Lógica de selección (1ra y 3ra si hay 3+) ---
    # ¡¡¡REVISA ESTA LÓGICA!!! Puede que no sea lo que necesitas para seleccionar ojos.
    # Podrías necesitar filtrar por clase (cls) o confianza (conf) antes.
    final_cropped_images = []
    num_detected = len(cropped_items)
    if num_detected >= 3:
        print(f"[WARN] Se detectaron {num_detected} objetos. Conservando solo el 1ro y 3ro (según lógica original). ¡REVISAR!")
        final_cropped_images = [cropped_items[0], cropped_items[2]]
    elif num_detected == 2:
        print("[DEBUG] Se detectaron 2 objetos. Conservando ambos.")
        final_cropped_images = cropped_items
    elif num_detected == 1:
        print("[WARN] Solo se detectó 1 objeto.")
        final_cropped_images = cropped_items # Devuelve lista con 1 elemento
    else:
        print("[WARN] No se detectaron objetos válidos o no se pudieron recortar.")
        final_cropped_images = [] # Lista vacía

    print(f"[DEBUG] extraer_bounding_boxes: Devolviendo {len(final_cropped_images)} imágenes recortadas.")
    return final_cropped_images

def capture_video(frame_list):
    VIDEO_PIPE = "libcamera-vid -t 0 --inline --flush --width 640 --height 480 --framerate 30 --codec mjpeg -o -"
    cap_process = subprocess.Popen(VIDEO_PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    print("Presiona cualquier tecla para tomar una foto. Presiona 'q' para salir.")

    buffer = bytearray()
    save_path = "capture.jpg"  # Ruta por defecto, puedes cambiarla

    LED_PINS = [17, 27, 22, 5, 6, 13, 19, 26, 21]  # Ajusta estos pines según tu conexión física

    #GPIO.setmode(GPIO.BCM)  # Usar numeración BCM
    #GPIO.setup(LED_PINS, GPIO.OUT)

    # Apagar todos los LEDs al inicio
    #for pin in LED_PINS:
    #    GPIO.output(pin, GPIO.LOW)
    
    
    #GPIO.output(LED_PINS[0], GPIO.HIGH)
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
            #GPIO.output(LED_PINS[led_index], GPIO.LOW)
            time.sleep(0.1)  # Pequeño retraso antes de capturar la imagen
            frame_list.append(Image.fromarray(frame))
            led_index += 1
            #if led_index < len(LED_PINS):
            #    GPIO.output(LED_PINS[led_index], GPIO.HIGH)
                
        if len(frame_list)==9:
            break
    cap_process.terminate()
    cv2.destroyAllWindows()

def capture_video(frame_list, led_index):
    VIDEO_PIPE = "libcamera-vid -t 0 --inline --flush --width 640 --height 480 --framerate 30 --codec mjpeg -o -"
    cap_process = subprocess.Popen(VIDEO_PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    print("Presiona cualquier tecla para tomar una foto. Presiona 'q' para salir.")

    buffer = bytearray()

    # Definir pines GPIO de los LEDs
    LED_PINS = [17, 27, 22, 5, 6, 13, 19, 26, 21]

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PINS, GPIO.OUT)

    # Apagar todos los LEDs primero
    for pin in LED_PINS:
        GPIO.output(pin, GPIO.LOW)

    # Encender solo el LED indicado por parámetro
    if 0 <= led_index < len(LED_PINS):
        GPIO.output(LED_PINS[led_index], GPIO.HIGH)

    try:
        while True:
            buffer.extend(cap_process.stdout.read(4096))
            start = buffer.find(b'\xff\xd8')  # JPEG start
            end = buffer.find(b'\xff\xd9')    # JPEG end

            if start != -1 and end != -1 and start < end:
                jpg_data = buffer[start:end+2]
                buffer = buffer[end+2:]

                frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    cv2.imshow("Raspberry Pi Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                if key == ord('q'):
                    break
                time.sleep(0.1)
                frame_list.append(Image.fromarray(frame))
                break  # Solo una captura por llamada

    finally:
        # Apagar el LED correspondiente al finalizar
        if 0 <= led_index < len(LED_PINS):
            GPIO.output(LED_PINS[led_index], GPIO.LOW)

        cap_process.terminate()
        cv2.destroyAllWindows()
        GPIO.cleanup()

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

def main():
    pass

if "__init__" == "__main__":
    main()

frame_list=[]

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

#dummy_capture_video(frame_list)#Captura de video
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


