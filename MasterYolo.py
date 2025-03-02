import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pathlib
from random import randint
from PIL import Image, ImageDraw, ImageFont

#/home/flakis/Desktop/EDSSAI/test/capture.jpg

# Para solucionar problemas de compatibilidad entre sistemas operativos
#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath

def cargar_img(path):
    # Cargar la imagen con OpenCV para mostrarla primero
    img_cv2 = cv2.imread(path)
    # Mostrar la imagen con OpenCV
    cv2.imshow('Imagen Original', img_cv2)
    cv2.waitKey(0)  # Espera hasta que se cierre la ventana
    cv2.destroyAllWindows()  # Cierra la ventana
    
    # Convertir la imagen de BGR (OpenCV) a RGB (PIL)
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    return img_pil

def cargar_modelo(modo):
    if modo == "Face":
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/flakis/Desktop/EDSSAI/models/best_FaceV2.pt', force_reload=True)
    else:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/flakis/Desktop/EDSSAI/models/bestPupilV7.pt', force_reload=True)
    return model

def evaluar_rostro(img):
    modo = "Face"
    print("Cargar modelo face")
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


def plotear_ojos(cropped_images):
    if cropped_images:
        for i, cropped in enumerate(cropped_images):
            img_array = np.array(cropped)
            print(f"Mostrando imagen {i+1} con forma: {img_array.shape}")
            # Convertir la imagen de PIL (que tiene formato RGB) a BGR para OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            # Mostrar la imagen con OpenCV
            cv2.imshow(f'Ojo {i+1}', img_bgr)

        # Esperar una tecla para cerrar las ventanas
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No se detectaron objetos en la imagen.")


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

    # Convertir la imagen de PIL a un array de NumPy para OpenCV
    cropped_img_np = np.array(cropped_img_gray)

    ecualizada = cv2.equalizeHist(cropped_img_np)

    cv2.imshow("Pupila Detectada", ecualizada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    
    
    # Convertir la imagen de PIL a un array de NumPy para OpenCV
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    # Determinar si la imagen es en color o escala de grises
    if len(img_bgr.shape) == 3:  # Imagen en color
        height, width, _ = img_bgr.shape
    else:  # Imagen en escala de grises
        height, width = img_bgr.shape

    # Aumentar el tamaño de la imagen antes de mostrarla
    img_resized = cv2.resize(img_bgr, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_LINEAR)

    # Mostrar la imagen con OpenCV
    cv2.imshow("Pupila Detectada", img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class_colors = {
    0: "red",
    1: "blue",
    2: "green",
    3: "yellow",
    4: "purple"
}
#/home/flakis/Desktop/EDSSAI/test/capture.jpg

img_path = 'yp1.jpg' 
img_path = '/home/flakis/Desktop/EDSSAI/test/capture.jpg'
img = cargar_img(img_path)
print("Imagen cargada")
print("Evaluar rostro inicia")
resultado_1er_Modelo = evaluar_rostro(img)
print("Evaluar modelo termina")
cropped_images = extraer_bounding_boxes(resultado_1er_Modelo, img)
plotear_ojos(cropped_images)
pupila, pupila_color = procesar_pupilas(resultado_1er_Modelo, img)
resultado_2do_Modelo = evaluar_pupila(pupila)

dibujar_diferencia(pupila_color, resultado_2do_Modelo,class_colors)
