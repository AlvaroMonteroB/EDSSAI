from ultralytics import YOLO
import cv2

# Cargar el modelo YOLOv8 preentrenado
model = YOLO('yolov5s.pt')  # Cambia 'yolov8n.pt' por otro modelo si lo necesitas

# Inicializar la cámara web
cap = cv2.VideoCapture(0)  # Usa 0 para la cámara web por defecto; cambia si tienes varias cámaras

# Verificar que la cámara se haya iniciado correctamente
if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

# Loop para capturar cada frame de la cámara
while True:
    # Leer el frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("No se pudo recibir el frame (fin de transmisión?). Saliendo ...")
        break

    # Realizar la detección en el frame actual
    results = model(frame)

    # Obtener el frame con las detecciones
    frame_with_boxes = results[0].plot()

    # Mostrar el frame con las detecciones
    cv2.imshow('Detección en tiempo real YOLOv8', frame_with_boxes)

    # Presiona 'q' para salir del loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()