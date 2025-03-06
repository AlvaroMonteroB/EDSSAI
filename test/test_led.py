import cv2
import subprocess
import time
import numpy as np
import os
import random
import string
import RPi.GPIO as GPIO
import time

# Configurar el comando para capturar video con libcamera-vid y leerlo con OpenCV
VIDEO_PIPE = "libcamera-vid -t 0 --inline --flush --width 1920 --height 1080 --framerate 30 --codec mjpeg -o -"

def capture_photo_from_buffer(frame, save_path):
    """Captura una foto directamente desde el frame actual y lo guarda en un archivo."""
    cv2.imwrite(save_path, frame)
    print(f"Foto guardada en '{save_path}'")
    
def name_check(path):
    # Obtener la lista de archivos en el directorio
    existing_files = os.listdir(path)
    
    # Generar un nombre aleatorio
    def generate_random_name():
        # Generar una cadena aleatoria de 8 caracteres (puedes cambiar la longitud)
        random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        return random_name
    
    # Generar un nombre aleatorio que no exista ya en el directorio
    new_name = generate_random_name()
    while new_name in existing_files:
        new_name = generate_random_name()

    return new_name
    

def main():
    # Inicia el proceso de captura de video con libcamera-vid
    cap_process = subprocess.Popen(VIDEO_PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    print("Presiona cualquier tecla para tomar una foto. Presiona 'q' para salir.")

    buffer = bytearray()
    path = "/home/flakis/Desktop/Phototest"  # Ruta por defecto, puedes cambiarla

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
            
            aux=name_check(path)
            save_path=path+"/"+aux+".jpg"
            capture_photo_from_buffer(frame, save_path)
            
            led_index += 1
            if led_index < len(LED_PINS):
                GPIO.output(LED_PINS[led_index], GPIO.HIGH)
              # Captura la foto desde el frame actual

    cap_process.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
