import cv2
import subprocess
import time
import numpy as np

# Configurar el comando para capturar video con libcamera-vid y leerlo con OpenCV
VIDEO_PIPE = "libcamera-vid -t 0 --inline --flush --width 640 --height 480 --framerate 30 --codec mjpeg -o -"
CAPTURE_CMD = "libcamera-still -o capture.jpg -t 100 --nopreview"

def capture_photo():
    """Captura una foto y la guarda como 'capture.jpg'."""
    subprocess.run(CAPTURE_CMD.split(), check=True)
    print("Foto guardada como 'capture.jpg'")

def main():
    # Inicia el proceso de captura de video con libcamera-vid
    cap_process = subprocess.Popen(VIDEO_PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    # Si no podemos iniciar el proceso, mostramos un mensaje de error
    if not cap_process.stdout:
        print("No se pudo iniciar el proceso de captura de video.")
        return

    print("Presiona cualquier tecla para tomar una foto. Presiona 'q' para salir.")

    while True:
        # Leemos el flujo de bytes y lo decodificamos en un frame con OpenCV
        frame_data = cap_process.stdout.read(640 * 480 * 3)  # Tamaño de un frame MJPEG 640x480
        if len(frame_data) != 640 * 480 * 3:
            print("Error al capturar el frame.")
            break

        # Convierte los datos del frame en una imagen
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((480, 640, 3))

        cv2.imshow("Raspberry Pi Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # Cualquier tecla presionada
            if key == ord('q'):
                break
            capture_photo()

    cap_process.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
