import cv2
import subprocess
import time

# Configurar el comando para capturar video con libcamera-vid y leerlo con OpenCV
VIDEO_PIPE = "libcamera-vid -t 0 --inline --flush --width 640 --height 480 --framerate 30 --codec mjpeg -o -"
CAPTURE_CMD = "libcamera-still -o capture.jpg -t 100 --nopreview"

def capture_photo():
    """Captura una foto y la guarda como 'capture.jpg'."""
    subprocess.run(CAPTURE_CMD.split(), check=True)
    print("📷 Foto guardada como 'capture.jpg'")

def main():
    cap = cv2.VideoCapture(VIDEO_PIPE, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("No se pudo abrir la cámara. Verifica tu conexión y permisos.")
        return

    print("🎥 Presiona cualquier tecla para tomar una foto. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame.")
            break

        cv2.imshow("Raspberry Pi Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:  # Cualquier tecla presionada
            if key == ord('q'):
                break
            capture_photo()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()