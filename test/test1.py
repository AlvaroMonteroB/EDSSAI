import cv2
import subprocess
import time
import numpy as np

# Configurar el comando para capturar video con libcamera-vid y leerlo con OpenCV
VIDEO_PIPE = "libcamera-vid -t 0 --inline --flush --width 640 --height 480 --framerate 30 --codec mjpeg -o -"

def capture_photo(cap_process, save_path):
    """Detiene el proceso de video, captura una foto y lo reinicia."""
    cap_process.terminate()  # Detener la transmisión de video
    cap_process.wait()  # Esperar a que el proceso termine completamente
    
    capture_cmd = f"libcamera-still -o {save_path} -t 100 --nopreview"
    subprocess.run(capture_cmd.split(), check=True)
    print(f"Foto guardada en '{save_path}'")
    
    return subprocess.Popen(VIDEO_PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

def main():
    # Inicia el proceso de captura de video con libcamera-vid
    cap_process = subprocess.Popen(VIDEO_PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    print("Presiona cualquier tecla para tomar una foto. Presiona 'q' para salir.")

    buffer = bytearray()
    save_path = "capture.jpg"  # Ruta por defecto, puedes cambiarla

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
            cap_process = capture_photo(cap_process, save_path)  # Capturar foto y reiniciar video

    cap_process.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
