import tkinter as tk
from tkinter import messagebox
import tkinter.ttk as ttk
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import math
import MasterYoloDummy as myd
import RPi.GPIO as GPIO

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
LED_PINS = {
    0: 17, 1: 18, 2: 27,
    3: 22, 4: 23, 5: 24,
    6: 25, 7: 5,  8: 6
}

# Inicializa los pines GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in LED_PINS.values():
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

def prender_led(cell_id):
    apagar_led()  # Apagar todos primero
    pin = LED_PINS.get(cell_id)
    if pin is not None:
        GPIO.output(pin, GPIO.HIGH)
        print(f"LED del botón {cell_id} encendido en pin {pin}.")

def apagar_led():
    for pin in LED_PINS.values():
        GPIO.output(pin, GPIO.LOW)
    print("Todos los LEDs apagados.")

class TestHirschbergApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Test de Hirschberg")
        self.root.geometry("600x400")
        
        self.current_cell_id = None
        self.photos = {}
        self.results = {}
        self.buttons = {}
        self.cap = None 
        self.captured_frame = None
        self.photo_display = None

        self.create_start_screen()


    def create_start_screen(self):
        self.release_camera_if_active()
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.geometry("400x200")
        title_label = tk.Label(self.root, text="Test de Hirschberg", font=("Arial", 24))
        title_label.pack(pady=20)
        start_button = tk.Button(self.root, text="Iniciar Test", command=self.show_grid_screen, font=("Arial", 16))
        start_button.pack()

    def show_grid_screen(self):
        # Limpiar ventanas anteriores y liberar cámara si estaba activa
        self.release_camera_if_active()
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.geometry("600x400")
        grid_frame = tk.Frame(self.root)
        grid_frame.pack(pady=20)

        for i in range(1, 10):
            row = (i - 1) // 3
            col = (i - 1) % 3

            # Determinar estado visual del botón
            if i in self.photos:
                button_text = "Foto Tomada"
                bg_color = "#b2f2bb"  # Verde pastel
            else:
                button_text = "Tomar Foto"
                bg_color = "#f8d7da"  # Rojo pastel

            btn = tk.Button(grid_frame, text=button_text, width=15, height=5,
                            command=lambda cell_id=i: self.show_camera_screen(cell_id),
                            bg=bg_color)
            btn.grid(row=row, column=col, padx=5, pady=5)
            self.buttons[i] = btn

        back_button = tk.Button(self.root, text="Regresar Inicio", command=self.create_start_screen)
        back_button.pack(pady=10)

        # Botón "Obtener Resultados"
        if len(self.photos) == 9:
            self.show_results_button = tk.Button(self.root, text="Obtener Resultados", command=self.show_loading_screen)
            self.show_results_button.pack(pady=5)


    def show_camera_screen(self, cell_id):
        self.current_cell_id = cell_id
        self.captured_frame = None

        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.geometry("800x600")

        self.camera_frame = tk.Frame(self.root)
        self.camera_frame.pack(pady=10)
        self.camera_label = tk.Label(self.camera_frame, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, bg='grey')
        self.camera_label.pack()

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        self.take_photo_button = tk.Button(button_frame, text="Tomar Foto", command=self.take_photo, state=tk.DISABLED)
        self.take_photo_button.pack(side=tk.LEFT, padx=10)
        self.accept_photo_button = tk.Button(button_frame, text="Aceptar", command=self.accept_photo, state=tk.DISABLED)
        self.accept_photo_button.pack(side=tk.LEFT, padx=10)
        back_button = tk.Button(self.root, text="Regresar", command=self.return_from_camera_screen)
        back_button.pack(pady=5)

        # Aquí se prende el LED
        prender_led(cell_id)

        # Mostrar imagen guardada o iniciar cámara
        if cell_id in self.photos:
            self.show_saved_image()
        else:
            self.start_camera()
    
    def return_from_camera_screen(self):
        self.release_camera_if_active()  # Libera la cámara si está activa
        apagar_led()                     # Apaga el LED correspondiente
        self.show_grid_screen()         # Regresa a la grilla


    def start_camera(self):
        # Liberar cámara anterior si existe y está abierta
        self.release_camera_if_active()

        print("Iniciando cámara...")
        self.cap = cv2.VideoCapture(0) # Reintenta abrir
        if not self.cap.isOpened():
            messagebox.showerror("Error de Cámara", "No se pudo acceder a la cámara.\nAsegúrate de que no esté en uso por otra aplicación.")
            self.show_grid_screen()
            return

        # Si la cámara se abrió, habilita el botón y empieza el feed
        self.take_photo_button.config(state=tk.NORMAL)
        self.update_camera_feed()


    def update_camera_feed(self):
        # Asegurarse que cap existe, está abierto y no es None
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Usa la función corregida para convertir el frame VIVO
                self.photo_display = self.convert_frame_for_display(frame)
                if self.photo_display: # Verifica que la conversión fue exitosa
                    self.camera_label.config(image=self.photo_display, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
                    self.camera_label.image = self.photo_display
                    # Programar siguiente frame sólo si la cámara sigue activa
                    if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                         self.camera_label.after(15, self.update_camera_feed) # Un poco más de tiempo entre frames
                # else: # Opcional: ¿Qué hacer si la conversión falla?
                #    print("Error convirtiendo frame de cámara para display.")
            else:
                 print("Advertencia: No se pudo leer frame de la cámara (ret=False).")
                 # Podría detener el feed aquí si los errores persisten.
        # else:
            # print("update_camera_feed: Cámara no lista o ya cerrada.") # Debug
            pass # No hacer nada si la cámara no está lista


    def convert_frame_for_display(self, frame_original):
        """
        Convierte un frame de OpenCV (BGR o Grayscale) a PhotoImage Tkinter,
        redimensionándolo para visualización. Devuelve None en caso de error.
        """
        if frame_original is None:
            print("Error en convert_frame_for_display: frame_original es None.")
            return None

        try:
            # Redimensionar
            frame_display = cv2.resize(
                frame_original,
                (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            # Convertir color BGR a RGB si es necesario
            if len(frame_display.shape) == 3:
                frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            else: # Asumir escala de grises
                frame_rgb = frame_display

            # Convertir a Pillow Image y luego a Tkinter PhotoImage
            img_pil = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=img_pil)
            return photo

        except Exception as e:
            print(f"Error durante conversión/redimensión de frame para display: {e}")
            return None

    #---------------------------------------------------------
    # FUNCIÓN MODIFICADA PARA PRUEBAS
    #---------------------------------------------------------
    def take_photo(self):
        """
        Captura una foto real del último frame de la cámara.
        """
        print("Presionado 'Tomar Foto'")

        # Verifica que la cámara esté activa y abierta
        if self.cap is None or not self.cap.isOpened():
            messagebox.showerror("Error", "La cámara no está activa.")
            return

        # Intenta capturar un nuevo frame directamente
        ret, frame = self.cap.read()
        if not ret or frame is None:
            messagebox.showerror("Error", "No se pudo capturar una imagen de la cámara.")
            return

        self.captured_frame = frame  # Guardar el frame real

        # Detener la cámara para congelar la vista
        self.release_camera_if_active()

        # Mostrar imagen capturada en el label
        self.photo_display = self.convert_frame_for_display(self.captured_frame)
        if self.photo_display:
            self.camera_label.config(image=self.photo_display, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
            self.camera_label.image = self.photo_display
        else:
            messagebox.showerror("Error", "Error al convertir imagen capturada para mostrar.")
            return

        # Activar botón Aceptar
        self.accept_photo_button.config(state=tk.NORMAL)
        self.take_photo_button.config(text="Volver a Tomar", command=self.retake_photo)


    def retake_photo(self):
        """Reinicia la cámara para volver a tomar la foto (modo normal)."""
        print("Presionado 'Volver a Tomar'")
        # Asegurarse que el botón Aceptar esté deshabilitado hasta nueva captura/carga
        self.accept_photo_button.config(state=tk.DISABLED)
        # Cambiar texto del botón tomar foto de nuevo a su estado inicial
        self.take_photo_button.config(text="Tomar Foto", command=self.take_photo)
        # Reiniciar la cámara (esto iniciará el feed de nuevo)
        self.start_camera()


    def accept_photo(self):
        """Guarda el frame actualmente en self.captured_frame."""
        print("Presionado 'Aceptar'")
        # Verificar que haya un frame "capturado" (real o simulado)
        if hasattr(self, "captured_frame") and self.captured_frame is not None:
            output_folder = "imagenes"
            if not os.path.exists(output_folder):
                try:
                    os.makedirs(output_folder)
                except OSError as e:
                    messagebox.showerror("Error", f"No se pudo crear la carpeta '{output_folder}': {e}")
                    return

            # Guardar la imagen (usar PNG es bueno para evitar compresión con pérdida)
            filename = f"{output_folder}/celda_{self.current_cell_id}.png"
            print(f"Guardando imagen en: {filename}")
            try:
                # Guardar el frame que está en self.captured_frame
                success = cv2.imwrite(filename, self.captured_frame)
                if success:
                    print("Guardado exitoso.")
                    self.photos[self.current_cell_id] = filename
                    # Actualizar botón en la grilla (solo estado visual)
                    self.update_grid_button_status() # Llama a la función que solo actualiza texto/estado
                    # Regresar a la pantalla de la grilla (esto también libera cámara si estaba activa)
                    self.show_grid_screen()
                else:
                    print("Error", f"OpenCV reportó un error al intentar guardar:\n{filename}")
                    #messagebox.showerror("Error", f"OpenCV reportó un error al intentar guardar:\n{filename}")
            except Exception as e:
                print("Error", f"Excepción al guardar la imagen:\n{filename}\nError: {e}")
                #messagebox.showerror("Error", f"Excepción al guardar la imagen:\n{filename}\nError: {e}")

        else:
           messagebox.showwarning("Advertencia", "No hay ninguna foto 'capturada' (real o de prueba) para aceptar.")


    def update_grid_button_status(self):
        """Actualiza el texto y color del botón en la grilla."""
        if self.current_cell_id in self.buttons:
            btn = self.buttons[self.current_cell_id]
            btn.config(text="Foto Tomada", bg="#b2f2bb")  # Verde pastel
        else:
            print(f"Advertencia: Botón para celda {self.current_cell_id} no encontrado en self.buttons.")

    def show_saved_image(self):
        """Muestra la imagen ya ACEPTADA y guardada previamente."""
        print(f"Mostrando imagen guardada para celda {self.current_cell_id}")
        if self.current_cell_id in self.photos:
            img_path = self.photos[self.current_cell_id]
            if not os.path.exists(img_path):
                 messagebox.showerror("Error", f"El archivo de imagen guardada no se encontró:\n{img_path}")
                 # ¿Qué hacer? Podríamos intentar iniciar la cámara o regresar
                 self.start_camera() # Intentar iniciar cámara como fallback
                 return

            try:
                # Cargar la imagen guardada con OpenCV para consistencia
                saved_frame = cv2.imread(img_path) # Carga como BGR por defecto
                if saved_frame is None:
                    raise ValueError("cv2.imread devolvió None")

                # Convertir para display usando la misma función
                self.photo_display = self.convert_frame_for_display(saved_frame)
                if self.photo_display:
                    self.camera_label.config(image=self.photo_display, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
                    self.camera_label.image = self.photo_display
                    # Si mostramos una foto ya guardada/aceptada, los botones se deshabilitan
                    self.take_photo_button.config(text="Foto ya Tomada",state=tk.NORMAL)
                    self.accept_photo_button.config(state=tk.DISABLED)
                else:
                     raise ValueError("Falló la conversión para display de imagen guardada")

            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar o mostrar la imagen guardada:\n{img_path}\nError: {e}")
                # Fallback: intentar iniciar cámara o regresar
                self.start_camera()
        else:
            # Si por alguna razón se llama a esta función sin que haya foto guardada, iniciar cámara
            print(f"Advertencia: show_saved_image llamada para celda {self.current_cell_id} sin foto registrada.")
            self.start_camera()


    # --- Funciones de Resultados y Reinicio (sin cambios relevantes a la petición) ---

    def show_loading_screen(self):
        """Muestra pantalla de carga y EJECUTA procesamiento real (BLOQUEANTE)."""
        self.release_camera_if_active()
        self._clear_widgets()
        self.root.geometry("450x200")
        self.root.title("Procesando Imágenes...")

        self.test_num_label = tk.Label(self.root, text="Iniciando...", font=("Arial", 16, "bold"))
        self.test_num_label.pack(pady=(20, 5))
        self.progress_bar = ttk.Progressbar(self.root, orient='horizontal', length=350, mode='determinate', maximum=9)
        self.progress_bar.pack(pady=10)
        self.step_label = tk.Label(self.root, text="- Preparando...", font=("Arial", 12), justify=tk.CENTER)
        self.step_label.pack(pady=(5, 20))
        self.root.update_idletasks() # Mostrar UI antes de bloquear

        # --- Bucle de Procesamiento Real (BLOQUEANTE) ---
        print("--- INICIO PROCESAMIENTO ---")
        all_successful = True
        self.detailed_results = {} # Limpiar resultados

        for i in range(1, 10):
            self.test_num_label.config(text=f"Test {i}/9")
            self.progress_bar['value'] = i - 1
            self.step_label.config(text="- Cargando imagen...")
            self.root.update_idletasks()
            print(f"\nProcesando Celda {i}...")

            image_path = self.photos.get(i)
            if not image_path or not os.path.exists(image_path):
                messagebox.showerror("Error", f"Falta imagen para Celda {i}:\n{image_path}")
                all_successful = False; self.detailed_results[i] = {'error': 'Imagen Faltante'}; print("Error: Imagen Faltante"); break
            frame = cv2.imread(image_path)
            if frame is None:
                messagebox.showerror("Error", f"No se pudo leer Celda {i}:\n{image_path}")
                all_successful = False; self.detailed_results[i] = {'error': 'Error Lectura'}; print("Error: Lectura Imagen"); break

            try:
                # Paso 1: Ojos
                self.step_label.config(text="- Identificando ojos..."); self.root.update_idletasks(); print("  Identificando ojos...")
                resultado_1er_Modelo = myd.evaluar_rostro(frame) 
                print('evaluacion hecha')
                cropped_image = myd.extraer_bounding_boxes(resultado_1er_Modelo, frame) 
                print('imagen cortada')
                #if cropped_image is None or cropped_image.size == 0: raise ValueError("Fallo extracción ojos.")

                # Paso 2: Pupilas/Reflejos
                self.step_label.config(text="- Identificando reflejos y pupila..."); self.root.update_idletasks(); print("  Procesando pupilas...")
                print('Entro en procesar pupilas')
                processed_images_pair = myd.procesar_pupilas(cropped_image) 
                print('Salgo de procesar pupilas')
                
                if not isinstance(processed_images_pair, list) or len(processed_images_pair) != 2: raise ValueError("Fallo procesar_pupilas.")
                print("  Evaluando ojo izquierdo...")
                resultado_ojo_izq = myd.evaluar_pupila(processed_images_pair[0]) 
                print("  Evaluando ojo derecho...")
                resultado_ojo_der = myd.evaluar_pupila(processed_images_pair[1]) 

                # Paso 3: Calcular/Guardar
                self.step_label.config(text="- Calculando y guardando..."); self.root.update_idletasks(); print("  Extrayendo centros...")
                centros_ojo_izq = myd.extraer_centros(resultado_ojo_izq)
                centros_ojo_der = myd.extraer_centros(resultado_ojo_der)
                print(f"  Centros Izq: {centros_ojo_izq}, Centros Der: {centros_ojo_der}")

                self.detailed_results[i] = {'images': processed_images_pair, 'centers': [centros_ojo_izq, centros_ojo_der]}
                self.progress_bar['value'] = i
                self.root.update_idletasks()
                print(f"  Celda {i} procesada OK.")
                # time.sleep(0.1) # Pausa opcional

            except Exception as e:
                print(f"  ERROR procesando Celda {i}: {e}")
                self.detailed_results[i] = {'error': str(e)} # Guardar error
                all_successful = False
                if not messagebox.askyesno("Error de Procesamiento", f"Error en Celda {i}:\n{e}\n\n¿Continuar con las demás?"):
                    break # Detener si el usuario dice No

        # --- Fin del Bucle ---
        print("--- FIN PROCESAMIENTO ---")
        if all_successful:
            self.test_num_label.config(text="Completado"); self.step_label.config(text="¡Proceso Finalizado!"); self.progress_bar['value'] = 9
        else:
             self.test_num_label.config(text="Interrumpido"); self.step_label.config(text="Proceso con errores.")
        self.root.update_idletasks()
        self.root.after(500, self.show_detailed_results_grid) # Ir a la nueva grilla de resultados

    # ----------------------------------------------------
    # --- PANTALLAS Y FUNCIONES DE VISUALIZACIÓN DE RESULTADOS ---
    # ----------------------------------------------------

    def show_detailed_results_grid(self):
        """Muestra la grilla 3x3 para ver los resultados detallados."""
        self.release_camera_if_active()
        self._clear_widgets()
        self.root.geometry("600x450")
        self.root.title("Resultados del Test - Seleccione Celda")
        grid_frame = tk.Frame(self.root)
        grid_frame.pack(pady=20, padx=20)
        self.buttons = {} # Limpiar botones de grilla anterior

        for i in range(1, 10):
            row = (i - 1) // 3; col = (i - 1) % 3
            button_text = f"Celda {i}\n"
            button_state = tk.DISABLED
            result_info = self.detailed_results.get(i)

            if result_info:
                 if 'error' in result_info: button_text += f"(Error: {result_info['error'][:10]}...)" # Mostrar parte del error
                 else: button_text += "Ver Detalles"; button_state = tk.NORMAL
            elif i in self.photos: button_text += "(No procesado)" # Había foto pero no resultado
            else: button_text += "(Foto Faltante)"

            btn = tk.Button(grid_frame, text=button_text, width=15, height=5, state=button_state,
                            command=lambda cell_id=i: self.display_single_result(cell_id))
            btn.grid(row=row, column=col, padx=10, pady=10)
            self.buttons[i] = btn

        restart_button = tk.Button(self.root, text="Reiniciar Test", command=self.restart_test, font=("Arial", 12))
        restart_button.pack(pady=20)

    def display_single_result(self, cell_id):
        """Abre la ventana Toplevel mostrando los detalles de una celda."""
        print(f"Mostrando resultados detallados para Celda {cell_id}")
        result_data = self.detailed_results.get(cell_id)
        if result_data and 'error' not in result_data:
            processed_images = result_data.get('images')
            lista_centros = result_data.get('centers')
            if processed_images and lista_centros:
                # Llamar al método que crea la ventana Toplevel
                self.mostrar_resultados_toplevel(processed_images, lista_centros, cell_id)
            else: messagebox.showwarning("Datos Incompletos", f"Faltan datos para Celda {cell_id}.")
        else: messagebox.showerror("Error", f"No hay resultados válidos para Celda {cell_id}.")

    def calcular_distancia_y_angulo(self, centros):
        """Calcula distancia y ángulo entre los dos primeros puntos."""
        if not centros or len(centros) < 2 or not all(isinstance(p, (tuple, list)) and len(p) == 2 for p in centros[:2]):
            return None, None
        (x1, y1) = centros[0]; (x2, y2) = centros[1]
        try:
            distancia = math.hypot(x2 - x1, y2 - y1) # math.hypot es lo mismo que sqrt(dx^2 + dy^2)
            angulo_rad = math.atan2(y2 - y1, x2 - x1) # Ángulo en radianes
            angulo_deg = math.degrees(angulo_rad)    # Convertir a grados
            print(angulo_deg, distancia)
            return angulo_deg, distancia
        except Exception as e:
            print(f"Error calculando distancia/ángulo para {centros}: {e}")
            return None, None

    def _draw_arrow_on_canvas(self, canvas, unit_vector, size, color, width):
        """Dibuja una flecha en un canvas dado un vector unitario."""
        center_x = size / 2.0
        center_y = size / 2.0
        arrow_length = size * 0.4 # Longitud visual de la flecha

        # Calcular el punto final de la flecha escalando el vector unitario
        # Y positivo es hacia abajo en el canvas
        end_x = center_x + arrow_length * unit_vector[0]
        end_y = center_y + arrow_length * unit_vector[1] # Sumar Y

        # Dibujar la flecha
        canvas.create_line(
            center_x, center_y, end_x, end_y,
            arrow=tk.LAST, fill=color, width=width
        )

    def mostrar_resultados_toplevel(self, processed_images, lista_centros, cell_id):
        """Muestra detalles: Ojos arriba, [Flecha Izq, Grid Celda, Flecha Der] en medio."""
        top = tk.Toplevel(self.root)
        top.title(f"Detalle Celda {cell_id}")
        top.grab_set()

        # --- Configuración ---
        grid_canvas_size = 75
        grid_line_color = "gray"
        grid_highlight_color = "green"
        grid_highlight_radius_factor = 0.3
        arrow_canvas_size = 60 # Tamaño del canvas para la flecha
        arrow_color = "blue"
        arrow_width = 2
        conversion_factor_px_mm = 4.0 
        conversion_factor_mm_angulo_dp = 15.0 

        img_refs = []
        arrow_vectors = [None] * len(processed_images) # Lista para guardar vectores unitarios [izq, der]

        main_frame = tk.Frame(top, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        if processed_images:
            # Configurar 2 columnas principales para los ojos
            main_frame.grid_columnconfigure(0, weight=1)
            main_frame.grid_columnconfigure(1, weight=1)

        # --- Procesar y mostrar cada ojo (EN LA FILA SUPERIOR) ---
        num_eyes = len(processed_images)
        for i, (img_cv, centros) in enumerate(zip(processed_images, lista_centros)):
            eye_label = "Ojo Izquierdo" if i == 0 else "Ojo Derecho"
            eye_frame = tk.LabelFrame(main_frame, text=eye_label, padx=10, pady=10)
            # Colocar los frames de los ojos en la fila 0, columnas 0 y 1
            eye_frame.grid(row=0, column=i, padx=10, pady=5, sticky="nsew")

            # Reiniciar variables para este ojo
            angulo = None
            distancia = None
            unit_vector = None # Vector para la flecha de este ojo

            if img_cv is None:
                 tk.Label(eye_frame, text="Error: Imagen no disponible").pack()
                 continue # Saltar al siguiente ojo

            try:
                # --- Preparación y Dibujo en Imagen (sin cambios) ---
                if len(img_cv.shape) == 2: img_colored = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
                else: img_colored = img_cv.copy()
                colors = [(0, 0, 255), (0, 255, 0)]
                valid_points = []
                for idx, point in enumerate(centros):
                    if isinstance(point, (tuple, list)) and len(point) == 2:
                        try:
                            cx, cy = map(round, point)
                            h_img, w_img = img_colored.shape[:2]
                            if 0 <= cx < w_img and 0 <= cy < h_img:
                                cv2.circle(img_colored, (cx, cy), radius=max(1, int(h_img / 50)), color=colors[idx % len(colors)], thickness=-1)
                                valid_points.append((cx, cy))
                            else: print(f"Punto ({cx},{cy}) fuera de límites ({w_img}x{h_img})")
                        except Exception as e: print(f"Error dibujando punto {point}: {e}")
                    else: print(f"Punto inválido: {point}")

                # --- Calcular Ángulo, Distancia y VECTOR para la flecha ---
                if len(valid_points) == 2:
                     cv2.line(img_colored, valid_points[0], valid_points[1], (255, 255, 0), 1)
                     angulo, distancia = self.calcular_distancia_y_angulo(valid_points)

                     # Calcular y guardar el vector unitario para la flecha
                     p_origen = np.array(valid_points[0])
                     p_destino = np.array(valid_points[1])
                     vector = p_destino - p_origen
                     norm = np.linalg.norm(vector)
                     if norm > 1e-6:
                         unit_vector = vector / norm
                         arrow_vectors[i] = unit_vector # Guardar el vector para este ojo

                # --- Conversión y Visualización de Imagen (sin cambios) ---
                target_width = 250
                h, w = img_colored.shape[:2]
                if w == 0: raise ValueError("Ancho de imagen es cero")
                scale = target_width / w
                target_height = int(h * scale)
                interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
                img_display = cv2.resize(img_colored, (target_width, target_height), interpolation=interpolation)
                img_pil = Image.fromarray(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
                img_tk = ImageTk.PhotoImage(img_pil)
                img_refs.append(img_tk)

                label_img = tk.Label(eye_frame, image=img_tk)
                label_img.image = img_tk
                label_img.pack()

                # --- Mostrar Info Numérica ---
                if angulo is not None and distancia is not None:
                     dist_mm = distancia / conversion_factor_px_mm
                     angulo_dp = dist_mm * conversion_factor_mm_angulo_dp
                     label_text = (f"Ángulo: {angulo:.1f}° \n"
                                   f"Distancia: {distancia:.1f} px ({dist_mm:.1f} mm)   ({angulo_dp:.1f} diop prism)")
                else: label_text = "Datos insuficientes"
                label_info = tk.Label(eye_frame, text=label_text, font=("Arial", 10), justify=tk.LEFT)
                label_info.pack(pady=(5, 0), anchor='w')

                # !!! Ya NO se crea el canvas de la flecha aquí dentro !!!

            except Exception as e:
                print(f"Error procesando/mostrando ojo {i} para celda {cell_id}: {e}")
                tk.Label(eye_frame, text=f"Error al procesar:\n{str(e)[:100]}...", fg="red").pack()

        # --- FILA INTERMEDIA: [Flecha Izq | Grid Referencia | Flecha Der] ---
        middle_row_frame = tk.Frame(main_frame)
        # Colocar este frame en la fila 1, abarcando las 2 columnas principales
        middle_row_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

        # Configurar 3 columnas dentro del frame intermedio para los 3 elementos
        middle_row_frame.grid_columnconfigure(0, weight=1) # Columna para flecha izq
        middle_row_frame.grid_columnconfigure(1, weight=1) # Columna para grid central
        middle_row_frame.grid_columnconfigure(2, weight=1) # Columna para flecha der

        # --- Flecha Izquierda (Columna 0) ---
        left_arrow_canvas = tk.Canvas(middle_row_frame, width=arrow_canvas_size, height=arrow_canvas_size, bg="white", highlightthickness=1, highlightbackground="lightgrey")
        left_arrow_canvas.grid(row=0, column=0, padx=(0, 5), sticky="e") # Alinear a la derecha de su celda
        if num_eyes > 0 and arrow_vectors[0] is not None:
             self._draw_arrow_on_canvas(left_arrow_canvas, arrow_vectors[0], arrow_canvas_size, arrow_color, arrow_width)
        else:
             left_arrow_canvas.create_text(arrow_canvas_size/2, arrow_canvas_size/2, text="N/A", fill="grey")


        # --- Grid de Referencia (Columna 1 - Centro) ---
        grid_frame = tk.Frame(middle_row_frame) # Frame para etiqueta y canvas del grid
        grid_frame.grid(row=0, column=1, sticky="nsew") # Centrado en la columna 1
        # Centrar elementos dentro de grid_frame usando pack (o grid interno)
        tk.Label(grid_frame, text="Posición Analizada:", font=("Arial", 10)).pack()
        grid_canvas = tk.Canvas(grid_frame, width=grid_canvas_size, height=grid_canvas_size, bg="white", highlightthickness=0)
        grid_canvas.pack()
        # Dibujar grid y resaltado (sin cambios)
        step = grid_canvas_size / 3.0
        for j in range(1, 3):
            grid_canvas.create_line(j * step, 0, j * step, grid_canvas_size, fill=grid_line_color)
            grid_canvas.create_line(0, j * step, grid_canvas_size, j * step, fill=grid_line_color)
        if 1 <= cell_id <= 9:
            row_idx = (cell_id - 1) // 3
            col_idx = (cell_id - 1) % 3
            center_x = (col_idx + 0.5) * step; center_y = (row_idx + 0.5) * step
            radius = (step / 2.0) * grid_highlight_radius_factor
            grid_canvas.create_oval(center_x - radius, center_y - radius, center_x + radius, center_y + radius, fill=grid_highlight_color, outline="")
        else:
            grid_canvas.create_text(grid_canvas_size/2, grid_canvas_size/2, text="?", font=("Arial", int(step)), fill="red")

        # --- Flecha Derecha (Columna 2) ---
        right_arrow_canvas = tk.Canvas(middle_row_frame, width=arrow_canvas_size, height=arrow_canvas_size, bg="white", highlightthickness=1, highlightbackground="lightgrey")
        right_arrow_canvas.grid(row=0, column=2, padx=(5, 0), sticky="w") # Alinear a la izquierda de su celda
        if num_eyes > 1 and arrow_vectors[1] is not None:
            self._draw_arrow_on_canvas(right_arrow_canvas, arrow_vectors[1], arrow_canvas_size, arrow_color, arrow_width)
        else:
            # Si no hay ojo derecho o no hay vector, mostrar N/A
            right_arrow_canvas.create_text(arrow_canvas_size/2, arrow_canvas_size/2, text="N/A", fill="grey")


        # --- Botón Cerrar (FILA INFERIOR) ---
        button_quit = ttk.Button(main_frame, text="Cerrar Vista", command=top.destroy)
        # Colocar el botón en la fila 2, abarcando las 2 columnas principales
        button_quit.grid(row=2, column=0, columnspan=2, pady=(15, 10)) # Más espacio arriba

        # --- Centrar Ventana (sin cambios) ---
        top.update_idletasks()
        # ... (código de geometría sin cambios) ...
        #top.geometry(f"+{x}+{y}")

        top.wait_window()

    def _clear_widgets(self):
        """Elimina todos los widgets hijos de la ventana root."""
        for widget in self.root.winfo_children():
            widget.destroy()

    def update_grid_button_status(self):
        """Actualiza texto/estado del botón en la grilla principal."""
        if self.current_cell_id and self.current_cell_id in self.buttons:
            btn = self.buttons[self.current_cell_id]
            if btn.winfo_exists(): # Comprobar si el botón aún existe
                 btn.config(text=f"Celda {self.current_cell_id}\nFoto Tomada")
                 # btn.config(state=tk.DISABLED) # Opcional: deshabilitar
        else:
             print(f"Advertencia: Botón para celda {self.current_cell_id} no encontrado/válido.")

    def restart_test(self):
        """Limpia el estado y vuelve a la pantalla inicial."""
        print("Reiniciando Test...")
        user_confirm = messagebox.askyesno("Confirmar Reinicio", "¿Estás seguro de que quieres reiniciar el test?\nSe perderán las fotos y resultados actuales.")
        if user_confirm:
            self.photos = {}
            self.results = {}
            self.detailed_results = {}
            self.current_cell_id = None
            self.captured_frame = None
            self.release_camera_if_active()
            # No borramos la carpeta 'imagenes' aquí, solo al cerrar la app.
            self.create_start_screen()
        else:
            print("Reinicio cancelado.")

    def release_camera_if_active(self):
        """Detiene el feed, cancela 'after' y libera la cámara."""
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            print("Liberando cámara...")
            try: self.cap.release()
            except Exception as e: print(f"Error al liberar cámara: {e}")
            self.cap = None
        apagar_led()  # Asegurar que LED se apague aunque se salga abruptamente

    def cleanup_imagenes_folder(self):
        """Elimina todos los archivos dentro de una carpeta"""
        print(f"Iniciando limpieza de la carpeta: {'imagenes'}")
        if os.path.isdir('imagenes'):
            for filename in os.listdir('imagenes'):
                file_path = os.path.join('imagenes', filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path); print(f"  Eliminado: {filename}")
                    # elif os.path.isdir(file_path): shutil.rmtree(file_path) # Evitar borrar carpetas por ahora
                except Exception as e: print(f"  ERROR al eliminar {file_path}: {e}")
            print(f"Limpieza de {'imagenes'} completada.")
        else: print(f"Carpeta no existe, no se necesita limpieza.")

if __name__ == "__main__":
    # Carpeta de prueba
    if not os.path.isdir("Aron"):
         messagebox.showerror("Error Crítico", "Ingresa una carpeta valida")

    main_root = tk.Tk()
    app = TestHirschbergApp(main_root)

    # Cerrar aplicacion
    def on_closing():
        app.release_camera_if_active()
        app.cleanup_imagenes_folder()
        main_root.destroy()

    main_root.protocol("WM_DELETE_WINDOW", on_closing)
    main_root.mainloop()