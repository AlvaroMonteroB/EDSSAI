import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Cargar la imagen desde el archivo
def mostrar_imagen(nombre_archivo):
    imagen = mpimg.imread(nombre_archivo)
    plt.figure(figsize=(6, 6))
    plt.imshow(imagen)
    plt.axis('off')  # Ocultar ejes
    plt.title(f'Imagen: {nombre_archivo}')
    plt.show()

# Mostrar la imagen 'yp1.jpg'
mostrar_imagen('yp1.jpg')