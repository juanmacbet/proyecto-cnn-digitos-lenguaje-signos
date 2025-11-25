import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Cargar el modelo (ya subido al Space)
modelo = load_model("model_cnn_numeros.h5")

# Clases del modelo
class_names = [str(i) for i in range(10)]

# Función de predicción
def predecir_numero(img):
    # Escala de grises
    img_proc = img.convert("L")
    # Redimensionar a 100x100
    img_proc = img_proc.resize((100, 100))
    # Convertir a array y normalizar
    img_array = image.img_to_array(img_proc) / 255.0
    # Añadir dimensión de batch
    img_array = np.expand_dims(img_array, axis=0)

    # Predecir
    pred = modelo.predict(img_array)
    numero = int(np.argmax(pred))
    confianza = float(pred[0][numero])

    return f"Número predicho: {class_names[numero]} (confianza: {confianza:.2f})"

# Interfaz Gradio
iface = gr.Interface(
    fn=predecir_numero,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Reconocimiento de Números en Lenguaje de Signos",
    description="Sube una imagen de una mano representando un número en lenguaje de signos y el modelo predecirá de qué número se trata."
)

# Lanzar la app
iface.launch()
