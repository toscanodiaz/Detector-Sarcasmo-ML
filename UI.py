import joblib
import numpy as np
import tensorflow as tf
import gradio as gr

from clean import limpiar_texto

modelo = tf.keras.models.load_model("modelo_dl.keras")

with open("vocabulario.txt", "r", encoding="utf-8") as f:
    vocabulario = [line.strip() for line in f]

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=30,
    vocabulary=vocabulario
)

def predecir_sarcasmo(texto):
    if texto is None or texto.strip() == "":
        return "Escribe un titular en inglés.", ""

    texto_limpio = limpiar_texto(texto)
    x_vec = vectorizer(np.array([texto_limpio])).numpy()

    prob = float(modelo.predict(x_vec, verbose=0)[0][0])
    pred = 1 if prob >= 0.5 else 0

    etiqueta = "Sarcasmo" if pred == 1 else "No sarcasmo"
    detalle = (
        f"Predicción: {etiqueta}\n"
        f"Probabilidad: {prob:.4f}\n\n"

    )
    return etiqueta, detalle

ejemplos = [
    ["local man wins award for doing absolutely nothing"],
    ["government announces new plan to fix traffic by adding more traffic"],
    ["scientists discover that being tired is caused by not sleeping enough"],
    ["city opens tenth luxury apartment building for people who cannot afford rent"],
    ["company celebrates employee wellness by scheduling mandatory 7 am meeting"],
    ["president signs new education reform bill"],
    ["scientists discover new species of deep sea fish"],
    ["local school opens new science laboratory for students"],
    ["weather forecast predicts heavy rain across the region"],
    ["hospital launches vaccination campaign for children"]
]

with gr.Blocks() as demo:
    gr.Markdown("# Detector de sarcasmo con deep learning")
    gr.Markdown("Escribe un **titular** y el modelo predecirá si parece **sarcástico** o **no sarcástico**.")
    gr.Markdown("El modelo fue entrenado con headlines en inglés, así que funciona mejor con textos cortos de ese estilo")

    entrada = gr.Textbox(
        label="Titular",
        placeholder="Escribe un titular en inglés aquí...",
        lines=4
    )

    boton = gr.Button("Analizar")
    salida_etiqueta = gr.Textbox(label="Resultado")
    salida_detalle = gr.Textbox(label="Detalle", lines=5)

    boton.click(
        fn=predecir_sarcasmo,
        inputs=entrada,
        outputs=[salida_etiqueta, salida_detalle]
    )

    gr.Examples(
        examples=ejemplos,
        inputs=entrada
    )

demo.launch()