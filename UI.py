import json
import numpy as np
import tensorflow as tf
import gradio as gr

from clean import limpiar_texto

modelo = tf.keras.models.load_model("modelo_dl.keras")

with open("preprocessing_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

best_threshold = float(metadata["best_threshold"])
max_tokens = int(metadata["max_tokens"])
sequence_length = int(metadata["sequence_length"])
vocabulario = metadata["vocabulario"]

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length,
    vocabulary=vocabulario
)

theme = gr.themes.Base(
    primary_hue="teal",
    secondary_hue="orange",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#0F1115",
    body_text_color="#F8FAFC",
    body_text_color_subdued="#D1D5DB",

    background_fill_primary="#151A22",
    background_fill_secondary="#1A1F29",

    block_background_fill="#151A22",
    block_border_color="#264653",
    block_label_text_color="#F8FAFC",
    block_title_text_color="#F8FAFC",

    input_background_fill="#1A1F29",
    input_border_color="#264653",
    input_border_color_focus="#2A9D8F",
    input_placeholder_color="#D1D5DB",

    button_primary_background_fill="#264653",
    button_primary_background_fill_hover="#2A9D8F",
    button_primary_text_color="#F8FAFC",
    button_primary_text_color_hover="#111827",
    button_primary_border_color="#264653",

    button_secondary_background_fill="#151A22",
    button_secondary_background_fill_hover="#264653",
    button_secondary_text_color="#F8FAFC",
    button_secondary_border_color="#264653",

    border_color_primary="#264653",
    color_accent="#2A9D8F",
    color_accent_soft="#E9C46A"
)

def predecir_sarcasmo(texto):
    if texto is None or texto.strip() == "":
        return "Escribe un titular en inglés.", ""

    texto_limpio = limpiar_texto(texto)
    x_vec = vectorizer(np.array([texto_limpio])).numpy()

    score_sarcasmo = float(modelo.predict(x_vec, verbose=0)[0][0])
    score_no_sarcasmo = 1 - score_sarcasmo

    pred = 1 if score_sarcasmo >= best_threshold else 0
    etiqueta = "Sarcasmo" if pred == 1 else "No sarcasmo"

    detalle = (
        f"Decisión: {etiqueta}\n"
        f"Puntaje de sarcasmo: {score_sarcasmo:.2%}\n"
        f"Puntaje de no sarcasmo: {score_no_sarcasmo:.2%}"
    )

    return etiqueta, detalle

ejemplos_sarcasticos = [
    ["local man wins award for doing absolutely nothing"],
    ["government announces new plan to fix traffic by adding more traffic"],
    ["scientists discover that being tired is caused by not sleeping enough"],
    ["city opens tenth luxury apartment building for people who cannot afford rent"],
    ["company celebrates employee wellness by scheduling mandatory 7 am meeting"],
    ["usda secretary rings nationwide dinner bell for y'all to get in here"],
    ["john ashcroft silences reporters with warning shot"],
    ["friend has some jerky in clear, unlabeled bag for you to try"],
    ["seedless watermelon coming to grips with fact it'll never be able to have kids"],
    ["astronomers discover previously unknown cluster of nothingness in deep space"]
]

ejemplos_no_sarcasticos = [
    ["president signs new education reform bill"],
    ["scientists discover new species of deep sea fish"],
    ["local school opens new science laboratory for students"],
    ["weather forecast predicts heavy rain across the region"],
    ["hospital launches vaccination campaign for children"],
    ["woman stops alleged bank robber by crashing into him"],
    ["will smith and jada pinkett smith look incredible as always at the 2016 golden globes"],
    ["another way companies make it harder for new mothers"],
    ["two non-binary college activists on creating space for themselves on campus"],
    ["man rides a horse into taco bell, and the internet is freaking out"]
]

with gr.Blocks(theme=theme) as demo:
    with gr.Row(equal_height=True):
        # COLUMNA IZQUIERDA
        with gr.Column(scale=5):
            gr.Markdown("# Detector de sarcasmo con deep learning")
            gr.Markdown("Escribe un **titular** y el modelo predecirá si parece **sarcástico** o **no sarcástico**.")
            gr.Markdown("El modelo fue entrenado con headlines en inglés, así que funciona mejor con textos cortos de ese estilo.")

            entrada = gr.Textbox(
                label="Titular",
                placeholder="Escribe un titular en inglés aquí...",
                lines=4
            )

            boton = gr.Button("Analizar", variant="primary")

            salida_etiqueta = gr.Textbox(label="Resultado")
            salida_detalle = gr.Textbox(label="Detalle", lines=6)

            boton.click(
                fn=predecir_sarcasmo,
                inputs=entrada,
                outputs=[salida_etiqueta, salida_detalle]
            )

        # COLUMNA DERECHA
        with gr.Column(scale=6):
            gr.Markdown("# Ejemplos de prueba")

            with gr.Group():
                gr.Markdown("### Sarcásticos")
                gr.Examples(
                    examples=ejemplos_sarcasticos,
                    inputs=entrada
                )

            with gr.Group():
                gr.Markdown("### No sarcásticos")
                gr.Examples(
                    examples=ejemplos_no_sarcasticos,
                    inputs=entrada
                )

demo.launch()