import json
import numpy as np
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocesamiento import preprocesar_mejorado

st.set_page_config(page_title="Sentimiento en noticias financieras", page_icon="📈", layout="centered")

@st.cache_resource
def cargar():
    meta = json.load(open("meta.json", encoding="utf-8"))
    modelo = keras.models.load_model("modelo_final.keras")
    if meta["tipo"] == "lstm":
        tok = tokenizer_from_json(open("tokenizer.json", encoding="utf-8").read())
        return meta, modelo, tok, None
    npz = np.load("vectores.npz", allow_pickle=True)
    return meta, modelo, None, dict(zip(npz["palabras"], npz["vectores"]))

def predecir(texto, meta, modelo, tok, vectores):
    procesado = preprocesar_mejorado(texto)
    if meta["tipo"] == "lstm":
        x = pad_sequences(tok.texts_to_sequences([procesado]), padding="post", maxlen=meta["maxlen"])
    else:
        vs = [vectores[w] for w in procesado.split() if w in vectores]
        x = (np.mean(vs, axis=0) if vs else np.zeros(100, dtype="float32")).reshape(1, -1)
    return procesado, modelo.predict(x, verbose=0)[0]

meta, modelo, tok, vectores = cargar()
NOMBRES = {"negative": "Negativo", "neutral": "Neutro", "positive": "Positivo"}
EJEMPLOS = {
    "Utilidades en aumento": "The company reported a 15 % increase in quarterly operating profit compared with last year .",
    "Ventas a la baja": "Net sales fell to EUR 48 million from EUR 61 million , and the company issued a profit warning .",
    "Hecho informativo": "The annual general meeting will be held on 15 April at the company headquarters .",
    "Con negación (caso difícil)": "The company did not meet its sales target for the second quarter .",
}

st.title("📈 Sentimiento en noticias financieras")
st.caption(f"Modelo: {meta['nombre']} · F1 macro en prueba: {meta['f1_macro_prueba']:.3f} · "
           "Entrenado con Financial PhraseBank (frases en inglés). Proyecto integrador - Redes Neuronales y Aprendizaje Profundo.")

ejemplo = st.selectbox("Prueba con un ejemplo o escribe tu propia frase:", ["(escribir frase)"] + list(EJEMPLOS))
texto = st.text_area("Frase financiera en inglés", value=EJEMPLOS.get(ejemplo, ""), height=110)

if st.button("Analizar sentimiento", type="primary") and texto.strip():
    procesado, probs = predecir(texto, meta, modelo, tok, vectores)
    clase = meta["clases"][int(np.argmax(probs))]
    st.subheader(f"Resultado: {NOMBRES[clase]}  ({probs.max():.1%})")
    st.bar_chart({NOMBRES[c]: float(p) for c, p in zip(meta["clases"], probs)}, horizontal=True)
    with st.expander("Texto procesado que ve el modelo"):
        st.code(procesado or "(vacío)")
    if not procesado:
        st.warning("La frase quedó vacía tras el preprocesamiento; el resultado no es confiable.")
st.info("Aviso: modelo educativo con pocos datos. No es una recomendación de inversión.")
