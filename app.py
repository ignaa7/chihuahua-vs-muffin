import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_paste_button import paste_image_button


# ── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(
    page_title="Muffin vs Chihuahua",
    page_icon="🐶",
    layout="centered"
)


# ── Custom CSS for a Premium Look ──────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');


    html, body, [class*="st-"] {
        font-family: 'Outfit', sans-serif;
    }


    .main {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        color: #f0f0f0;
    }


    .stApp {
        background: transparent;
    }


    /* Glassmorphism card */
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-top: 2rem;
        transition: transform 0.3s ease;
    }


    .prediction-card:hover {
        transform: translateY(-5px);
    }


    h1 {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        text-align: center;
        font-size: 3.5rem !important;
    }


    .stButton>button {
        background: linear-gradient(90deg, #e52e71, #ff8a00);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }


    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }


    /* Subtitle styling */
    .stWrite {
        text-align: center;
        color: #aaa;
    }


    .stProgress > div > div > div {
        background-image: linear-gradient(to right, #e52e71, #ff8a00);
    }

[data-testid="collapsedControl"] {
    display: none !important;
}
button[data-testid="stBaseButton-headerNoPadding"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# ── Cargar modelo (cacheado para no recargarlo en cada interacción) ──────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("modelo_chihuahua_muffin.keras")


model = load_model()
clases = ["Chihuahua", "Muffin"]  # ajusta el orden según tu entrenamiento


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🐶 Muffin vs Chihuahua")

# ← FIX 2: Descripción centrada y alineada verticalmente con el título
st.markdown(
    "<p style='text-align:center; color:#aaa; margin-top:-1rem;'>"
    "Sube una imagen y el modelo decidirá si es un <strong>Chihuahua</strong> o un <strong>Muffin</strong>."
    "</p>",
    unsafe_allow_html=True
)


# ── Entrada de imagen: Subir o Pegar ──────────────────────────────────────────
_, center_col, _ = st.columns([1, 2, 1])


with center_col:
    st.markdown("<div style='text-align: center'>", unsafe_allow_html=True)
    archivo = st.file_uploader("📤 Subir imagen", type=["jpg", "jpeg", "png"])
    
    st.write("**O pega desde el portapapeles:**")
    archivo_pegado = paste_image_button(
        label="📋 Pegar imagen", 
        background_color="#e52e71", 
        hover_background_color="#ff8a00",
        errors="ignore"
    )
    st.markdown("</div>", unsafe_allow_html=True)


img_original = None


if archivo is not None:
    img_original = Image.open(archivo).convert("RGB")
elif archivo_pegado is not None and archivo_pegado.image_data is not None:
    img_original = archivo_pegado.image_data.convert("RGB")


if img_original is not None:
    
    # Mostrar imagen original (sin redimensionar) con buen aspecto
    st.image(img_original, caption="Imagen subida", use_container_width=True)


    # Preprocesado para el modelo (96x96 como fue entrenado)
    img_resized = img_original.resize((96, 96), Image.Resampling.BILINEAR)
    arr = np.expand_dims(np.array(img_resized).astype(np.float32), axis=0)


    # Predicción
    with st.spinner("🍳 Horneando muffins o analizando perretes..."):
        pred = model.predict(arr)[0][0]


    p_muffin = float(pred)
    p_chihuahua = 1.0 - p_muffin
    
    if p_muffin > 0.5:
        clase_pred = "Muffin"
        emoji = "🧁"
        confianza = p_muffin
        color = "#ff8a00"
    else:
        clase_pred = "Chihuahua"
        emoji = "🐶"
        confianza = p_chihuahua
        color = "#e52e71"


    st.markdown(f"""
    <div class="prediction-card">
        <h2 style='text-align: center; color: {color}; margin-top: 0;'>{emoji} ¡Detectado!</h2>
        <div style='text-align: center; font-size: 2.5rem; font-weight: 600; margin-bottom: 1rem;'>
            Es un **{clase_pred}**
        </div>
        <div style='text-align: center; font-size: 1.2rem; color: #888;'>
            Confianza del <strong>{confianza:.2%}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Histograma de Probabilidad")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"🐶 **Chihuahua**: {p_chihuahua:.2%}")
        st.progress(p_chihuahua)
    with col2:
        st.write(f"🧁 **Muffin**: {p_muffin:.2%}")
        st.progress(p_muffin)


# ── Sidebar & Info ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Sobre este modelo")
    st.info("""
    Este modelo es una **Red Neuronal Convolucional (CNN)** compacta entrenada para distinguir entre Chihuahuas y Muffins.
    
    **Características:**
    - Arquitectura compacta (140k parámetros)
    - Entrada: 96x96 px (RGB)
    - Salida: Sigmoide (Binaria)
    """)
    
    st.divider()
    st.markdown("### 💡 ¿Sabías que...?")
    st.caption("A veces, los ojos de un Chihuahua y los arándanos de un muffin son visualmente idénticos para una IA básica. ¡Este modelo intenta ser más listo!")


# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem; margin-top: 2rem;'>
    Desarrollado con ❤️ y Streamlit • 2026
</div>
""", unsafe_allow_html=True)
