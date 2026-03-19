# Clasificador Chihuahua vs Muffin

Clasificador binario de imágenes que distingue entre chihuahuas y muffins, construido con una CNN personalizada (sin transfer learning) y desplegado como una aplicación web interactiva con Streamlit.

## Demo

[Pruébalo aquí](https://chihuahua-vs-mffn.streamlit.app)

---

## Modelo

- **Arquitectura:** CNN personalizada con `SeparableConv2D` + `GlobalAveragePooling2D`
- **Parámetros:** ~47,905 (187 KB)
- **Accuracy en test:** 89.58% sobre 144 imágenes
- **Tiempo de entrenamiento:** ~2 minutos en CPU

### Técnicas utilizadas

- `SeparableConv2D`: convoluciones eficientes con ~8x menos parámetros que una Conv2D estándar
- `BatchNormalization`: estabiliza el entrenamiento
- `GlobalAveragePooling2D`: reduce parámetros y mejora la generalización
- `Dropout` (0.25, 0.3, 0.5) + Regularización L2
- `Data Augmentation`: rotación, zoom, flip y contraste aleatorios
- `EarlyStopping` + `ReduceLROnPlateau`

---

## Dataset

| Elemento | Descripción |
|---|---|
| Total imágenes | 1088 |
| Clases | Chihuahua / Muffin |
| Tamaño de imagen | 96 x 96 px |
| Train / Val / Test | 800 / 144 / 144 |

---

## Estructura del repositorio

    app.py                         # Interfaz web de Streamlit
    modelo_chihuahua_muffin.keras  # Modelo entrenado exportado
    requirements.txt               # Dependencias del proyecto
    README.md

---

## Ejecución local

    git clone https://github.com/TU-USUARIO/TU-REPO.git
    cd TU-REPO
    pip install -r requirements.txt
    streamlit run app.py

---

## Dependencias

    streamlit>=1.32.0
    tensorflow>=2.16.0
    numpy>=1.26.0
    Pillow>=10.0.0
    streamlit-paste-button
