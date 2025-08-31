import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from tensorflow.keras import datasets, models, layers
from PIL import Image, ImageOps

st.set_page_config(page_title="Fashion-MNIST Clothing Type Classifier", layout="wide")

# ---------- Constants ----------
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = "fashion_mnist_cnn.h5"
EPOCHS = 30  # fixed (no UI)
VAL_SPLIT = 0.20
BATCH_SIZE = 64
SEED = 42

LONG_LABELS = {
    'T-shirt/top': 'T-shirt/top',
    'Trouser': 'Trouser',
    'Pullover': 'Pullover',
    'Dress': 'Dress (includes frocks, chudidhars, gowns)',
    'Coat': 'Coat',
    'Sandal': 'Sandal',
    'Shirt': 'Shirt',
    'Sneaker': 'Sneaker',
    'Bag': 'Bag',
    'Ankle boot': 'Ankle boot'
}

# ---------- Data ----------
@st.cache_data(show_spinner=True)
def load_fmnist():
    (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
    train_images = (train_images.astype("float32") / 255.0)[..., None]
    test_images  = (test_images.astype("float32")  / 255.0)[..., None]
    return train_images, train_labels, test_images, test_labels

train_images, train_labels, test_images, test_labels = load_fmnist()

# ---------- Model ----------
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

@st.cache_resource(show_spinner=True)
def get_or_train_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    tf.keras.utils.set_random_seed(SEED)
    model = build_model()
    with st.spinner("Training model once on Fashion-MNIST…"):
        model.fit(
            train_images, train_labels,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_split=VAL_SPLIT, verbose=0
        )
    model.save(MODEL_PATH)
    return model

model = get_or_train_model()

# ---------- Preprocess uploaded image to FMNIST style ----------
def to_fmnist_tensor(file) -> np.ndarray:
    """
    Returns two tensors (no_invert, invert), each shape (1,28,28,1) in [0,1].
    We will run the model on both and pick the one with higher top probability.
    """
    img = Image.open(file).convert("L")        # grayscale
    # center-crop to square
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    # resize to 28×28 (FMNIST size)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    # auto-contrast to pop the clothing silhouette
    img = ImageOps.autocontrast(img)

    arr = np.array(img).astype("float32") / 255.0
    arr = arr[..., None]
    arr_no_inv = arr[None, ...]         # (1,28,28,1)
    arr_inv    = (1.0 - arr)[None, ...] # inverted (dark background, bright item)
    return arr_no_inv, arr_inv

def best_probs_for_two(t1, t2):
    p1 = model.predict(t1, verbose=0)[0]
    p2 = model.predict(t2, verbose=0)[0]
    # choose the set with larger top-1 confidence
    return (p1 if p1.max() >= p2.max() else p2)

def bar_plot(probs):
    fig = plt.figure(figsize=(8, 4))
    x = np.arange(NUM_CLASSES)
    plt.bar(x, probs)
    plt.xticks(x, [str(i) for i in range(NUM_CLASSES)])
    plt.xlabel("Class ID")
    plt.ylabel("Probability")
    plt.title("Class probabilities")
    plt.ylim(0, 1)
    fig.tight_layout()
    return fig

# ---------- UI (matches your screenshot layout) ----------
st.markdown("# 🧥 Fashion-MNIST Clothing Type Classifier")
st.markdown("*Upload a clothing image (PNG/JPG)*")
file = st.file_uploader(" ", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if file is not None:
    # show the uploaded image (as-is)
    st.image(file, use_column_width=True)

    # preprocess -> predict (choose better of normal vs inverted)
    x_no_inv, x_inv = to_fmnist_tensor(file)
    probs = best_probs_for_two(x_no_inv, x_inv)

    pred_id = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_id]
    conf = float(probs[pred_id])

    st.markdown(f"*Prediction:* {LONG_LABELS[pred_name]}")
    st.markdown(f"*Confidence:* *{conf*100:.2f}%*")

    # probabilities bar chart (class ids on x-axis)
    st.pyplot(bar_plot(probs), clear_figure=True)