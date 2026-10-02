"""
Test verification script for Model D integration
Validates:
1. Loading densenet121_frequency_v5.h5
2. Architecture integrity
3. Exact frequency preprocessing (Gaussian sigma=1.0)
4. Inference output probabilities and 6-class mapping
5. Grad-CAM generation
"""

import os
import sys
import numpy as np
import cv2
import json
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = Path("experiments/densenet_frequency_v5/densenet121_frequency_v5.h5")
CLASS_MAPPING = [
    "COVID-19",
    "Normal",
    "Pleural Effusion",
    "Pneumonia",
    "Pulmonary Nodule / Mass",
    "Tuberculosis"
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def apply_gaussian_lp(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def preprocess_cxr_image(img_bgr: np.ndarray):
    if len(img_bgr.shape) == 2:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    else:
        img = img_bgr.copy()

    # Step 2: BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 3: Gaussian blur (3,3), sigma=0.8
    img = cv2.GaussianBlur(img, (3, 3), 0.8)

    # Step 4: CIE LAB conversion & CLAHE on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_chan = np.ascontiguousarray(lab[:, :, 0], dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(l_chan)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Step 5: Frequency preprocessing (Gaussian sigma=1.0)
    img_filtered = apply_gaussian_lp(img, sigma=1.0)

    # Step 6: Lanczos-4 resize to 224x224
    img_resized = cv2.resize(img_filtered, (224, 224), interpolation=cv2.INTER_LANCZOS4)

    # Step 7: ImageNet normalization
    img_norm = (img_resized.astype(np.float32) / 255.0 - MEAN) / STD
    return img_norm, img_resized

def test_integration():
    print("Testing Model D Loading...")
    assert MODEL_PATH.exists(), f"Model path not found: {MODEL_PATH}"
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded: {model.name}")
    print(f"Layers: {[l.name for l in model.layers]}")
    assert len(model.layers) == 7

    # Dummy test image
    dummy_bgr = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    norm_img, display_img = preprocess_cxr_image(dummy_bgr)
    assert norm_img.shape == (224, 224, 3)
    assert display_img.shape == (224, 224, 3)

    # Inference test
    input_batch = np.expand_dims(norm_img, axis=0)
    probs = model.predict(input_batch, verbose=0)[0]
    print(f"Predicted raw probabilities: {probs}")
    assert len(probs) == 6
    assert abs(np.sum(probs) - 1.0) < 1e-4

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_MAPPING[pred_idx]
    conf = float(probs[pred_idx])
    print(f"Predicted class: {pred_class}, Confidence: {conf:.4f}")

    # Grad-CAM test
    print("Testing Grad-CAM generation...")
    base_layer = model.get_layer("densenet121")
    last_conv_layer = base_layer.get_layer("relu")
    base_grad_model = keras.Model(
        inputs=base_layer.inputs,
        outputs=[last_conv_layer.output, base_layer.output]
    )
    top_layers = [
        model.get_layer("gap"),
        model.get_layer("bn"),
        model.get_layer("dense_features"),
        model.get_layer("dropout"),
        model.get_layer("class_predictions")
    ]
    gap, bn, dense, drop, dense_1 = top_layers

    with tf.GradientTape() as tape:
        conv_outputs, _ = base_grad_model(input_batch)
        tape.watch(conv_outputs)
        x = gap(conv_outputs)
        x = bn(x, training=False)
        x = dense(x)
        x = drop(x, training=False)
        preds = dense_1(x)
        loss = preds[:, pred_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap_np = heatmap.numpy()
    assert heatmap_np.shape == (7, 7)
    print("Grad-CAM generation SUCCESS!")

    # Overlay generation
    heatmap_resized = cv2.resize(heatmap_np, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed = np.uint8(heatmap_color * 0.45 + display_img * 0.55)
    assert superimposed.shape == (224, 224, 3)
    print("Grad-CAM overlay SUCCESS!")
    print("\nALL INTEGRATION TESTS PASSED CLEANLY!")

if __name__ == "__main__":
    test_integration()
