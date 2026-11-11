# Leaf Classifier (Mobile ONNX)

Live web application for real-time leaf classification using an ONNX MobileNetV3 model in the browser via `onnxruntime-web`.

## 🌐 Live Application URL

👉 **[https://jkrejcikova.github.io/mlr-ex08/](https://jkrejcikova.github.io/mlr-ex08/)**

---

### 📱 Scan to Open on Mobile

<p align="center">
  <a href="https://jkrejcikova.github.io/mlr-ex08/">
    <img src="qr.svg" width="200" height="200" alt="QR Code to Live App" />
  </a>
  <br>
  <strong><a href="https://jkrejcikova.github.io/mlr-ex08/">https://jkrejcikova.github.io/mlr-ex08/</a></strong>
</p>

---

## 🍃 Model Overview

- **Model File**: `leaf_model.onnx`
- **Architecture**: MobileNetV3
- **Input Size**: `96 × 96` pixels
- **Input Tensor Shape**: `[1, 3, 96, 96]` (NCHW format, Float32)
- **Target Classes**:
  - `apple` (🍎)
  - `cherry` (🍒)
  - `chestnut` (🌰)
  - `maple` (🍁)

### Preprocessing Pipeline (from `model_info.json`)
1. **Central Square Crop**: Extract the centered square region $\min(W, H) \times \min(W, H)$ from the camera frame or uploaded photo.
2. **Resize**: Scale down to $96 \times 96$ pixels.
3. **RGB Normalization**: Scale pixel intensities from `0..255` to `0.0..1.0`.
4. **Standardization**:
   - $\text{Mean} = [0.485, 0.456, 0.406]$
   - $\text{Std} = [0.229, 0.224, 0.225]$
   - Channel formula: $x_{\text{norm}} = (x - \text{mean}_c) / \text{std}_c$
5. **Softmax Output**: Model outputs raw logits; softmax converts logits into probability distribution summing to 1.0.

---

## 📲 Safari on iPhone & Mobile Support

- Camera stream accessed with `{ video: { facingMode: { ideal: 'environment' } } }` for rear camera selection.
- `<video>` configured with `playsinline`, `webkit-playsinline`, `autoplay`, and `muted` to prevent fullscreen takeover on iOS Safari.
- ONNX Runtime Web configured with single-threaded WASM (`ort.env.wasm.numThreads = 1`) to guarantee compatibility on iOS Safari without requiring `Cross-Origin-Opener-Policy` headers.
- Photo upload fallback for environments without camera access or permission restrictions.

---

## 🧪 Verification Check

The app includes an automated verification check against `check_image.jpg`:
- **Expected Probabilities** (`check_expected.json`):
  - `cherry`: `0.9999688` (99.997%)
  - `chestnut`: `2.647e-5`
  - `apple`: `2.699e-6`
  - `maple`: `1.992e-6`
- Click the **Verify check_image.jpg** button in the web interface to run this test at any time.
