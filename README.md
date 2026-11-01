# Portrait-to-Sketch Conditional GAN (Pix2Pix)

This project implements a **Conditional Generative Adversarial Network (cGAN)** based on the **Pix2Pix** architecture to translate color portrait photographs into realistic face sketches. The model is built using **TensorFlow / Keras**.

---

## 🚀 Project Overview

* **Generator:** U-Net architecture utilizing an encoder-decoder structure with skip connections for high-quality image-to-image translation.
* **Discriminator:** PatchGAN classifier that evaluates whether local N × N image patches are real or fake.
* **Dataset:** Trained and tested on face databases such as the **CUHK Face Sketch Database (CUFS)**.

---

## 📁 Project Structure

```text
├── buildGenerator.py       # U-Net Generator architecture
├── buildDiscriminator.py   # PatchGAN Discriminator architecture
├── runModel.py             # Data pipelines, loss functions, and training loop
├── _config.yml             # Jekyll theme configuration for GitHub Pages
└── README.md
```

---

## 🧠 Architecture Breakdown

This model uses a conditional GAN setup where the network learns a mapping from an input image (color portrait) to an output image (sketch).

### 1. U-Net Generator
Unlike standard autoencoders, the generator uses a **U-Net** configuration to preserve spatial details:
* **Encoder:** Series of Downsampling blocks (Convolution -> BatchNormalization -> LeakyReLU).
* **Decoder:** Series of Upsampling blocks (Transposed Convolution -> BatchNormalization -> Dropout -> ReLU).
* **Skip Connections:** Concatenates features directly across the U-Net bottle-neck to link structural textures from the encoder directly to the decoder.

### 2. PatchGAN Discriminator
Rather than outputting a single scalar value for the entire image, the **PatchGAN** discriminator:
* Evaluates the image at the scale of local 70 × 70 patches.
* Classifies each patch as "Real" or "Fake".
* Penalizes high-frequency structural errors effectively, forcing the generator to produce sharp edges.

### 3. Loss Functions
The network is optimized using a combined loss function:
\[\mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_{L1}(G)\]
* **Adversarial Loss:** Binary Cross-Entropy ensuring the sketches look authentic.
* **L₁ Loss:** Measures the absolute pixel-wise difference between the generated sketch and the ground truth to enforce structural layout consistency (λ = 100).

---

## 🛠️ Requirements & Dependencies

Make sure you have the following libraries installed before running the project:
* **Python 3.8+**
* **TensorFlow 2.x**
* **Matplotlib**
* **Pillow (PIL)**
* **Natsort**

Install all dependencies quickly via pip:
```bash
pip install tensorflow matplotlib pillow natsort
```

---

## 📊 Dataset Setup

1. Download or prepare your image dataset (e.g., [CUHK Face Sketch Database (CUFS)](http://cuhk.edu.hk)).
2. Ensure your dataset contains aligned pairs of color photographs and sketches.
3. Update the `PATH` variable in `runModel.py` to point to your data directory:
   ```python
   PATH = '../path/to/data'
   ```

---

## 🏃‍♂️ Usage

To start training the model, run the main execution script:
```bash
python runModel.py
```

### What happens during execution:
* **Data Processing:** Loads and preprocesses image pairs (photos and corresponding sketches).
* **Initialization:** Instantiates the U-Net generator and PatchGAN discriminator models.
* **Training:** Runs training epochs with defined adversarial (GAN) and L₁ pixel-wise losses.
* **Export:** Saves model checkpoints and exports the final trained model (`.h5` format).

---

## 📈 Results & Visualizations

During training, performance is measured by tracking both Generator loss (L₁ + GAN objectives) and Discriminator loss over successive epochs. 

## 📜 References

* **Image-to-Image Translation with Conditional Adversarial Networks** (Isola et al., CVPR 2017)
* [How to Develop a Pix2Pix GAN for Image-to-Image Translation](https://machinelearningmastery.com) - Machine Learning Mastery
* [TensorFlow Pix2Pix Tutorial](https://tensorflow.org)

---

## ⚖️ License

This project is licensed under the **MIT License**.