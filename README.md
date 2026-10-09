# InternshipCollab

Machine learning and data visualization work from my summer 2025 Python Developer internship at the University of Arkansas at Little Rock.

| Project | File | What it does |
|---|---|---|
| Parking space classifier | `image-classification-python-scikit-learn-master (2).zip` | SVM that labels a parking-space image as **empty** or **occupied** |
| Cats vs. dogs CNN (my build) | `Untitled4.ipynb` | Small Keras CNN trained from scratch in Google Colab |
| Cats vs. dogs CNN (deeper model) | `cats_vs_dogs_image_classification_using_cnn_95 (2).ipynb` | 4-block CNN with augmentation, callbacks, and full evaluation |
| Data visualization workshop | `DataVisualisationTask.ipynb` | Fill-in-the-blank teaching worksheet on bar charts and misleading axes |
| Write-up | `Summary Code (2) (1).pdf` | Step-by-step explanation of the parking classifier and its real-world use |

---

## 1. Parking space classifier (scikit-learn)

Detects whether a parking space is empty or occupied, the building block of a parking lot occupancy counter.

**How it works**
1. Reads images from two folders, `empty/` and `not_empty/`.
2. Resizes each image to 15×15 pixels and flattens it into a 675-value vector (15 × 15 × 3 RGB).
3. Splits the data 80/20 into training and test sets, stratified so both keep the same empty/occupied ratio.
4. Trains a Support Vector Machine (`SVC`) and uses `GridSearchCV` to tune `C` and `gamma`.
5. Prints test accuracy and saves the best model to `model.p` with `pickle`.

The included `model.p` is the trained model (best parameters: `C=10`, `gamma=0.01`).

**Run it**
```bash
unzip "image-classification-python-scikit-learn-master (2).zip"
cd image-classification-python-scikit-learn-master
pip install scikit-learn scikit-image numpy
# Set input_dir in main.py to your dataset folder (containing empty/ and not_empty/)
python main.py
```

**Use the saved model**
```python
import pickle
from skimage.io import imread
from skimage.transform import resize

model = pickle.load(open("model.p", "rb"))
img = resize(imread("spot.jpg"), (15, 15)).flatten().reshape(1, -1)
print("occupied" if model.predict(img)[0] == 1 else "empty")
```

The parking-space image dataset is not included in this repo. The project is based on the Computer Vision Engineer tutorial linked in the inner README.

---

## 2. Cats vs. dogs CNN: my Colab build (`Untitled4.ipynb`)

A convolutional neural network I built and trained in Google Colab on the 25,000-image Kaggle [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) dataset.

- **Data:** downloaded with the Kaggle API, then randomly split 80/20 into `train/` and `val/` folders (20,000 / 5,000 images).
- **Model:** 2 convolution + max-pooling blocks → Dense(512) → Dropout(0.5) → sigmoid output.
- **Training:** 150×150 images, horizontal-flip augmentation, Adam optimizer, binary cross-entropy, 10 epochs.
- **Result:** about **97% training accuracy** and **80% validation accuracy**. The gap shows overfitting, which the deeper model below addresses.
- Plots accuracy and loss curves, saves the model (`.h5`), and predicts on a single image.

**Run it:** open in Colab, upload your `kaggle.json` API key to `/content`, and run the cells top to bottom.

---

## 3. Cats vs. dogs CNN: deeper model (`cats_vs_dogs_..._95.ipynb`)

A more advanced pipeline, adapted from a public Kaggle notebook, that I studied to learn how to fix the overfitting in my first model.

- **Split:** stratified 80/10/10 train/validation/test.
- **Augmentation:** rotation, zoom, shear, shifts, and horizontal flip.
- **Model:** 4 convolution blocks (32→64→128→256 filters), each with BatchNormalization, MaxPooling, and Dropout(0.2), then Dense(512) and a 2-class softmax output.
- **Callbacks:** `ReduceLROnPlateau` and `EarlyStopping`.
- **Results (128×128 input):** 95.7% validation accuracy, **95.1% test accuracy**, with precision and recall of 0.95–0.96 for both classes.
- Includes a classification report, confusion matrix, and Kaggle submission predictions.

**Run it:** on Kaggle with the Dogs vs. Cats competition data attached (paths expect `/kaggle/input/dogs-vs-cats/`).

---

## 4. Data visualization workshop (`DataVisualisationTask.ipynb`)

A beginner activity I helped run on using charts for business decisions:
- Build a revenue-by-product bar chart.
- Change the y-axis limit (200 vs. 100) to see how scale can make the same data look more or less dramatic.

This is a **template**. The blanks (`____`) are meant for participants to fill in, so the notebook raises an error until they do.

---

## Tech stack

Python · scikit-learn · scikit-image · NumPy · pandas · TensorFlow / Keras · Matplotlib · Seaborn · Google Colab · Kaggle

## Author

**Ryan Kim**, B.S. Computer Science & Business Administration, Northeastern University
