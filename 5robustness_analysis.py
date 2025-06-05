import pandas as pd
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew
import os
from time import time
import albumentations as A

# Configuración
DATA_DIR = 'features'  # Carpeta con CSV de características
IMG_DIR = 'preprocessed_images'  # Carpeta con imágenes preprocesadas
OUTPUT_DIR = 'results3'  # Carpeta para resultados
CLASSES = {
    1: 'TUMOR', 2: 'STROMA', 3: 'COMPLEX', 4: 'LYMPHO',
    5: 'DEBRIS', 6: 'MUCOSA', 7: 'ADIPOSE', 8: 'EMPTY'
}
FEATURES = [
    'glcm_contrast_1', 'glcm_contrast_3', 'glcm_contrast_5',
    'glcm_correlation_1', 'glcm_correlation_3', 'glcm_correlation_5',
    'glcm_energy_1', 'glcm_energy_3', 'glcm_energy_5',
    'glcm_homogeneity_1', 'glcm_homogeneity_3', 'glcm_homogeneity_5',
    'hist_r_mean', 'hist_r_variance', 'hist_r_skewness',
    'hist_g_mean', 'hist_g_variance', 'hist_g_skewness',
    'hist_b_mean', 'hist_b_variance', 'hist_b_skewness'
]

# Cargar modelo y scaler
with open(os.path.join(OUTPUT_DIR, 'best_model_RandomForest.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Cargar datos de prueba
df_test = pd.read_csv(os.path.join(DATA_DIR, 'test_features.csv'))
X_test = df_test[FEATURES].values
y_test = df_test['label'].values
image_paths = df_test['image_path'].values

# Verificar datos
print(f"Test: {len(df_test)} filas, NaN: {df_test.isna().sum().sum()}")
print("\nDistribución de clases en prueba:")
print(df_test['label'].value_counts(normalize=True) * 100)

# Normalizar características
X_test_scaled = scaler.transform(X_test)

# 1. Evaluación en conjunto de prueba
start_time = time()
y_pred = model.predict(X_test_scaled)
end_time = time()
print(f"\nTiempo de predicción (test): {end_time - start_time:.2f} s")

# Métricas por clase
print("\nReporte de clasificación (Random Forest - Test):")
print(classification_report(y_test, y_pred, target_names=CLASSES.values(), digits=3))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES.values(), yticklabels=CLASSES.values())
plt.title('Confusion Matrix - Random Forest (Test)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(OUTPUT_DIR, 'cm_RandomForest_Test_detailed.png'))
plt.close()

# 2. Validación cruzada (entrenamiento)
df_train = pd.read_csv(os.path.join(DATA_DIR, 'train_features.csv'))
X_train = df_train[FEATURES].values
y_train = df_train['label'].values
X_train_scaled = scaler.transform(X_train)
start_time = time()
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
end_time = time()
print(f"\nValidación cruzada (5-fold, F1-score macro):")
print(f"Media: {cv_scores.mean():.3f}, Desviación: {cv_scores.std():.3f}, Tiempo: {end_time - start_time:.2f} s")

# 3. Prueba con subconjunto visual (10 imágenes)
subset_idx = np.random.choice(len(X_test), 10, replace=False)
subset_X = X_test_scaled[subset_idx]
subset_y = y_test[subset_idx]
subset_paths = image_paths[subset_idx]
subset_pred = model.predict(subset_X)

plt.figure(figsize=(15, 10))
for i, (path, true_label, pred_label) in enumerate(zip(subset_paths, subset_y, subset_pred)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.title(f"True: {CLASSES[true_label]}\nPred: {CLASSES[pred_label]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'subset_predictions2.png'))
plt.close()

# 4. Prueba de robustez con ruido
transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)
])

def extract_features(img):
    # GLCM
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    gray_img = (gray_img * 255).astype(np.uint8)
    glcm = graycomatrix(gray_img, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    glcm_features = []
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        prop_values = graycoprops(glcm, prop)
        glcm_features.extend(np.mean(prop_values, axis=1))
    
    # Histogramas RGB
    hist_features = []
    for channel in range(3):
        channel_data = img[:, :, channel] / 255.0
        mean = np.mean(channel_data)
        variance = np.var(channel_data)
        skewness = 0.0 if np.var(channel_data) < 1e-6 else skew(channel_data.flatten())
        hist_features.extend([mean, variance, skewness])
    
    return np.array(glcm_features + hist_features)

noisy_X = []
noisy_y = []
for path, label in zip(image_paths[:100], y_test[:100]):  # 100 imágenes para prueba rápida
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    noisy_img = transform(image=img)['image']
    features = extract_features(noisy_img)
    noisy_X.append(features)
    noisy_y.append(label)

noisy_X = scaler.transform(np.array(noisy_X))
noisy_pred = model.predict(noisy_X)
noisy_f1 = f1_score(noisy_y, noisy_pred, average='macro')
print(f"\nF1-score con imágenes ruidosas (100 imágenes): {noisy_f1:.3f}")