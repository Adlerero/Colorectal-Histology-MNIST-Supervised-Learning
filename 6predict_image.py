import numpy as np
import cv2
import pickle
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew
import os

# Configuración
MODEL_DIR = 'results'  # Directorio con modelo y scaler
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
    'hist_b_mean', 'hist_b_variance',
 'hist_b_skewness'
]

# Cargar modelo y scaler
with open(os.path.join(MODEL_DIR, 'best_model_RandomForest.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Función para preprocesar y extraer características
def preprocess_and_extract_features(image_path):
    # Leer y preprocesar imagen
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB
    img = cv2.resize(img, (150, 150))  # Redimensionar a 150x150

    # Desenfoque Gaussiano para histogramas
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Extraer GLCM
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    gray_img = (gray_img * 255).astype(np.uint8)
    glcm = graycomatrix(gray_img, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=256, symmetric=True, normed=True)
    glcm_features = []
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        prop_values = graycoprops(glcm, prop)
        glcm_features.extend(np.mean(prop_values, axis=1))

    # Extraer histogramas RGB
    hist_features = []
    for channel in range(3):
        channel_data = img_blurred[:,:,channel] / 255.0  # Normalizar a 0-1
        mean = np.mean(channel_data)
        variance = np.var(channel_data)
        skewness = 0.0 if np.var(channel_data) < 1e-6 else skew(channel_data.flatten())
        hist_features.extend([mean, variance, skewness])
    
    # Combinar características
    features = np.array(glcm_features + hist_features)
    return features

# Predecir clase
def predict_image(image_path):
    # Extraer características
    features = preprocess_and_extract_features(image_path)
    
    # Normalizar
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Predecir
    pred_class = model.predict(features_scaled)[0]
    pred_label = CLASSES[pred_class]
    
    return pred_class, pred_label

# Ejemplo de uso
if __name__ == "__main__":
    # Reemplaza con la ruta de tu nueva imagen
    new_image_path = 'colon/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000/08_EMPTY/10AA3_CRC-Prim-HE-06_005.tif_Row_1201_Col_4051.tif'  # Ej. 'c:\\AdlerUP\\test\\new_image.tif'
    try:
        class_num, class_label = predict_image(new_image_path)
        print(f"\nImagen: {new_image_path}")
        print(f"Clase predicha: {class_label} ({class_num})")
    except Exception as e:
        print(f"Error: {e}")
