import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew
from pathlib import Path

# Configuración
SPLIT_DIR = 'preprocessed_images'  # Carpeta con los CSV de imágenes preprocesadas
OUTPUT_DIR = 'features'  # Carpeta para guardar características
DISTANCES = [1, 3, 5]  # Distancias para GLCM
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Direcciones: 0°, 45°, 90°, 135°
GLCM_PROPERTIES = ['contrast', 'correlation', 'energy', 'homogeneity']
CLASSES = {
    1: '01_TUMOR',
    2: '02_STROMA',
    3: '03_COMPLEX',
    4: '04_LYMPHO',
    5: '05_DEBRIS',
    6: '06_MUCOSA',
    7: '07_ADIPOSE',
    8: '08_EMPTY'
}

# Crear directorio de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Función para extraer características GLCM
def extract_glcm_features(img):
    # Convertir a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Ajuste de contraste (ecualización de histograma)
    gray_img = cv2.equalizeHist(gray_img)
    
    # Discretizar a 8 bits (256 niveles)
    gray_img = (gray_img * 255).astype(np.uint8)
    
    # Calcular GLCM
    glcm = graycomatrix(gray_img, distances=DISTANCES, angles=ANGLES, 
                        levels=256, symmetric=True, normed=True)
    
    # Extraer propiedades
    features = []
    for prop in GLCM_PROPERTIES:
        prop_values = graycoprops(glcm, prop)  # Shape: (n_distances, n_angles)
        # Promediar sobre las 4 direcciones
        prop_mean = np.mean(prop_values, axis=1)  # Shape: (n_distances,)
        features.extend(prop_mean)
    
    return features

# Función para extraer características de histogramas RGB
def extract_histogram_features(img):
    features = []
    for channel in range(3):  # R, G, B
        # Normalizar a 0-1
        channel_data = img[:, :, channel] / 255.0
        # Calcular estadísticas
        mean = np.mean(channel_data)
        variance = np.var(channel_data)
        # Verificar baja variabilidad para evitar advertencia en skew
        if variance < 1e-6:  # Umbral para datos casi idénticos
            skewness = 0.0
        else:
            skewness = skew(channel_data.flatten())
        features.extend([mean, variance, skewness])
    return features

# Procesar cada conjunto
for split in ['train', 'val', 'test']:
    # Cargar CSV
    csv_path = os.path.join(SPLIT_DIR, f'{split}_preprocessed.csv')
    if not os.path.exists(csv_path):
        print(f"Error: No se encuentra {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    
    # Lista para guardar características
    features_list = []
    
    for idx, row in df.iterrows():
        img_path = row['image_path']
        label = row['label']
        
        # Cargar imagen
        img = cv2.imread(img_path)
        if img is None:
            print(f"Advertencia: No se pudo cargar {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extraer características GLCM
        glcm_features = extract_glcm_features(img)
        
        # Extraer características de histogramas RGB
        hist_features = extract_histogram_features(img)
        
        # Combinar características
        features = glcm_features + hist_features
        
        # Guardar con etiqueta
        features_list.append({
            'image_path': img_path,
            **{f'glcm_{prop}_{d}': glcm_features[i] for i, (prop, d) in enumerate(
                [(p, d) for p in GLCM_PROPERTIES for d in DISTANCES])},
            'hist_r_mean': hist_features[0],
            'hist_r_variance': hist_features[1],
            'hist_r_skewness': hist_features[2],
            'hist_g_mean': hist_features[3],
            'hist_g_variance': hist_features[4],
            'hist_g_skewness': hist_features[5],
            'hist_b_mean': hist_features[6],
            'hist_b_variance': hist_features[7],
            'hist_b_skewness': hist_features[8],
            'label': label
        })
    
    # Crear DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Guardar en CSV
    output_csv = os.path.join(OUTPUT_DIR, f'{split}_features.csv')
    features_df.to_csv(output_csv, index=False)
    
    print(f"Conjunto {split}: {len(features_df)} imágenes procesadas, características guardadas en {output_csv}")
    
    # Reporte de distribución
    print(f"\nDistribución en {split}:")
    counts = features_df['label'].value_counts().sort_index()
    for label, count in counts.items():
        print(f"{CLASSES[label]} (Label {label}): {count} imágenes ({count/len(features_df)*100:.2f}%)")