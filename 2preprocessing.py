import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from pathlib import Path

# Configuración
DATA_DIR = 'colon/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000'  # Actualiza con la ruta real
SPLIT_DIR = 'dataset_splits'  # Carpeta con los CSV de división
OUTPUT_DIR = 'preprocessed_images'  # Carpeta para guardar imágenes preprocesadas
IMG_SIZE = (150, 150)  # Tamaño original
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

# Crear directorios de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)
for split in ['train', 'val', 'test']:
    for label in CLASSES.values():
        os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)

# Definir aumentaciones (solo para entrenamiento)
augmentations = A.Compose([
    A.Rotate(limit=360, p=0.8, border_mode=cv2.BORDER_REFLECT),  # Usar BORDER_REFLECT para evitar bordes iniciales
    A.HorizontalFlip(p=0.5),  # Espejo horizontal
    A.VerticalFlip(p=0.5),  # Espejo vertical
    A.Affine(translate_percent=0.1, scale=1.0, rotate=0, p=0.5, border_mode=cv2.BORDER_REFLECT),  # Desplazamientos
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5)  # Brillo/contraste
])

# Función para rellenar bordes con media RGB
def fill_borders(img, mean_rgb):
    h, w = img.shape[:2]
    # Detectar bordes negros (valores cercanos a 0)
    mask = np.all(img < 10, axis=2)  # Píxeles con RGB < 10
    if np.any(mask):
        # Crear imagen con media RGB
        mean_img = np.full_like(img, mean_rgb)
        # Copiar contenido original, rellenar bordes con media RGB
        img[mask] = mean_img[mask]
    return img

# Función de preprocesamiento
def preprocess_image(img):
    # Normalización (0-1)
    img = img / 255.0
    # Reducción de ruido (desenfoque Gaussiano)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

# Función para guardar imagen
def save_image(img, output_path):
    img = (img * 255).astype(np.uint8)  # Convertir de 0-1 a 0-255
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# Procesar cada conjunto
for split in ['train', 'val', 'test']:
    # Cargar CSV
    csv_path = os.path.join(SPLIT_DIR, f'{split}_split.csv')
    if not os.path.exists(csv_path):
        print(f"Error: No se encuentra {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    
    # Lista para guardar rutas y etiquetas
    processed_data = []
    
    for idx, row in df.iterrows():
        img_path = row['image_path']
        label = row['label']
        class_folder = CLASSES[label]
        
        # Cargar imagen
        img = cv2.imread(img_path)
        if img is None:
            print(f"Advertencia: No se pudo cargar {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Calcular media RGB
        mean_rgb = np.mean(img, axis=(0, 1)).astype(np.uint8)
        
        # Lista de imágenes a procesar
        images_to_process = []
        
        if split == 'train':
            # Aplicar aumentaciones
            images_to_process.append(img)  # Imagen original
            # Generar 3 imágenes aumentadas
            for i in range(3):
                aug_img = augmentations(image=img)['image']
                # Rellenar bordes con media RGB
                aug_img = fill_borders(aug_img, mean_rgb)
                images_to_process.append(aug_img)
        else:
            # Sin aumentaciones para validación/test
            images_to_process.append(img)
        
        # Procesar cada imagen
        for i, proc_img in enumerate(images_to_process):
            # Preprocesamiento
            preprocessed_img = preprocess_image(proc_img)
            
            # Generar nombre único
            img_name = Path(img_path).stem
            if split == 'train' and i > 0:
                output_name = f"{img_name}_aug{i}.tif"
            else:
                output_name = f"{img_name}.tif"
            
            # Guardar imagen
            output_path = os.path.join(OUTPUT_DIR, split, class_folder, output_name)
            save_image(preprocessed_img, output_path)
            
            # Verificar bordes (debugging)
            saved_img = cv2.imread(output_path)
            saved_img = cv2.cvtColor(saved_img, cv2.COLOR_BGR2RGB)
            border_pixels = np.all(saved_img[:5, :, :] < 10, axis=2)  # Primeras 5 filas
            if np.any(border_pixels):
                print(f"Advertencia: Bordes negros detectados en {output_path}")
            
            # Guardar ruta y etiqueta
            processed_data.append({
                'image_path': output_path,
                'label': label
            })
    
    # Guardar CSV con imágenes preprocesadas
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(os.path.join(OUTPUT_DIR, f'{split}_preprocessed.csv'), index=False)
    
    print(f"Conjunto {split}: {len(processed_df)} imágenes preprocesadas")

# Reporte final
for split in ['train', 'val', 'test']:
    csv_path = os.path.join(OUTPUT_DIR, f'{split}_preprocessed.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"\nDistribución en {split}:")
        counts = df['label'].value_counts().sort_index()
        for label, count in counts.items():
            print(f"{CLASSES[label]} (Label {label}): {count} imágenes ({count/len(df)*100:.2f}%)")