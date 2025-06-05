import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuración
DATA_DIR = 'colon/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000'
CLASSES = {
    '01_TUMOR': 1,
    '02_STROMA': 2,
    '03_COMPLEX': 3,
    '04_LYMPHO': 4,
    '05_DEBRIS': 5,
    '06_MUCOSA': 6,
    '07_ADIPOSE': 7,
    '08_EMPTY': 8
}
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
OUTPUT_DIR = 'dataset_splits'  # Carpeta para guardar los CSV

# Crear directorio de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Recopilar imágenes y etiquetas
image_paths = []
labels = []

for class_folder, label in CLASSES.items():
    folder_path = os.path.join(DATA_DIR, class_folder)
    if not os.path.exists(folder_path):
        print(f"Advertencia: La carpeta {folder_path} no existe")
        continue
    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith('.tif'):
            img_path = os.path.join(folder_path, img_name)
            image_paths.append(img_path)
            labels.append(label)
        else:
            print(f"Ignorando archivo no .tif: {img_name}")

# Convertir a DataFrame
data = pd.DataFrame({
    'image_path': image_paths,
    'label': labels
})

# 2. Verificar distribución de clases
class_counts = data['label'].value_counts().sort_index()
print("Distribución de clases:")
for label, count in class_counts.items():
    class_name = [k for k, v in CLASSES.items() if v == label][0]
    print(f"{class_name} (Label {label}): {count} imágenes ({count/len(data)*100:.2f}%)")

# 3. Dividir en entrenamiento + (validación + prueba)
train_data, temp_data, train_labels, temp_labels = train_test_split(
    data['image_path'],
    data['label'],
    train_size=TRAIN_RATIO,
    stratify=data['label'],
    random_state=42
)

# 4. Dividir temp_data en validación y prueba
val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)  # Ajustar proporción
val_data, test_data, val_labels, test_labels = train_test_split(
    temp_data,
    temp_labels,
    train_size=val_ratio_adjusted,
    stratify=temp_labels,
    random_state=42
)

# 5. Crear DataFrames para cada conjunto
train_df = pd.DataFrame({'image_path': train_data, 'label': train_labels})
val_df = pd.DataFrame({'image_path': val_data, 'label': val_labels})
test_df = pd.DataFrame({'image_path': test_data, 'label': test_labels})

# 6. Mezclar dentro de cada conjunto
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 7. Guardar en CSV
train_df.to_csv(os.path.join(OUTPUT_DIR, 'train_split.csv'), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, 'val_split.csv'), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, 'test_split.csv'), index=False)

# 8. Reportar tamaños
print("\nTamaños de los conjuntos:")
print(f"Entrenamiento: {len(train_df)} imágenes ({len(train_df)/len(data)*100:.2f}%)")
print(f"Validación: {len(val_df)} imágenes ({len(val_df)/len(data)*100:.2f}%)")
print(f"Prueba: {len(test_df)} imágenes ({len(test_df)/len(data)*100:.2f}%)")

# 9. Verificar estratificación
print("\nDistribución de clases en cada conjunto:")
for split_name, split_df in [('Entrenamiento', train_df), ('Validación', val_df), ('Prueba', test_df)]:
    print(f"\n{split_name}:")
    split_counts = split_df['label'].value_counts().sort_index()
    for label, count in split_counts.items():
        class_name = [k for k, v in CLASSES.items() if v == label][0]
        print(f"{class_name} (Label {label}): {count} imágenes ({count/len(split_df)*100:.2f}%)")