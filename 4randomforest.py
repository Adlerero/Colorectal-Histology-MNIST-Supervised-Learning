import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from time import time

# Configuración
DATA_DIR = 'features'  # Carpeta con los CSV de características
OUTPUT_DIR = 'results2'  # Carpeta para guardar resultados
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

# Crear directorio de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar datos
def load_data(split):
    df = pd.read_csv(os.path.join(DATA_DIR, f'{split}_features.csv'))
    X = df[FEATURES].values
    y = df['label'].values
    return X, y

X_train, y_train = load_data('train')
X_val, y_val = load_data('val')
X_test, y_test = load_data('test')

# Normalizar características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Guardar scaler
with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# Función para evaluar modelo
def evaluate_model(model, X, y, split, model_name):
    start_time = time()
    y_pred = model.predict(X)
    end_time = time()
    
    # Métricas macro-promediadas
    metrics = {
        'split': split,
        'model': model_name,
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, average='macro'),
        'recall': recall_score(y, y_pred, average='macro'),
        'f1': f1_score(y, y_pred, average='macro'),
        'time (s)': end_time - start_time
    }
    
    # Métricas por clase
    precision_per_class = precision_score(y, y_pred, average=None)
    recall_per_class = recall_score(y, y_pred, average=None)
    f1_per_class = f1_score(y, y_pred, average=None)
    
    # Crear DataFrame para métricas por clase
    per_class_metrics = pd.DataFrame({
        'Class': [CLASSES[i+1] for i in range(len(CLASSES))],
        'Precision': precision_per_class,
        'Recall': recall_per_class,
        'F1-score': f1_per_class
    })
    
    # Guardar métricas por clase
    per_class_metrics.to_csv(os.path.join(OUTPUT_DIR, f'per_class_metrics_{model_name}_{split}.csv'), index=False)
    
    # Matriz de confusión
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES.values(), yticklabels=CLASSES.values())
    plt.title(f'Confusion Matrix - {model_name} ({split})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(OUTPUT_DIR, f'cm_{model_name}_{split}.png'))
    plt.close()
    
    return metrics

# Entrenar Random Forest
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, None]
}
rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, n_jobs=-1)
start_time = time()
rf.fit(X_train, y_train)
rf_time = time() - start_time
print(f"Random Forest - Mejores parámetros: {rf.best_params_}, Tiempo: {rf_time:.2f} s")

# Evaluar modelo
results = []
results.append(evaluate_model(rf.best_estimator_, X_val, y_val, 'Validation', 'RandomForest'))
results.append(evaluate_model(rf.best_estimator_, X_test, y_test, 'Test', 'RandomForest'))

# Guardar modelo
with open(os.path.join(OUTPUT_DIR, 'best_model_RandomForest.pkl'), 'wb') as f:
    pickle.dump(rf.best_estimator_, f)

# Crear tabla comparativa
results_df = pd.DataFrame(results)
results_df = results_df[['model', 'split', 'accuracy', 'precision', 'recall', 'f1', 'time (s)']]
results_df.to_csv(os.path.join(OUTPUT_DIR, 'results_comparison.csv'), index=False)

# Visualizar tabla
plt.figure(figsize=(10, 4))
plt.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc='center', loc='center')
plt.axis('off')
plt.title('Random Forest Results')
plt.savefig(os.path.join(OUTPUT_DIR, 'results_table.png'))
plt.close()

# Imprimir resultados
print("\nResultados:")
print(results_df)