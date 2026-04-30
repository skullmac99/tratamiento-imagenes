import os
# La variable de entorno DEBE ir antes de importar tensorflow
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Esto limpia mensajes innecesarios de log

# Forzar a que Python encuentre las librerías de CUDA en tu entorno
import sys
conda_env_path = os.environ.get('CONDA_PREFIX')
if conda_env_path:
    cuda_path = os.path.join(conda_env_path, 'Library', 'bin')
    if os.path.exists(cuda_path):
        os.add_dll_directory(cuda_path)

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Configuración de GPU ---
print("Buscando hardware...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU detectada: {len(gpus)} dispositivo(s) listo(s).")
    except RuntimeError as e:
        print(f"❌ Error configurando GPU: {e}")
else:
    print("⚠️ No se detectó GPU. El entrenamiento será en CPU (más lento).")
# ==========================================
# 1. CONFIGURACIÓN DE HIPERPARÁMETROS
# ==========================================
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 5 # Ajusta este número según tus conductas base (ej. mano a cara, boca, pierna, etc.)
BASE_DIR = 'dataset_cnn'

train_dir = os.path.join(BASE_DIR, 'entrenamiento')
val_dir = os.path.join(BASE_DIR, 'validacion')
test_dir = os.path.join(BASE_DIR, 'prueba')

# ==========================================
# 2. PREPROCESAMIENTO Y AUMENTO DE DATOS
# ==========================================
# Generador para entrenamiento con Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generadores para validación y prueba (SOLO normalización, sin augmentation)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ==========================================
# 3. ARQUITECTURA DEL MODELO (Basado en el Diagrama)
# ==========================================


model = Sequential([
    Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    # Capa de Convolución 1 + Pooling 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Capa de Convolución 2 + Pooling 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Aplanado
    Flatten(),
    
    # Capas Densas y Dropout para evitar Overfitting
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dropout(0.5), # Apaga el 50% de las neuronas
    
    # Capa de Salida
    Dense(NUM_CLASSES, activation='softmax') # Softmax para clasificación multiclase
])

# Compilación del modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 4. ENTRENAMIENTO DEL MODELO
# ==========================================
# Callbacks: Guardar el mejor modelo y detener si deja de mejorar (Early Stopping)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('mejor_modelo_inquietud.h5', monitor='val_accuracy', save_best_only=True)
]

print("Iniciando entrenamiento...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    workers=4,          # Usa 4 hilos para cargar imágenes
    use_multiprocessing=False # En Windows es mejor dejarlo en False con Conda
)

# ==========================================
# 5. EVALUACIÓN Y GRÁFICAS DE RENDIMIENTO
# ==========================================
# Extraer métricas del historial
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epocas_reales = range(1, len(acc) + 1)

# Graficar Precisión (Accuracy)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epocas_reales, acc, label='Entrenamiento')
plt.plot(epocas_reales, val_acc, label='Validación')
plt.title('Precisión (Accuracy) por Época')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Graficar Pérdida (Loss)
plt.subplot(1, 2, 2)
plt.plot(epocas_reales, loss, label='Entrenamiento')
plt.plot(epocas_reales, val_loss, label='Validación')
plt.title('Pérdida (Loss) por Época')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()

model.save('modelo_final_inquietud.h5')
print("Modelo guardado exitosamente.")
# (Opcional) Evaluar con el conjunto de prueba final
# test_generator = val_test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=16, class_mode='categorical')
# test_loss, test_acc = model.evaluate(test_generator)
# print(f"Precisión en el conjunto de prueba: {test_acc*100:.2f}%")