import cv2
import numpy as np
import tensorflow as tf
import os

# 1. Configuración de hardware y rutas
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
modelo_path = 'mejor_modelo_inquietud.h5' # O el nombre de tu carpeta de modelo

# 2. Cargar el modelo entrenado
print("Cargando modelo...")
model = tf.keras.models.load_model(modelo_path)

# 3. Definir tus clases (¡Asegúrate de que el orden sea el mismo que en tus carpetas!)
# Puedes ver el orden con: print(train_generator.class_indices)
classes = ['Conducta_1', 'Conducta_2', 'Conducta_3', 'Conducta_4', 'Conducta_5']

# 4. Configurar la captura de video
cap = cv2.VideoCapture(0) # 0 para la webcam integrada

print("Iniciando cámara... Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocesamiento del frame (Igual al entrenamiento)
    # 1. Redimensionar a 224x224 (o lo que usaste en IMG_WIDTH/HEIGHT)
    img = cv2.resize(frame, (224, 224))
    
    # 2. Convertir a array y normalizar (rescale 1./255)
    img_array = np.expand_dims(img, axis=0) / 255.0

    # 3. Predicción
    predictions = model.predict(img_array, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx] * 100

    # 4. Mostrar resultados en pantalla
    label = f"{classes[class_idx]}: {confidence:.2f}%"
    color = (0, 255, 0) if confidence > 70 else (0, 255, 255) # Verde si es confiable
    
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Deteccion de Comportamiento en Tiempo Real', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()