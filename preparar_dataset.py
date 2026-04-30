import os
import cv2
import shutil
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURACIÓN DE DIRECTORIOS
# ==========================================
RAW_DATA_DIR = "datasets_raw" # Aquí pones lo que descargues de Kaggle/Github
CNN_DATA_DIR = "dataset_cnn"  # Aquí se creará la estructura final para entrenar

# Las clases que definiste en tu proyecto
CLASES = ["mano_cara", "mano_boca", "movimiento_pierna", "desvio_mirada", "postura_neutral"]

# Proporciones de tu documento TMPI_3_3
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
IMG_SIZE = (224, 224)

def crear_estructura_directorios():
    """Crea las carpetas base para el entrenamiento."""
    for split in ['entrenamiento', 'validacion', 'prueba']:
        for clase in CLASES:
            os.makedirs(os.path.join(CNN_DATA_DIR, split, clase), exist_ok=True)

def procesar_video_a_frames(video_path, output_dir_clase, prefix, max_frames=50):
    """Extrae frames de un video y los guarda preprocesados."""
    cap = cv2.VideoCapture(video_path)
    count = 0
    frames_guardados = 0
    
    while cap.isOpened() and frames_guardados < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extraer 1 de cada 5 frames para evitar redundancia extrema
        if count % 5 == 0:
            frame_resized = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_AREA)
            img_name = f"{prefix}_frame_{frames_guardados}.jpg"
            cv2.imwrite(os.path.join(output_dir_clase, img_name), frame_resized)
            frames_guardados += 1
        count += 1
        
    cap.release()
    return frames_guardados

def distribuir_datos():
    """Divide las imágenes procesadas en 70/15/15."""
    for clase in CLASES:
        clase_dir = os.path.join(RAW_DATA_DIR, clase)
        if not os.path.exists(clase_dir):
            print(f"Directorio crudo no encontrado para la clase: {clase}. Saltando...")
            continue
            
        archivos = os.listdir(clase_dir)
        archivos_validos = [f for f in archivos if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(archivos_validos) == 0:
            continue

        # División de datos (Sklearn train_test_split)
        train_files, temp_files = train_test_split(archivos_validos, test_size=(1 - TRAIN_RATIO), random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=(TEST_RATIO / (TEST_RATIO + VAL_RATIO)), random_state=42)

        # Mover archivos a sus respectivas carpetas
        splits = {'entrenamiento': train_files, 'validacion': val_files, 'prueba': test_files}
        
        for split_name, files in splits.items():
            for f in files:
                src = os.path.join(clase_dir, f)
                dst = os.path.join(CNN_DATA_DIR, split_name, clase, f)
                # Aquí podrías aplicar tu función cv2.resize si los archivos crudos son fotos de distintos tamaños
                img = cv2.imread(src)
                if img is not None:
                    img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(dst, img_resized)

        print(f"Clase '{clase}' procesada: {len(train_files)} Train | {len(val_files)} Val | {len(test_files)} Test")

def main():
    print("Creando estructura de directorios CNN...")
    crear_estructura_directorios()
    
    print("Iniciando procesamiento y distribución de datos...")
    distribuir_datos()
    print("¡Dataset listo para alimentar el modelo!")

if __name__ == "__main__":
    main()