import cv2
import mediapipe as mp
import numpy as np

# Inicializar soluciones de MediaPipe para rostros
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Configuración de la webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Puntos específicos de interés en el rostro
CONTORNO_CARA = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
                 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
CEJA_DERECHA = [70, 63, 105, 66, 107]
CEJA_IZQUIERDA = [336, 296, 334, 293, 300]
LABIOS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
OJOS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
NARIZ = [1, 2, 3, 4, 5, 6, 168, 195, 197, 8, 9]

# Definición de los puntos del iris
LEFT_IRIS = [474, 475, 476, 477]  # Puntos del iris izquierdo
RIGHT_IRIS = [469, 470, 471, 472]  # Puntos del iris derecho

# Colores para diferentes partes del rostro
CYAN = (255, 255, 0)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
PURPLE = (255, 0, 255)
YELLOW = (0, 255, 255)
COLORS = {
    "CONTORNO_CARA": BLUE,
    "CEJA_DERECHA": GREEN,
    "CEJA_IZQUIERDA": GREEN,
    "LABIOS": RED,
    "OJOS": PURPLE,
    "NARIZ": YELLOW
}


# Para dibujar puntos específicos más grandes
def draw_landmark_point(image, landmark, color=(0, 0, 255), radius=3):
    height, width, _ = image.shape
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    cv2.circle(image, (x, y), radius, color, -1)


# Para dibujar texto en pantalla
def draw_text(img, text, position, color=(0, 255, 0), thickness=2, font_scale=0.7):
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, lineType=cv2.LINE_AA)


# Configuración de Face Mesh
with mp_face_mesh.FaceMesh(
        max_num_faces=3,
        refine_landmarks=True,  # Importante para detectar el iris
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Fallo al leer la webcam.")
            break

        # Obtener dimensiones de la imagen
        height, width, _ = image.shape

        # Voltear la imagen horizontalmente para una visualización tipo espejo
        image = cv2.flip(image, 1)

        # Convertir a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Procesar la imagen con Face Mesh
        results = face_mesh.process(image_rgb)

        # Dibujar los puntos de referencia
        if results.multi_face_landmarks:
            # Informar el número de rostros detectados
            num_faces = len(results.multi_face_landmarks)
            draw_text(image, f"Rostros detectados: {num_faces}", (10, 30), RED, 2, 1)

            # Procesar cada rostro detectado
            for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                # Calcular las coordenadas para posicionar la información
                face_y_pos = 70 + face_idx * 40
                draw_text(image, f"Rostro {face_idx + 1}", (10, face_y_pos), COLORS["CONTORNO_CARA"], 2)

                # Dibujar partes específicas del rostro con sus respectivos colores
                landmark_groups = {
                    "CONTORNO_CARA": CONTORNO_CARA,
                    "CEJA_DERECHA": CEJA_DERECHA,
                    "CEJA_IZQUIERDA": CEJA_IZQUIERDA,
                    "LABIOS": LABIOS,
                    "OJOS": OJOS,
                    "NARIZ": NARIZ
                }

                # Dibujar cada grupo de landmarks
                for name, landmarks in landmark_groups.items():
                    color = COLORS[name]
                    for idx in landmarks:
                        draw_landmark_point(image, face_landmarks.landmark[idx], color, 4)

                # Opcionalmente, dibujar la malla completa
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                # )

                # Detectar la dirección de la mirada (usando los puntos del iris)
                # Verificamos que tengamos suficientes landmarks para los iris
                if len(face_landmarks.landmark) > 478:  # Aseguramos que los landmarks refinados están presentes
                    # Obtener landmarks del iris (corregido)
                    left_iris_landmarks = [face_landmarks.landmark[idx] for idx in LEFT_IRIS]
                    right_iris_landmarks = [face_landmarks.landmark[idx] for idx in RIGHT_IRIS]

                    # Dibujar iris
                    for landmark in left_iris_landmarks + right_iris_landmarks:
                        draw_landmark_point(image, landmark, CYAN, 2)

                    # Calcular dirección aproximada de la mirada basado en la posición del iris
                    left_iris_x = sum([landmark.x for landmark in left_iris_landmarks]) / len(
                        left_iris_landmarks) * width
                    right_iris_x = sum([landmark.x for landmark in right_iris_landmarks]) / len(
                        right_iris_landmarks) * width
                    center_x = width / 2

                    # Determinar si está mirando a la izquierda o derecha
                    gaze_offset = ((left_iris_x + right_iris_x) / 2) - center_x
                    gaze_direction = "Centro"
                    if gaze_offset < -15:
                        gaze_direction = "Derecha"
                    elif gaze_offset > 15:
                        gaze_direction = "Izquierda"

                    draw_text(image, f"Mirada: {gaze_direction}", (width - 200, face_y_pos), CYAN, 2)
                else:
                    # Si no hay landmarks refinados, informar al usuario
                    draw_text(image, "Iris no detectado", (width - 200, face_y_pos), RED, 2)

        # Mostrar la imagen resultante
        cv2.imshow('Face Landmarks', image)

        # Salir si se presiona ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()