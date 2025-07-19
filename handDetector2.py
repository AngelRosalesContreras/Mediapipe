import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees


def palm_centroid(coordinates_list):
    coordinates = np.array(coordinates_list)
    centroid = np.mean(coordinates, axis=0)
    centroid = int(centroid[0]), int(centroid[1])
    return centroid


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# Cambiar el tamaño de la pantalla (ancho y alto en píxeles)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Ancho
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Alto

# Pulgar
thumb_points = [1, 2, 4]

# Índice, medio, anular y meñique
palm_points = [0, 1, 2, 5, 9, 13, 17]
fingertips_points = [8, 12, 16, 20]
finger_base_points = [6, 10, 14, 18]

# Colores
GREEN = (48, 255, 48)
BLUE = (192, 101, 21)
YELLOW = (0, 204, 255)
PURPLE = (128, 64, 128)
PEACH = (180, 229, 255)
COLORS = [(48, 255, 48), (192, 101, 21), (0, 204, 255), (128, 64, 128), (180, 229, 255)]

with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,  # Cambiado para detectar hasta 2 manos
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Lista para almacenar los contadores de dedos para cada mano
        all_fingers_counter = []

        if results.multi_hand_landmarks:
            # Iterar a través de cada mano detectada
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                coordinates_thumb = []
                coordinates_palm = []
                coordinates_ft = []
                coordinates_fb = []

                # Clasificar la mano como izquierda o derecha
                handedness = results.multi_handedness[hand_idx].classification[0].label
                hand_label = f"{handedness} Hand"

                # Recopilar coordenadas para esta mano específica
                for index in thumb_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_thumb.append([x, y])

                for index in palm_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_palm.append([x, y])

                for index in fingertips_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_ft.append([x, y])

                for index in finger_base_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_fb.append([x, y])

                # Pulgar
                p1 = np.array(coordinates_thumb[0])
                p2 = np.array(coordinates_thumb[1])
                p3 = np.array(coordinates_thumb[2])

                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)

                # Calcular el ángulo
                angle = degrees(acos((l1 ** 2 + l3 ** 2 - l2 ** 2) / (2 * l1 * l3)))
                thumb_finger = np.array(False)
                if angle > 150:
                    thumb_finger = np.array(True)

                # Índice, medio, anular y meñique
                nx, ny = palm_centroid(coordinates_palm)
                cv2.circle(frame, (nx, ny), 3, (0, 255, 0), 2)
                coordinates_centroid = np.array([nx, ny])
                coordinates_ft = np.array(coordinates_ft)
                coordinates_fb = np.array(coordinates_fb)

                # Distancias
                d_centrid_ft = np.linalg.norm(coordinates_centroid - coordinates_ft, axis=1)
                d_centrid_fb = np.linalg.norm(coordinates_centroid - coordinates_fb, axis=1)
                dif = d_centrid_ft - d_centrid_fb
                fingers = dif > 0
                fingers = np.append(thumb_finger, fingers)
                fingers_counter = str(np.count_nonzero(fingers == True))
                all_fingers_counter.append(fingers_counter)

                # Grosor para visualización
                thickness = [2, 2, 2, 2, 2]
                for (i, finger) in enumerate(fingers):
                    if finger == True:
                        thickness[i] = -1

                # Dibujar landmarks de la mano
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Dibujar información para cada mano
                offset_y = 100 * (hand_idx + 1)

                # Mostrar etiqueta de |1mano
                cv2.putText(frame, hand_label, (10, offset_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            COLORS[hand_idx % len(COLORS)], 2)

                # Rectángulo para el contador de dedos
                cv2.rectangle(frame, (0, offset_y), (80, offset_y + 80), COLORS[hand_idx % len(COLORS)], -1)
                cv2.putText(frame, fingers_counter, (15, offset_y + 65), 1, 5, (255, 255, 255), 2)

                # Visualización de dedos para esta mano
                base_x = 100
                for i in range(5):
                    cv2.rectangle(frame,
                                  (base_x + i * 60, offset_y),
                                  (base_x + 50 + i * 60, offset_y + 50),
                                  COLORS[i], thickness[i])

                # Etiquetas de dedos
                finger_names = ["Pulgar", "qIndice", "Medio", "Anular", "Menique"]
                for i, name in enumerate(finger_names):
                    cv2.putText(frame, name, (base_x + i * 60, offset_y + 70),
                                1, 0.8, (255, 255, 255), 2)

        # Visualización del número total de manos detectadas
        cv2.putText(frame, f"Manos: {len(all_fingers_counter)}", (width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()