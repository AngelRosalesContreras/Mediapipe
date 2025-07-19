import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Inicializar MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Configuración de la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Ancho
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Alto

# Colores
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
COLORS = [BLUE, GREEN, RED, YELLOW, PURPLE, CYAN]


# Clase para representar una partícula
class Particle:
    def __init__(self, x, y, color=None):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2, 2)  # Velocidad horizontal aleatoria
        self.vy = random.uniform(-3, -0.5)  # Velocidad vertical (hacia arriba)
        self.radius = random.randint(2, 6)  # Tamaño aleatorio
        self.life = random.uniform(0.5, 2)  # Tiempo de vida en segundos
        self.born_time = time.time()
        self.color = color if color else random.choice(COLORS)
        self.alpha = 1.0  # Transparencia (1 = opaco)

    def update(self):
        # Actualizar posición
        self.x += self.vx
        self.y += self.vy

        # Añadir algo de gravedad y aleatoriedad
        self.vy += 0.05  # Gravedad
        self.vx += random.uniform(-0.2, 0.2)  # Movimiento aleatorio

        # Reducir tamaño y transparencia con el tiempo
        age = time.time() - self.born_time
        life_fraction = age / self.life
        self.alpha = max(0, 1 - life_fraction)

        # Comprobar si la partícula sigue viva
        return age < self.life

    def draw(self, frame):
        # Dibujar la partícula con su transparencia actual
        if self.alpha > 0:
            overlay = frame.copy()
            cv2.circle(overlay, (int(self.x), int(self.y)), self.radius, self.color, -1)
            cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)


# Función para crear partículas basadas en el movimiento
def create_particles_based_on_movement(prev_landmarks, curr_landmarks, width, height, point_idx, threshold=0.01):
    if prev_landmarks is None or curr_landmarks is None:
        return []

    # Obtener coordenadas actuales y previas
    curr_x = int(curr_landmarks.landmark[point_idx].x * width)
    curr_y = int(curr_landmarks.landmark[point_idx].y * height)
    prev_x = int(prev_landmarks.landmark[point_idx].x * width)
    prev_y = int(prev_landmarks.landmark[point_idx].y * height)

    # Calcular la magnitud del movimiento
    dx = curr_x - prev_x
    dy = curr_y - prev_y
    movement = np.sqrt(dx * dx + dy * dy)

    particles = []

    # Crear partículas solo si hay suficiente movimiento
    if movement > threshold * width:
        # Número de partículas proporcional al movimiento
        num_particles = int(min(20, movement / 5))

        # Crear partículas a lo largo de la trayectoria del movimiento
        for _ in range(num_particles):
            # Posición aleatoria a lo largo de la trayectoria
            t = random.random()
            x = prev_x + t * dx
            y = prev_y + t * dy

            # Color basado en velocidad
            speed_factor = min(1.0, movement / 50)
            r = int(255 * speed_factor)
            g = int(255 * (1 - speed_factor))
            b = 255
            color = (b, g, r)  # OpenCV usa BGR

            particles.append(Particle(x, y, color))

    return particles


# Lista para almacenar partículas activas
particles = []

# Para seguimiento de landmarks previos
prev_landmarks = None

# Iniciar procesamiento de pose
with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Voltear horizontalmente para efecto espejo
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # Convertir a RGB para MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar imagen
        results = pose.process(frame_rgb)

        # Actualizar partículas existentes
        particles = [p for p in particles if p.update()]

        # Limitar el número máximo de partículas para rendimiento
        if len(particles) > 500:
            particles = particles[-500:]

        # Crear partículas basadas en el movimiento
        if results.pose_landmarks and prev_landmarks:
            # Puntos de interés para generar partículas
            key_points = [
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.RIGHT_WRIST,
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE
            ]

            for point in key_points:
                new_particles = create_particles_based_on_movement(
                    prev_landmarks, results.pose_landmarks, width, height, point)
                particles.extend(new_particles)

        # Dibujar partículas
        for particle in particles:
            particle.draw(frame)

        # Dibujar pose
        if results.pose_landmarks:
            # Guardar landmarks actuales para la siguiente iteración
            prev_landmarks = results.pose_landmarks

            # Dibujar landmarks de pose
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(128, 0, 250), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=WHITE, thickness=2))

            # Resaltar puntos clave (manos, pies)
            key_points = [
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.RIGHT_WRIST,
                mp_pose.PoseLandmark.LEFT_ANKLE,
                mp_pose.PoseLandmark.RIGHT_ANKLE
            ]

            for point in key_points:
                landmark = results.pose_landmarks.landmark[point]
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                # Dibujar un círculo más grande en los puntos clave
                cv2.circle(frame, (x, y), 10, YELLOW, -1)
                cv2.circle(frame, (x, y), 12, RED, 2)

        # Añadir texto instructivo
        cv2.putText(frame, "Mueve tus brazos y piernas para crear partículas",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

        # Mostrar imagen
        cv2.imshow("Espejo con Partículas", frame)

        # Salir con ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()