import cv2
import torch
from pathlib import Path
from Architecture import ClassificationModel

MODEL_PATH = Path("/home/frasero/PycharmProjects/Models/FaceRecognition(state_dict).pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ClassificationModel().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

CLASSES = ['anger', 'contempt', 'fear', 'happy', 'ahegao', 'disgust', 'surprise', 'neutral', 'sad']

# Детектор облич (Haar — простий варіант)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Детектуємо обличчя (по сірому, але вирізати будемо кольорове)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame
        face_resized = cv2.resize(face, (96, 96))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_tensor = torch.tensor(face_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        face_tensor = face_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(face_tensor)
            pred = logits.argmax(dim=1)
            print(pred)

        label = f"{CLASSES[pred]}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()