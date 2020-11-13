import sys
import dlib
import cv2

pula_quadros = 30
captura = cv2.VideoCapture(0)
contadorQuadros = 0
detector = cv2.CascadeClassifier("recursos/haarcascade_frontalface_default.xml")

while captura.isOpened():
    conectado, frame = captura.read()
    contadorQuadros += 1
    if contadorQuadros % pula_quadros == 0:
        imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objetosDetectados = detector.detectMultiScale(imagemCinza, scaleFactor=1.2, minSize=(50, 50))
        for (x, y, l, a) in objetosDetectados:
            cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)
        cv2.imshow("Preditor de Objetos", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

captura.release()
cv2.destroyAllWindows()
sys.exit(0)