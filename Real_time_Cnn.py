import sys
import dlib
import cv2

pula_quadros = 30
captura = cv2.VideoCapture(0)
contadorQuadros = 0

detector = dlib.cnn_face_detection_model_v1("recursos/mmod_human_face_detector.dat")
while captura.isOpened():
    conectado, frame = captura.read()
    contadorQuadros += 1
    if contadorQuadros % pula_quadros == 0:
        facesDetectadas = detector(frame, 1)
        for face in facesDetectadas:
            e, t, d, b, c = (
            int(face.rect.left()), int(face.rect.top()), int(face.rect.right()), int(face.rect.bottom()),
            face.confidence)
            print(c)
            cv2.rectangle(frame, (e, t), (d, b), (255, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == 27:
            break

captura.release()
cv2.destroyAllWindows()
sys.exit(0)