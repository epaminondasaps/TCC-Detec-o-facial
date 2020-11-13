import cv2
import dlib
import time

inicio = time.time()
imagem = cv2.imread("fotos/img4.jpg")
detector = dlib.cnn_face_detection_model_v1("recursos/mmod_human_face_detector.dat")
Detectadas = detector(imagem, 2)
print("Numero de faces detectadas: ", len(Detectadas))
for face in Detectadas:
    e, t, d, b, c = (int(face.rect.left()), int(face.rect.top()), int(face.rect.right()), int(face.rect.bottom()), face.confidence)
    print(c)
    cv2.rectangle(imagem, (e, t), (d, b), (255, 255, 0), 2)

cv2.imshow("CNN", imagem)
a = range(10000000)
b = []

for i in a:
    b.append(i*2)

fim = time.time()
diferenca = fim - inicio
print(diferenca)
cv2.waitKey(0)
cv2.destroyAllWindows()
