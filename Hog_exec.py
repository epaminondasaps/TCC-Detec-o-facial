import cv2
import dlib
import time

inicio = time.time()

foto = cv2.imread("fotos/img5.jpg")
detector = dlib.get_frontal_face_detector()
Detectadas = detector(foto, 2)
print("Numero de faces detectadas: ", len(Detectadas))
for face in Detectadas:

    e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
    cv2.rectangle(foto, (e, t), (d, b), (0, 255, 255), 2)

cv2.imshow("Hog", foto)
a = range(10000000)
b = []

for i in a:
    b.append(i*2)

fim = time.time()
diferenca = fim - inicio
print(diferenca)
cv2.waitKey(0)
cv2.destroyAllWindows()

