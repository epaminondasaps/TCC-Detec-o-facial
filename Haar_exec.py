import cv2
import time

inicio = time.time()
imagem = cv2.imread("fotos/img1.jpg")
classificador = cv2.CascadeClassifier("recursos/haarcascade_frontalface_default.xml")
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.2, minSize=(50,50))
print("Numero de faces detectadas: ", len(facesDetectadas))
for (x, y, l, a) in facesDetectadas:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)


cv2.imshow("Haar", imagem)
a = range(10000000)
b = []

for i in a:
    b.append(i*2)

fim = time.time()
diferenca = fim - inicio
print(diferenca)
cv2.waitKey(0)
cv2.destroyAllWindows()