import cv2
import os


dir = os.path.dirname(os.path.abspath(__file__))

caminho_imagem1 = os.path.join(dir, '../imagens/car.jpg')
#imagem de carros em uma rodovia

imagem1 = cv2.imread(caminho_imagem1)
janela = "janela"

imagem1 = cv2.resize(imagem1,(800,400))
imagem1_cinza = cv2.cvtColor(imagem1,cv2.COLOR_BGR2GRAY)

caminho_classificador_carro = os.path.join(dir, '../cascades/cars.xml')

classificador_carro = cv2.CascadeClassifier(caminho_classificador_carro)
deteccao_carro = classificador_carro.detectMultiScale(imagem1_cinza,scaleFactor= 1.03,minNeighbors=5)

for x,y,w,h in deteccao_carro:
    cv2.rectangle(imagem1,(x,y),(x+w,y+h),(0,0,255),1)

('Detecções de carro:' + str(len(deteccao_carro)))
cv2.imshow(janela,imagem1)
#cv2.waitKey(0)

#identificando corpos inteiros

caminho_imagem2 = os.path.join(dir, '../imagens/people3.jpg')
imagem2 = cv2.imread(caminho_imagem2)
imagem2 = cv2.resize(imagem2,(800,350))
imagem2_cinza = cv2.cvtColor(imagem2,cv2.COLOR_BGR2GRAY)

caminho_classificador_corpo = os.path.join(dir, '../cascades/fullbody.xml')
classificador_corpo = cv2.CascadeClassifier(caminho_classificador_corpo)

deteccao_corpo = classificador_corpo.detectMultiScale(imagem2_cinza,scaleFactor=1.05,minNeighbors=6)

for x,y,w,h in deteccao_corpo:
    cv2.rectangle(imagem2,(x,y),(x+w,y+h),(0,0,255),1)

print('Detecções de corpos:' + str(len(deteccao_corpo)))

janela2 = 'janela2'
cv2.imshow(janela2,imagem2)

caminho_imagem3 = os.path.join(dir, '../imagens/clock.jpg')

imagem3 = cv2.imread(caminho_imagem3)

imagem3_cinza = cv2.cvtColor(imagem3,cv2.COLOR_BGR2GRAY)

caminho_deteccao_relogio = os.path.join(dir, '../cascades/clocks.xml')

classificador_relogio = cv2.CascadeClassifier(caminho_deteccao_relogio)
#melhor fator de escala até agora: 1.0142
#vizinhos minimos:5 não deu falsos positivos
deteccao_relogio = classificador_relogio.detectMultiScale(imagem3_cinza,scaleFactor= 1.0142,minNeighbors= 5,minSize=(20,20))

for x,y,w,h in deteccao_relogio:
    cv2.rectangle(imagem3,(x,y),(x+w,y+h),(0,255,255),2)

print("Quantidade de relógios:" + str(len(deteccao_relogio)))
janela3 = 'janela3'
cv2.imshow(janela3,imagem3)
cv2.waitKey(0)



