import cv2
import numpy as np
import os

def getCountour(img,count):
    countours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in countours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area > 40:
            count += 1
            x, y, w, h = cv2.boundingRect(cnt)

            roi = imgCountour[y:y + h, x:x + w]

            output_path = os.path.join(output_dir, f'photo_{count}.png')
            cv2.imwrite(output_path, roi)

    return count




input_dir = ''

# Uzyskaj listę plików w folderze
files = os.listdir(input_dir)

# Zlicz liczbę plików
num_files = len(files)
licznik = 0

for i in range(num_files-1):
    if i < 10:
        input_path = os.path.join(input_dir, f'000{i}.jpg')
    else:
        input_path = os.path.join(input_dir, f'00{i}.jpg')

    img = cv2.imread(input_path)
    #img = cv2.resize(img, dsize=(1000, 800), interpolation=cv2.INTER_LANCZOS4)

    imgCountour = img.copy()

    imgBlank = np.zeros_like(img)

    # Ustaw katalog wyjściowy
    output_dir = ''

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgGray,250,255,cv2.THRESH_BINARY_INV)

    imgBlur = cv2.GaussianBlur(thresh,(5,5),0)
    imgCanny = cv2.Canny(imgBlur,120,120)

    licznik = getCountour(imgCanny,licznik)

print(licznik)