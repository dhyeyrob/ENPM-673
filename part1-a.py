# Importing Libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to detect the AR tag
def frame_det(path):
    path1 = path
    cap = cv2.VideoCapture(path1)
    fl = True
    while fl == True:
        for i in range(60):
            success, img = cap.read()

            if success == True:
                imgcnt = img.copy()                                         # making a copy of the captured frame
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            # converting our image frame to gray scale

        RET, img_ts= cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)   # thresholding
        img_ts_2 = img_ts.copy()
        nimg = np.fft.fft2(img_ts)                                          # applying fft
        nimg2 = np.fft.fftshift(nimg)

        rows, cols = img_ts.shape
        crow, ccol = rows/2, cols / 2
        nimg2[int(crow) - 100:int(crow) + 100, int(ccol) - 200:int(ccol) + 200] = 0  # applying hpf
        f_ishift = np.fft.ifftshift(nimg2)                                           # applying inv fft
        img_back = np.fft.ifft2(f_ishift)
        img_back2 = np.abs(img_back)

        img_back_3 = np.array(img_back2, dtype=int)

        plt.subplot(131), plt.imshow(img, cmap='gray')
        plt.title('Input Image      '), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(img_back2, cmap='gray'),\
        plt.xticks([]), plt.yticks([])
        plt.imshow(img_back2, "gray"), plt.title(" Image After Applying HPF")
        plt.show()
        img_back2 = img_back2.astype('uint8')                                      # changing image type
        img_bk = img_back2.copy()
        imgbl = cv2.GaussianBlur(img_back2, (7, 7), 5)
        imgcanny = cv2.cv2.Canny(imgbl, 50, 100)
        # detecting contours
        contours, hierarchy = cv2.findContours(imgcanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE   )
        Hierarchy = hierarchy[0]
        for komp in zip(contours,Hierarchy):
            currentContour = komp[0]
            currentHierarchy = komp[1]
            x,y,w,h = cv2.boundingRect(currentContour)
            arean = cv2.contourArea(currentContour)
            # drawing the inner contour
            if currentHierarchy[2] < 0 and arean < 20:
                cv2.rectangle(imgcnt, (x,y), (x+w,y+h),(0,255,0),3)

        cv2.imshow("ar tag", imgcnt)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        fl = False

frame_det("Tag1.mp4")
