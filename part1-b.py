# Importing Libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("ref_marker.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                   # converting image to gray scale
img_rs = cv2.resize(img_gray, (8,8))                               # resizing to 8x8
img_rs2 = img_rs.copy()
img_th = cv2.threshold(img_rs, 200,1, cv2.THRESH_BINARY)           # Thresholding
print("binary 8x8 grid\n",img_th[1])
AR_pts = img_th[1][2:6, 2:6]

# computing orientation of tag
if AR_pts[0][0] == 1:
    angle = 180
    position = "TL"

elif AR_pts[0][3] == 1:
    angle = 90
    position = "TR"


elif AR_pts[3][0] == 1:
    angle = -90
    position = "BL"

else:
    angle = 0
    position = "BR"
print(f"angle{angle} position{position}")

tagID = []
# compute id of tag
for i in range(AR_pts.shape[0]):
    for j in range(AR_pts.shape[1]):
        if i == 1 and j == 1:
            tagID.append(AR_pts[1][1])
        elif i == 1 and j == 2:
            tagID.append(AR_pts[1][2])
        elif i == 2 and j == 1:
            tagID.append(AR_pts[2][1])
        elif i == 2 and j == 2:
            tagID.append(AR_pts[2][2])

id_str = ''

for i in tagID:
    id_str = id_str + str(i)

img_cr_2 = img_th[1][3:5, 3:5]


print("id of AR tag\n", id_str )

# corner detection using good features to detect
corners = cv2.goodFeaturesToTrack(img_gray, 25, 0.01, 40)
corners = np.int0(corners)
lis = []
for j in corners:
    x,y = j.ravel()
    lis.append([x,y])
    cv2.circle(img, (x,y), 3, 255, -1)

print("corner points",lis)

plt.subplot(152), plt.imshow(img_rs, cmap="gray", vmin=0,vmax=1), plt.title("resized")
plt.subplot(153), plt.imshow(AR_pts, cmap="gray", vmin=0,vmax=1), plt.title("4x4 grid")
plt.subplot(154), plt.imshow(img_cr_2, cmap="gray", vmin=0,vmax=1), plt.title("2x2 grid")
plt.subplot(155), plt.imshow(img, cmap="gray", vmin=0,vmax=1), plt.title("corners")
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
