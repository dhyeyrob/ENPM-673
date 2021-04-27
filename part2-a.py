# importing Libraries
import cv2
import numpy as np
from scipy.spatial import distance as dist

# Function to detect contours
def detect_contour(frame):
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thres = cv2.threshold(imgray, 190, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hierarchy = hierarchy[0]

    corner_list = list()
    contour_ = list()

    contour_list = list()

    for component in zip(contours, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]
        perimeter = cv2.arcLength(currentContour, True)
        appr = cv2.approxPolyDP(currentContour, 0.015 * perimeter, True)
        if len(appr) == 4 and currentHierarchy[3] != -1:
            corner_list.append(appr)

    for corners in corner_list:
        area = cv2.contourArea(corners)
        if area > 100 and area < 7000:
            cv2.drawContours(frame, [corners], 0, (0, 255, 0), 2)
            cnt = np.array([corners[0][0], corners[1][0], corners[2][0], corners[3][0]])
            contour_list.append(order_pts(cnt))

    return contour_list, thres

# Function to calculate homography
def homography(p1, p2):
    A = []
    for val in range(0, len(p1)):
        x_1, y_1 = p1[val][0], p1[val][1]
        x_2, y_2 = p2[val][0], p2[val][1]
        A.append([x_1, y_1, 1, 0, 0, 0, -x_2 * x_1, -x_2 * y_1, -x_2])
        A.append([0, 0, 0, x_1, y_1, 1, -y_2 * x_1, -y_2 * y_1, -y_2])

    A = np.array(A)
    u, S, Vh = np.linalg.svd(A)
    l = Vh[-1, :] / Vh[-1, -1]
    H = np.reshape(l, (3, 3))
    return H


# Function to compute warp perspective
def warp_perspective(img, H, dsize):
    img = cv2.transpose(img)
    r, c = dsize
    H = np.linalg.inv(H)
    warp_img = np.zeros((r, c, img.shape[2]), dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            uv = np.dot(H, [i, j, 1])
            i2, j2, _ = (uv / uv[2] + 0.3).astype(int)
            if i2 >= 0 and i2 < img.shape[0]:
                if j2 >= 0 and j2 < img.shape[1]:
                    warp_img[i, j] = img[int(i2), int(j2)]

    warp_img = cv2.transpose(warp_img)
    return warp_img

# Function to compute tagid and orientation of tag
def tag_id(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    crop = img[50:150, 50:150]
    _, crop = cv2.threshold(crop, 200, 255, 0)

    cropT = cv2.transpose(crop)

    ARtag = np.zeros((4, 4), dtype='int')
    row = 0
    _pts = list()

    for i in range(0, cropT.shape[0], 25):
        col = 0
        for j in range(0, cropT.shape[1], 25):

            cB = 0
            cW = 0
            for px in range(0, 25):
                for py in range(0, 25):

                    if cropT[i + px, j + py] == 0:
                        cB += 1
                    else:
                        cW += 1

            _pts.append(cW)

    AR_pts = np.transpose(np.array(_pts).reshape(4, 4))
    m = np.mean(AR_pts)

    for i in range(4):
        for j in range(4):
            if AR_pts[i, j] < int(m):
                AR_pts[i, j] = 0
            else:
                AR_pts[i, j] = 1

    tagID = list()
    for i in range(AR_pts.shape[0]):
        for j in range(AR_pts.shape[1]):
            if i == 1 and j == 1:
                tagID.append(AR_pts[i, j])
            elif i == 1 and j == 2:
                tagID.append(AR_pts[i, j])
            elif i == 2 and j == 1:
                tagID.append(AR_pts[i, j])
            elif i == 2 and j == 2:
                tagID.append(AR_pts[i, j])
    tagIDbin = tagID[::-1]
    tagIDdec = tagIDbin[0] * 8 + tagIDbin[1] * 4 + tagIDbin[2] * 2 + tagIDbin[3] * 1

    if AR_pts[0, 0] == 1:
        angle = 180
        position = "TL"
    elif AR_pts[3, 0] == 1:
        angle = -90
        position = "BL"
    elif AR_pts[0, 3] == 1:
        angle = 90
        position = "TR"
    else:
        angle = 0
        position = "BR"

    return angle, position, tagIDdec


# Function to reorient the image
def reorientation(pose):
    if pose == "BR":
        p1 = np.array([
            [0, 0],
            [200, 0],
            [200, 200],
            [0, 200]])
        return p1
    elif pose == "TR":
        p1 = np.array([
            [200, 0],
            [200, 200],
            [0, 200],
            [0, 0]])
        return p1
    elif pose == "TL":
        p1 = np.array([
            [200, 200],
            [0, 200],
            [0, 0],
            [200, 0]])
        return p1

    elif pose == "BL":
        p1 = np.array([
            [0, 200],
            [0, 0],
            [200, 0],
            [200, 200]])
        return p1

# perform inverse warping
def inv_warping(img, H, dsize, w_img):
    img = cv2.transpose(img)
    r, c = dsize
    w_img = cv2.transpose(w_img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            uv = np.dot(H, [i, j, 1])
            i2, j2, _ = (uv / uv[2] + 0.3).astype(int)
            if i2 >= 0 and i2 < r:
                if j2 >= 0 and j2 < c:
                    w_img[int(i2), int(j2)] = img[i, j]
    w_img = cv2.transpose(w_img)
    return w_img


# Function to order the points
def order_pts(pts):
    # sort the points based on their x-coordinates
    x = pts[np.argsort(pts[:, 0]), :]

    left = x[:2, :]
    right = x[2:, :]

    left = left[np.argsort(left[:, 1]), :]
    (tl, bl) = left

    D = dist.cdist(tl[np.newaxis], right, "euclidean")[0]
    (br, tr) = right[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


print("Please enter any of the following number 0,1,2 or 3. They refer to Tag0, Tag1, Tag2 and multipleTags videos respectively")
user_input = int(input("Please enter either of 0,1,2 or 3:"))
if user_input == 0:
    cap = cv2.VideoCapture("Tag0.mp4")
    out = cv2.VideoWriter('testudotag0.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (960, 540))
elif user_input == 1:
    cap = cv2.VideoCapture("Tag1.mp4")
    out = cv2.VideoWriter('testudotag1.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (960, 540))
elif user_input == 2:
    cap = cv2.VideoCapture("Tag2.mp4")
    out = cv2.VideoWriter('testudotag2.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (960, 540))
elif user_input == 3:
    cap = cv2.VideoCapture("multipleTags.mp4")
    out = cv2.VideoWriter('testudomultipleTags.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (960, 540))

#cap = cv2.VideoCapture(r"C:\Users\dhyey\Desktop\JH\ph\Tag2.mp4")

# path for the image to be superimposed
frame2 = cv2.imread("testudo.PNG")

testudo = cv2.resize(frame2, (200, 200))                                   # resizing the testudo image

if cap.isOpened() == False:
    print("Error loading")
Frame = 0

w_c = np.array([[0, 0], [200, 0], [200, 200], [0, 200]]).reshape(4, 2)    # world coordinates

if cap.isOpened() == False:
    print("Error loading")
Frame = 0

w_c = np.array([[0, 0], [200, 0], [200, 200], [0, 200]]).reshape(4, 2)   # world coordinates
# out = cv2.VideoWriter('testudoTag2.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (960, 540))
testudo_list = list()

# Specify the path of the output video


while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break
    img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)                    # resizing the frame
    # Finding the contours of the AR Tag
    contour, edge = detect_contour(img)

    for i in range(len(contour)):
        H = homography(contour[i], w_c)                               # computing homography
        tag = warp_perspective(img, H, (200, 200))                    # implementing warp perspective
        angle, orientation, ID = tag_id(tag)                          # computing angle, orientation, id
        w_c2 = reorientation(orientation)                             # reorienting
        H_testudo = homography(w_c2, contour[i])                      # computing new homography
        warped_testudo = inv_warping(testudo, H_testudo, (img.shape[1], img.shape[0]), img)  # inverse warping
        cv2.imshow("testudo img",warped_testudo)
        out.write(warped_testudo)


    if cv2.waitKey(1) & 0xFF == 27:
        break

out.release()
cap.release()
cv2.destroyAllWindows()
