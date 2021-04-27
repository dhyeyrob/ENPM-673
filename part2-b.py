# Importing Libraries
import cv2
import numpy as np
from scipy.spatial import distance as dist

# Function to detect contours
def detect_contour(frame):
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thres = cv2.threshold(imgray, 190, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hierarchy = hierarchy[0]  # innermost hierarchy

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

# Function to compute the projection matrix
def projection_matrix(h, k):
    h1 = h[:, 0]
    h2 = h[:, 1]
    #h3 = h[:, 2]
    k1 = np.linalg.inv(k)
    l = 2 / (np.linalg.norm(np.matmul(k1, h1)) + np.linalg.norm(np.matmul(k1, h2)))
    bt = l * np.matmul(k1, h)
    det = np.linalg.det(bt)
    if det > 0:
        b = bt
    else:
        b = -1 * bt

    ro1 = b[:, 0]
    ro2 = b[:, 1]
    ro3 = np.cross(ro1, ro2)
    t = b[:, 2]
    R = np.column_stack((ro1, ro2, ro3, t))
    return np.dot(k, R)

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

# Function to project the cube
def project_cube(img, P):
    points = np.float32(
        [[0, 0, 0, 1], [0, 200, 0, 1], [200, 200, 0, 1], [200, 0, 0, 1], [0, 0, -200, 1], [0, 200, -200, 1],
         [200, 200, -200, 1], [200, 0, -200, 1], ])
    uv = np.matmul(points, P.T)
    z1 = uv[0][2]
    z2 = uv[1][2]
    z3 = uv[2][2]
    z4 = uv[3][2]
    z5 = uv[4][2]
    z6 = uv[5][2]
    z7 = uv[6][2]
    z8 = uv[7][2]

    pt1 = uv[0] / z1
    pt2 = uv[1] / z2
    pt3 = uv[2] / z3
    pt4 = uv[3] / z4
    pt5 = uv[4] / z5
    pt6 = uv[5] / z6
    pt7 = uv[6] / z7
    pt8 = uv[7] / z8
    uv = np.vstack((pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8))
    final_points = np.array([[uv[0][0], uv[0][1]],
                             [uv[1][0], uv[1][1]],
                             [uv[2][0], uv[2][1]],
                             [uv[3][0], uv[3][1]],
                             [uv[4][0], uv[4][1]],
                             [uv[5][0], uv[5][1]],
                             [uv[6][0], uv[6][1]],
                             [uv[7][0], uv[7][1]]])

    cv2.circle(img, (int(uv[0][0]), int(uv[0][1])), 5, (0, 255, 255), -1)
    cv2.circle(img, (int(uv[1][0]), int(uv[1][1])), 5, (0, 255, 255), -1)
    cv2.circle(img, (int(uv[2][0]), int(uv[2][1])), 5, (0, 255, 255), -1)
    cv2.circle(img, (int(uv[3][0]), int(uv[3][1])), 5, (0, 255, 255), -1)
    cv2.circle(img, (int(uv[4][0]), int(uv[4][1])), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(uv[5][0]), int(uv[5][1])), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(uv[6][0]), int(uv[6][1])), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(uv[7][0]), int(uv[7][1])), 5, (255, 0, 0), -1)

    final_points = np.int32(final_points).reshape(-1, 2)
    img = cv2.drawContours(img, [final_points[:4]], -1, (0, 175, 175), 3)
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(final_points[i]), tuple(final_points[j]), (255,0,0), 3)
        # Drawing Top layer in Red
        img = cv2.drawContours(img, [final_points[4:]], -1, (0, 0, 200), 3)


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
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


# Intrinsic properties of camera
k = np.array([[1406.08415449821, 0, 0],
              [2.20679787308599, 1417.99930662800, 0],
              [1014.13643417416, 566.347754321696, 1]]).T

print("Please enter any of the following number 0,1,2 or 3. They refer to Tag0, Tag1, Tag2 and multipleTags videos respectively")
user_input = int(input("Please enter either of 0,1,2 or 3:"))

if user_input == 0:
    cap = cv2.VideoCapture("Tag0.mp4")
    out = cv2.VideoWriter('Cubetag0.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (960, 540))
elif user_input == 1:
    cap = cv2.VideoCapture("Tag1.mp4")
    out = cv2.VideoWriter('Cubetag1.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (960, 540))
elif user_input == 2:
    cap = cv2.VideoCapture("Tag2.mp4")
    out = cv2.VideoWriter('Cubetag2.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (960, 540))
elif user_input == 3:
    cap = cv2.VideoCapture("multipleTags.mp4")
    out = cv2.VideoWriter('CubemultipleTags.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (960, 540))

#cap = cv2.VideoCapture(r"C:\Users\dhyey\Desktop\JH\ph\multipleTags.mp4")

if cap.isOpened() == False:
    print("Error loading")

w_c = np.array([[0, 0], [200, 0], [200, 200], [0, 200]]).reshape(4, 2) # world coordinates
#out = cv2.VideoWriter('CubemultipleTags.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (960, 540))
while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break
    img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)          # resizing frame
    contour, edge = detect_contour(img)                      # detect contours

    for i in range(len(contour)):
        H = homography(contour[i], w_c)                      # compute homography

        tag_frame = warp_perspective(img, H, (200, 200))     # warping the image
        angle, orientation, ID = tag_id(tag_frame)           # determine orientation, angle and id

        w_c2 = reorientation(orientation)                    # Reorienting
        wc3 = [[0, 200],
            [0, 0],
            [200, 0],
            [200, 200]]
        H_new = homography(w_c2, contour[i])                 # computing new homography
        P = projection_matrix(H_new, k)                      # computing projection matrix
        project_cube(img, P)                                 # projecting the cube
        cv2.imshow("cube", img)
        out.write(img)

    if cv2.waitKey(1) & 0xFF == 27:
        break


out.release()
cap.release()
cv2.destroyAllWindows()


