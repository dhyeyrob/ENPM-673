
import cv2
from matplotlib import pyplot as plt
import random
import numpy as np
from tqdm import *

global SEARCH_AREA
global KERNEL
SEARCH_AREA = 40
KERNEL = 11

# A function to compute depth map
def get_depth_map(disp_map, depth_map, B, f):
    h,w = disp_map.shape
    for i in range(h):
        for j in range(w):
            depth_map[i,j] = (B * f)/disp_map[i,j]
    # Rescaling to gray scale
    norm_image = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    plt.subplot(121), plt.imshow(depth_map, cmap="hot", interpolation="nearest"), plt.title("depth heat map")
    plt.subplot(122), plt.imshow(depth_map, cmap="gray", interpolation="nearest"), plt.title("depth gray map")
    plt.show()

# A function to get sum of absolute difference
def compute_sum_of_diff(pixel_vals_1, pixel_vals_2):

    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1

    return np.sum(abs(pixel_vals_1 - pixel_vals_2))

# A function to compare intensity values of pixels
def compute_difference(y, x, block_left, right_array, block_size=5):

    x_min = max(0, x - SEARCH_AREA)
    x_max = min(right_array.shape[1], x + SEARCH_AREA)
    first = True
    min_sad = None
    min_index = None
    # getting the index of pixel with minimum difference
    for x in range(x_min, x_max):
        block_right = right_array[y: y+block_size,
                                  x: x+block_size]
        s = compute_sum_of_diff(block_left, block_right)
        if first:
            min_sad = s
            min_index = (y, x)
            first = False
        else:
            if s < min_sad:
                min_sad = s
                min_index = (y, x)

    return min_index

# A function to compute disparity map
def get_disparity_map(left_rec, right_rec):
    left_array = np.asarray(left_rec)
    right_array = np.asarray(right_rec)
    left_array = left_array.astype(int)
    right_array = right_array.astype(int)
    if left_array.shape != right_array.shape:
        print("warning")
    h, w = left_array.shape
    disparity_map = np.zeros((h, w))
    # Sliding over each pixel location of left image and comparing with right image pixels
    for y in tqdm(range(KERNEL, h-KERNEL)):
        for x in range(KERNEL, w-KERNEL):
            block_left = left_array[y:y + KERNEL,
                                    x:x + KERNEL]
            # Comparing intensity values
            min_index = compute_difference(y, x, block_left,right_array,block_size=KERNEL)
            disparity_map[y, x] = abs(min_index[1] - x)

    # Rescaling to gray scale
    norm_image = cv2.normalize(disparity_map, None, alpha= 0, beta=255, norm_type= cv2.NORM_MINMAX, dtype= cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    plt.subplot(121), plt.imshow(disparity_map, cmap="hot", interpolation= "nearest"), plt.title("diparity heat map")
    plt.subplot(122), plt.imshow(disparity_map, cmap="gray", interpolation= "nearest"), plt.title("disparity gray map")
    plt.show()
    plt.imshow(norm_image, cmap='gray', interpolation='nearest')
    plt.show()

    return disparity_map

# A function to compute camera pose from E matrix
def ExtractCameraPose(E):

    u, s, v = np.linalg.svd(E, full_matrices=True)
    w = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]]).reshape(3, 3)

    # Computing 4 possible camera poses
    c1 = u[:, 2].reshape(3, 1)
    r1 = np.dot(np.dot(u, w), v).reshape(3, 3)
    c2 = -u[:, 2].reshape(3, 1)
    r2 = np.dot(np.dot(u, w), v).reshape(3, 3)
    c3 = u[:, 2].reshape(3, 1)
    r3 = np.dot(np.dot(u, w.T), v).reshape(3, 3)
    c4 = -u[:, 2].reshape(3, 1)
    r4 = np.dot(np.dot(u, w.T), v).reshape(3, 3)
    if np.linalg.det(r1) < 0:
        c1 = -c1
        r1 = -r1
    if np.linalg.det(r2) < 0:
        c2 = -c2
        r2 = -r2
    if np.linalg.det(r3) < 0:
        c3 = -c3
        r3 = -r3
    if np.linalg.det(r4) < 0:
        c4 = -c4
        r4 = -r4
    cam_center = np.array([c1, c2, c3, c4])
    cam_rotation = np.array([r1, r2, r3, r4])
    return cam_center, cam_rotation

# A function to get matching features
def detect_matching_feat(image1_gray, image2_gray):
    # using SIFT detector
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1_gray, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2_gray, None)
    img_1 = cv2.drawKeypoints(image1_gray, keypoints_1, image1)
    img_2 = cv2.drawKeypoints(image2_gray, keypoints_2, image2)

    # Getting the matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
    req = []
    left_pts = []
    right_pts = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            left_pts.append(keypoints_1[m.queryIdx].pt)
            right_pts.append(keypoints_2[m.trainIdx].pt)
            req.append(m)
    left_feat_n = np.array(left_pts)
    right_feat_n = np.array(right_pts)
    img_3 = cv2.drawMatches(image1_gray, keypoints_1, image2_gray, keypoints_2, req, image2_gray, flags=2)
    plt.plot(), plt.imshow(img_3, cmap="gray"), plt.title("sift1")
    plt.show()

    return left_feat_n, right_feat_n

# A function to draw epipolar lines
def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255,3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int,[c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1,(x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1,tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

# A function to rectify and apply perspective transform on images
def stereo_rectify(points1, points2, F_nw, imgL, imgR):
    h1, w1 = imgL.shape
    h2, w2 = imgR.shape
    thresh = 0
    # Getting homography matrices H1 and H2 for perpective transform
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(points1), np.float32(points2), F_nw, imgSize=(w1, h1), threshold=thresh)

    # Warping the images
    imgL_undistorted = cv2.warpPerspective(imgL, H1, (w1, h1))
    imgR_undistorted = cv2.warpPerspective(imgR, H2, (w2, h2))
    F_re = np.dot(H2.T, (np.dot(F_nw, np.linalg.inv(H1))))

    # Getting the feature matches of the warped images
    poi1_r , poi2_r = detect_matching_feat(imgL_undistorted, imgR_undistorted)
    poi1_re = np.int32(poi1_r)
    poi2_re = np.int32(poi2_r)

    # Drawing epipolar lines on the warped images
    lines2 = cv2.computeCorrespondEpilines(poi1_re.reshape(-1, 1, 2), 2, F_re)
    lines2 = lines2.reshape(-1, 3)
    img5, img6 = drawlines(imgR_undistorted, imgL_undistorted, lines2, poi2_re, poi1_re)

    lines1 = cv2.computeCorrespondEpilines(poi2_re.reshape(-1, 1, 2), 2, F_re)
    lines1 = lines1.reshape(-1, 3)
    img3, img4 = drawlines(imgL_undistorted, imgR_undistorted, lines1, poi1_re, poi2_re)

    plt.subplot(121), plt.imshow(img3, cmap="gray"), plt.title("sift1 rectified")
    plt.subplot(122), plt.imshow(img5, cmap="gray"), plt.title("sift2 rectified")
    plt.show()

    return imgL_undistorted, imgR_undistorted

# A function to compute Essential matrix E
def get_essential_mat(k0, k1, F):
    E = np.dot(k1.T, np.dot(F, k0))
    u, s, v = np.linalg.svd(E)

    # Change the singular values of E matrix to 1,1,0
    s_new = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]).reshape(3, 3)
    final_E = np.dot(u, np.dot(s_new, v))
    return final_E

# A function to compute Fundamental matrix
def get_fundamental_mat(l_pts, r_pts):
    A = np.empty((8, 9))
    for i in range(len(l_pts)- q):
        x1 = l_pts[i][0]
        x2 = r_pts[i][0]
        y1 = l_pts[i][1]
        y2 = r_pts[i][1]
        A[i] = np.array([x1 * x2, x2 * y1, x2,
                         y2 * x1, y2 * y1, y2,
                         x1, y1, 1])
    # Compute F matrix by evaluating SVD
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # Enforce the F matrix to rank 2
    U_n, S_n, V_n = np.linalg.svd(F)
    S2 = np.array([[S_n[0], 0, 0], [0, S_n[1], 0], [0, 0, 0]])
    F = np.dot(np.dot(U_n, S2), V_n)
    return F

# RANSAC Function TO Get The Best F Matrix
def Ransac(left_f_p, right_f_p, sample_num = 3,threshold = 0.001 ):
    #best_fit_model = None
    #total_iter = math.inf
    total_iter = 990
    iter_done = 0
    inliers = 0
    max_inliers = 0
    desired_prob = 0.99
    max_count = 0
    '''for i in range(len(x)):
        lis.append(x[i],y[i])
    '''
    #total_data = len(left_f_p)
    v = len(left_f_p)
    optimal_F = np.zeros((3, 3))
    while iter_done < total_iter:
        counter = 0
        x = random.sample(range(v - 1), 8)
        F = get_fundamental_mat(left_f_p[x], right_f_p[x])
        left_optimal_poi = []
        right_optimal_poi = []
        for z in range(v):
            right_r_p = np.array([right_f_p[z,0], right_f_p[z,1], 1])
            left_r_p = np.array([left_f_p[z,0], left_f_p[z,1], 1])
            err = np.dot(right_r_p.T, np.dot(F, left_r_p))

            if err < threshold:
                left_optimal_poi.append(left_f_p[z])
                right_optimal_poi.append(right_f_p[z])
                counter = counter + 1

        temp_left = np.array(left_optimal_poi)
        temp_right = np.array(right_optimal_poi)

        if counter > max_count:
            max_count = counter
            final_left_optimal = temp_left
            final_right_optiml = temp_right
            optimal_F = F
        iter_done = iter_done + 1

    return optimal_F, final_left_optimal, final_right_optiml

global q
print("please select either of dataset 1, 2 or 3")
a = int(input("enter either of 1, 2 or 3:"))
if a == 1:
    image1 = cv2.imread(r"Dataset 1\im0.png")
    image2 = cv2.imread(r"Dataset 1\im1.png")
    q = 1
    k_0 = [[5299.313, 0, 1263.818], [0, 5299.313, 977.763], [0, 0, 1]]
    k_1 = [[5299.313, 0, 1438.004], [0, 5299.313, 977.763], [0, 0, 1]]
    B = 177.28
    f = 5299.313

elif a == 2:
    image1 = cv2.imread(r"Dataset 2\im0.png")
    image2 = cv2.imread(r"Dataset 2\im1.png")
    q = 0
    k_0 = [[4396.869, 0, 1353.072],[0, 4396.869, 989.702],[ 0 ,0 ,1]]
    k_1 = [[4396.869, 0, 1538.86],[0, 4396.869, 989.702], [0, 0, 1]]
    B = 144.049
    f = 4396.869

elif a == 3:
    image1 = cv2.imread(r"Dataset 3\im0.png")
    image2 = cv2.imread(r"Dataset 3\im1.png")
    q = 0
    k_0 = [[5806.559, 0, 1429.219], [0, 5806.559, 993.403], [0, 0, 1]]
    k_1 = [[5806.559, 0, 1543.51], [0, 5806.559, 993.403], [0, 0, 1]]
    B = 174.019
    f = 5806.559


image1_gra = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) # Convert image to gray scale
image2_gra = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) # Convert image to gray scale
left_feat , right_feat = detect_matching_feat(image1_gra, image2_gra)  # Detect Matching Features
F_new, l_f_p , r_f_p = Ransac(left_feat, right_feat)                   # Using RANSAC find optimal F matrix
print("Fundamental Matrix", F_new)
k0 = np.array(k_0)
k1 = np.array(k_1)
E = get_essential_mat(k0, k1, F_new)                                   # Compute Essential Matrix From F matrix
print("ESSENTIAL MATRIX",E)
cam_c , cam_r = ExtractCameraPose(E)                                   # From E matrix compute camera pose
print("camera pose:")
print(cam_c, cam_r)

# Rectify the image and change change the perpsective of image
img_left_rect, img_right_rect = stereo_rectify(left_feat, right_feat, F_new, image1_gra, image2_gra)
obtained_disp = get_disparity_map(img_left_rect, img_right_rect)       # compute disparity
h_n , w_n = obtained_disp.shape
depth_mp = np.zeros((h_n,w_n))
get_depth_map(obtained_disp, depth_mp, B , f)                          # Compute the depth
