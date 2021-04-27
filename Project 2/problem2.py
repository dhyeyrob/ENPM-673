import cv2
import numpy as np
from matplotlib import pyplot as plt

# A Function For Inverse Warping
def inv_warping(img, H, dsize, w_img):
    #img = cv2.transpose(img)
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

# A Function TO Perform Warp Perspective
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

    #warp_img = cv2.transpose(warp_img)
    return warp_img

# A Function To Calculate Histogram
def calculate_histogram(top_view, thresh):
    his = [0.0] * (top_view.shape[0])
    for j in range(top_view.shape[0]):
        for i in range(top_view.shape[1]):
            if thresh[i][j] == 255:
                his[j] = his[j] + 1

    return his

# A Function To Detect Lane Pixel Coordinates
def lane_point_detect(his, top_view, thresh, data_s):

    if data_s == 1:
        coef1 = [0, 0 , 0, 0]
        coef2 = [-5, 5, -5, 5]

    elif data_s == 2:
        coef1 = [15, -15, -45, 10]
        coef2 = [-10, 20, -100, 150]
    max_hisl = 0
    max_hisr = 0
    max_k_l = 0
    max_k_r = 0

    # Detecting Left And Right Peaks From Histogram
    for k in range(len(his)):
        if k < 100:
            # mask[:, k] = (0,0,255)
            if his[k] > max_hisl:
                line_l = [k, 340]
                max_hisl = his[k]
                max_k_l = k
        elif k > 420:
            if his[k] > max_hisr:
                line_r = [k, 340]
                max_hisr = his[k]
                max_k_r = k

    left_lane_poly_pts = [[max_k_l + coef1[0], 0], [max_k_l + coef1[1], 499]]
    right_lane_poly_pts = [[max_k_r + coef1[2], 0], [max_k_r + coef1[3], 499]]

    left_fit_x = []
    left_fit_y = []
    if max_k_l < 10:
        left_fit_y = [0, 499]
        left_fit_x = [37, 41]
    right_fit_x = []
    right_fit_y = []
    if max_k_r < 400:
        right_fit_x = [440, 460]
        right_fit_y = [0, 499]
    else:
        # right_fit_x = [max_k_r - 70,max_k_r]
        # right_fit_x = [max_k_r -90, max_k_r - 60]
        right_fit_x = [max_k_r]
        right_fit_y = [499]

    for z in range(len(his)):
        if (max_k_l + coef2[0]) <= z < max_k_l:
            for x in range(top_view.shape[1]):
                if thresh[x, z] == 255:
                    left_lane_poly_pts.append([z, x])
                    left_fit_x.append(z)                    # Append X coordinates of left lane
                    left_fit_y.append(x)                    # Append Y coordinates of left lane

        elif max_k_l < z <= (max_k_l + coef2[1]):
            for x in range(top_view.shape[1]):
                if thresh[x, z] == 255:
                    left_lane_poly_pts.append([z, x])
                    left_fit_x.append(z)                    # Append X coordinates of left lane
                    left_fit_y.append(x)                    # Append Y coordinates of left lane
        elif (max_k_r + coef2[2]) <= z < max_k_r:
            for x in range(top_view.shape[1]):
                if thresh[x, z] == 255:
                    #right_lane_poly_pts.append([z, x])
                    right_fit_x.append(z)                   # Append X coordinates of right lane
                    right_fit_y.append(x)                   # Append Y coordinates of right lane

        elif max_k_r < z <= (max_k_r + coef2[3]):
            for x in range(top_view.shape[1]):
                if thresh[x, z] == 255:
                    #right_lane_poly_pts.append([z, x])
                    right_fit_x.append(z)                  # Append X coordinates of right lane
                    right_fit_y.append(x)                  # Append Y coordinates of right lane

    l_pts = np.array(left_lane_poly_pts, dtype='int32')
    r_pts = np.array(right_lane_poly_pts, dtype='int32')
    left_fit_x_arr = np.array(left_fit_x)
    left_fit_y_arr = np.array(left_fit_y)
    right_fit_y_arr = np.array(right_fit_y)
    right_fit_x_arr = np.array(right_fit_x)
    return l_pts, r_pts, left_fit_x_arr, left_fit_y_arr, right_fit_x_arr, right_fit_y_arr, max_k_l, max_k_r, coef1

# A Function To Calculate Radius Of Curvature And Vehicle Offset
def rad_of_curv(top_view, left_fit_y_arr, left_fit_x_arr, right_fit_y_arr, right_fit_x_arr):
    plot_x = np.linspace(0, top_view.shape[0] - 1, top_view.shape[0])
    left_poly_fit = np.polyfit(left_fit_y_arr, left_fit_x_arr, 2)       # Poly Fitting The Left Lane Coors
    right_poly_fit = np.polyfit(right_fit_y_arr, right_fit_x_arr, 2)    # Poly Fitting The Right lane Coors
    left_eq = left_poly_fit[0] * (plot_x ** 2) + left_poly_fit[1] * (plot_x) + left_poly_fit[2]
    right_eq = right_poly_fit[0] * (plot_x ** 2) + right_poly_fit[1] * (plot_x) + right_poly_fit[2]
    x_eval = np.max(plot_x)
    ym_per_pix = 30.0 / 720                                             # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700                                              # meters per pixel in x dimension
    left_fit_cr = np.polyfit(plot_x * ym_per_pix, left_eq * xm_per_pix, 2)
    right_fit_cr = np.polyfit(plot_x * ym_per_pix, right_eq * xm_per_pix, 2)

    # Calculating ROC of left lane
    left_curverad = ((1 + (2 * left_fit_cr[0] * x_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])

    # Calculating ROC of Right Lane
    right_curverad = ((1 + (2 * right_fit_cr[0] * x_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    # Average ROC
    mean_curv = (left_curverad + right_curverad) // 2
    lis = []
    lis2 = []
    for g in range(500):
        lis.append([int(right_eq[g]), g])
        lis2.append([int(left_eq[g]), g])
    r_newp = np.array(lis, dtype='int32')
    l_newp = np.array(lis2, dtype='int32')

    left_eq_c = left_poly_fit[0] * (x_eval ** 2) + left_poly_fit[1] * (x_eval) + left_poly_fit[2]
    right_eq_c = right_poly_fit[0] * (x_eval ** 2) + right_poly_fit[1] * (x_eval) + right_poly_fit[2]

    center_lane = (left_eq_c + right_eq_c) // 2
    img_center = top_view.shape[1] / 2
    car_offset = (img_center - center_lane) * xm_per_pix    # Calculating Car Offset, if +ve:right turn, -ve: left turn

    return right_eq, car_offset, center_lane, l_newp, r_newp, mean_curv

# Function To Display Radius Of Curvature And THe Offset
def display_txt(average_radc, car_offst, overlayed):
    curvature_info = 'Average Curvature: ' + str(average_radc) + 'm'
    cv2.putText(overlayed, (curvature_info), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (125, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(overlayed, ('Vehicle offset: ' + str(round(car_offst, 3)) + 'm'), (10, 100), cv2.FONT_HERSHEY_COMPLEX,1, (125, 0, 0), 2, cv2.LINE_AA)

# Function To Overlay Detected Lane Lines On To The Original Frame
def pre_overlay(top_view, max_k_l, max_k_r, center_lane, l_pts, r_newp, coef1, right_eq):
    shade = [[max_k_l + coef1[0], 0], [right_eq[0],0],[right_eq[125], 125],[right_eq[250], 250],[right_eq[375], 375] ,[right_eq[499], 499],[max_k_l + coef1[1], 499] ]
    shade_np = np.array(shade, dtype='int32')

    # top_view[left_fit_y, left_fit_x] = (255,0,0)
    left_curve = cv2.polylines(top_view, [l_pts], False, (0, 0, 255), 8)        # Drawing Left Lane
    right_curve = cv2.polylines(top_view, [r_newp], False, (0, 0, 255), 8)      # Drawing Right Lane
    shade_curve = cv2.fillPoly(top_view, [shade_np], (150, 150, 0))             # Shading The Region Between Lanes

    # Drawing Center Line
    cv2.arrowedLine(top_view, (int(center_lane), 499), (int(center_lane), 0), (0, 255, 0), 6)

# Main Function
def main_fn(frame, corner_points, d_s, thr_val):
    dsize = (500, 500)
    dest_mat = [(0, 0), (0, dsize[0]), (dsize[1], dsize[0]), (dsize[1], 0)]  # World Coordinates
    dest_points = np.float32(dest_mat).reshape(-1, 1, 2)
    # print(dest_points)
    Homo = cv2.findHomography(corner_points, dest_points)[0]                # Calculating Homography
    top_view = warp_perspective(frame, Homo, dsize)                         # Changing The Perspective To Top View
    tpcopy = top_view.copy()
    undistorted_img = cv2.undistort(tpcopy, K, dist)
    tp_gray = cv2.cvtColor(tpcopy, cv2.COLOR_BGR2GRAY)
    tp_blur = cv2.GaussianBlur(tp_gray, (7, 7), 2)                          # Image Smoothening
    tp_canny = cv2.Canny(tp_blur, 50, 200)
    ret, thresh = cv2.threshold(tp_gray, thr_val, 255, cv2.THRESH_BINARY)   # Binary Thresholding
    histogram = calculate_histogram(top_view, thresh)                       # Calculating The Histogram

    # Detecting Lane Pixel Coordinates
    l_pts, r_pts, left_fit_x_arr, left_fit_y_arr, right_fit_x_arr, right_fit_y_arr, max_kl, max_kr, cof1= lane_point_detect(histogram, top_view, thresh, d_s)

    # Radius Of Curvature And Find Offset
    right_eq, offset, center_line, l_newp , r_newp, mean_curvature = rad_of_curv(top_view, left_fit_y_arr, left_fit_x_arr, right_fit_y_arr, right_fit_x_arr)

    # Overlaying Lane Lines On Top View Frame
    pre_overlay(top_view, max_kl, max_kr, center_line, l_pts, r_newp, cof1, right_eq)
    Homo_real = cv2.findHomography(dest_points, corner_points)[0]           # Calculating Homography
    overlayed = inv_warping(top_view, Homo_real, (frame.shape[1], frame.shape[0]), frame)  # Inverse Warping
    display_txt(mean_curvature,offset, overlayed)                           # Diplaying Text
    return  overlayed

frame_num = 0
size = (960,770)
global K
global dist
K = np.array([[ 9.037596e+02, 0.000000e+00, 6.957519e+02], [0.000000e+00, 9.019653e+02, 2.242509e+02],[ 0.000000e+00, 0.000000e+00, 1.000000e+00]])
dist = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
result = cv2.VideoWriter('lane.avi', cv2.VideoWriter_fourcc(*'XVID'),50, size)

# User Input To Select Data Set 1 Or 2
data_set = int(input("Please enter which data set you want to take. ENTER EITHER 1 OR 2:"))

if data_set == 1:
    # Generating A Video File
    result1 = cv2.VideoWriter('new_lane_data_set_1.avi', cv2.VideoWriter_fourcc(*'XVID'), 15, (1392, 512))
    corners_source = [(585, 275), (715, 275), (950, 512), (140, 512)]  # Pre Computed Points For Homography
    corner_points1 = np.float32(corners_source).reshape(-1, 1, 2)
    th_val = 240                                                       # Binary Thresholding Value
    for i in range(0, 302):
        frame_num = i
        path_name = r"data_1-20210327T181450Z-001\data_1\data\ "
        p2 = r"C:/Users/dhyey/Desktop/JH/ph/data_1-20210327T181450Z-001/data_1/data/"
        variable_path = p2 + ("0000000000" + str(frame_num))[-10:] + ".PNG"  # Image Stitching
        frame1 = cv2.imread(variable_path)
        overlayed = main_fn(frame1,corner_points1, data_set, th_val)         # Calling The Main Function
        cv2.imshow("overlaying line", overlayed)
        result1.write(overlayed)
        # cv2.imshow("i to v",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    result1.release()
    cv2.destroyAllWindows()

elif data_set == 2:
    # Generating A Video File
    result1 = cv2.VideoWriter('new_lane_data_set_2.avi', cv2.VideoWriter_fourcc(*'XVID'), 15, (1280, 720))
    corners_source = [(510, 522), (834, 522), (1040, 680), (300, 680)]   # Precomputed Points For Homography
    corner_points2 = np.float32(corners_source).reshape(-1, 1, 2)
    th_val = 160                                                         # Binary Thresholding Value
    cap = cv2.VideoCapture(r"data_2-20210327T181509Z-001\data_2\challenge_video.MP4")

    while True:
        success, frame2 = cap.read()
        if success == True:
            overlayed = main_fn(frame2, corner_points2, data_set, th_val)  # Calling The Main Function
            cv2.imshow("overlaying line", overlayed)
            result1.write(overlayed)
            # cv2.imshow("i to v",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    result1.release()
    cap.release()
    cv2.destroyAllWindows()

