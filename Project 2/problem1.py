import numpy as np
import cv2
from matplotlib import pyplot as plt

def hist_eq(frame):
    # Extract R,G and B channels from the image frame
    b, g, r = cv2.split(frame)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])

    # calculate cdf function
    cdf_b = np.cumsum(h_b)
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)

    # Masking pixels with value=0 and replacing them with mean of the pixel values
    cdf_m_b = np.ma.masked_equal(cdf_b, 0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / (cdf_m_b.max() - cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b, 0).astype('uint8')

    cdf_m_g = np.ma.masked_equal(cdf_g, 0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / (cdf_m_g.max() - cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g, 0).astype('uint8')


    cdf_m_r = np.ma.masked_equal(cdf_r, 0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / (cdf_m_r.max() - cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r, 0).astype('uint8')

    # merge the images back into three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]

    img_back = cv2.merge((img_b, img_g, img_r))

    return img_back

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
result1 = cv2.VideoWriter('hist_eq.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (400, 400))
img = cv2.VideoCapture(r"C:\Users\dhyey\Desktop\JH\ph\Night Drive - 2689.mp4")

while True:
    success, img_frame = img.read()

    if success == True:

        hist,bins = np.histogram(img_frame.flatten(),256,[0,256])               # Flattening the frame and getting histogram
        plt.hist(img_frame.flatten(),256,[0,256], color = 'r')                  # Original Histogram
        plt.xlim([0,256])
        #plt.show()                                                             # UNCOMMENT TO VIEW ORIGINAL HISTOGRAM
        frame_rsz = cv2.resize(img_frame, (400, 400))
        image_eq = hist_eq(frame_rsz)              # Calling Histogram Eq function
        hist_n, bins_n = np.histogram(image_eq.flatten(), 256, [0, 256])
        plt.hist(image_eq.flatten(), 256, [0, 256], color='r')             # Equalized Histogram
        plt.title("Equalized Histogram")
        plt.xlim([0, 256])
        #plt.show()                                                        # UNCOMMENT TO VIEW EQUALIZED HISTOGRAM
        cv2.imshow("hist equalized", image_eq)
        result1.write(image_eq)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
result1.release()
cv2.destroyAllWindows()
