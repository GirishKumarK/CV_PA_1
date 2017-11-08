'''
READ ME FIRST :
    PLEASE COMMENT/UNCOMMENT THE RESPECTIVE IMAGES BEFORE RUNNING
    FOR A VALID REASON THAT I DO NOT WISH TO PICTURE BOMB THE SCREEN
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time

start = time.time()

# Open and Load the Image
'''COMMENT/UNCOMMENT THE RESPECTIVE IMAGES BEFORE RUNNING HERE'''
col_img = cv.imread('uneq_hist_1.jpg')
#col_img = cv.imread('uneq_hist_2.jpg')
#col_img = cv.imread('uneq_hist_3.jpg')
#col_img = cv.imread('test1.jpg')
#col_img = cv.imread('test2.jpg')
#col_img = cv.imread('test3.jpg')
#col_img = cv.imread('test4.jpg')
#col_img = cv.imread('test5.jpg')
col_img_size = col_img.shape
cv.imshow('Original Image', col_img)
cv.waitKey(1)

# Convert the Three Channel Image to Single Channel
g_img = np.uint8((0.30 * col_img[:,:,0]) + (0.59 * col_img[:,:,1]) + (0.11 * col_img[:,:,2]))
cv.imshow('Grayscale Image', g_img)
cv.waitKey(1)


def get_Hist(image):
    # Variables
    L = 256
    # Gray Value Distribution
    gval = np.array([i for i in range(0, L)]).reshape(1, -1)
    # Get the Histogram Distribution
    hist = list()
    for n in range(0, L):
        hist.append(np.count_nonzero(image == n))
    # Histogram Distribution
    hist = np.array(hist).reshape(1, -1)
    # Cumulative Density Distribution
    cdf = np.cumsum(hist).reshape(1, -1)
    # Return Distributions
    return [gval[0], hist[0], cdf[0]]

def plot_Hist(params):
    image, string = params[0], params[1]
    # Get the Distributions
    gray, hist, cdf = get_Hist(image)
    # Plot Histogram
    plt.figure(string)
    plt.step(gray, hist, 'r')
    plt.xlabel('Gray Values')
    plt.ylabel('No. of Gray Value Occurences')
    plt.title(string)
    # Plot Histogram CDF
    string = string + ' CDF'
    plt.figure(string)
    plt.step(gray, cdf, 'r')
    plt.xlabel('Gray Values')
    plt.ylabel('No. of Gray Value Occurences')
    plt.title(string)
    # Return Image and Histogram Distribution
    return [image, cdf]

def Eq_Hist(params):
    # Histogram Equalization Function
    image, cdf = params[0], params[1]
    # Variables
    L, M, N = float(256), float(image.shape[1]), float(image.shape[0])
    # Get the Equalization Distribution
    hist_Eq = [0 for i in range(0, int(L))]
    # Get Minimum Value of CDF greater than Zero
    cdf_min = float(min(i for i in cdf if i > 0))
    # Get the Equalized Histogram using the Formula
    for n in range(0, int(L)):
        hist_Eq[n] = round(((L - 1) * (cdf[n] - cdf_min))/((M * N) - 1))
    # Again, if any value <= 0, put it to 0
    hist_Eq = np.array(hist_Eq)
    hist_Eq[hist_Eq <= 0.0] = 0.0
    # Return Image and Equalized Histogram Distribution
    return [image, hist_Eq]

def Clip_Hist(params):
    # Get Original Image CDF Distribution
    cdf_clip = params[0]
    # Variables alpha, beta, b
    alpha, beta, b = 50, 2, 150
    # Gray Values 0 through alpha
    cdf_clip[:alpha] = 0
    # Gray Values alpha through b
    cdf_clip[alpha:b] = [(beta * (h - alpha)) for h in range(alpha, b)]
    # Gray Values b through 255 (end)
    cdf_clip[b:] = (beta * (b - alpha))
    # If any value <= 0, put it to 0
    cdf_clip[cdf_clip <= 0] = 0
    # Return Clipped CDF Distribution
    return cdf_clip

def Rng_Comp(params):
    # Get Original Image CDF Distribution
    cdf_rc = params[0]
    # Get the C value
    c = params[1]
    # Get the CDF Distribution after applying the Formula
    cdf_rc = [(c * np.log10([1 + gray])) for gray in cdf_rc]
    # Return Range Compressed CDF Distribution
    return np.uint8(cdf_rc)

def new_Img(params):
    image, new_hist = params[0], params[1]
    # Function to Obtain the New Image
    new_img = np.zeros(image.shape)
    # Match the Pixel Locations of New Image with Reference Image Pixel Values
    for r in range(0, image.shape[0]):
        for c in range(0, image.shape[1]):
            new_img[r, c] = new_hist[image[r, c]]
    # Return the New Image
    return new_img

def plot_HistFxn(params):
    new_hist, string = params[0], params[1]
    gray = np.array([i for i in range(0, 256)]).reshape(1, -1)
    gray = gray[0]
    # Plot Histogram
    plt.figure(string)
    plt.step(gray, new_hist, 'r')
    plt.xlabel('Gray Values')
    plt.ylabel('No. of Gray Value Occurences')
    plt.title(string)
    return None

def show_NewImg(params):
    # Obtain the Parameters - Image and Title
    new_img, string = params[0], params[1]
    # Output the Image
    cv.imshow(string, np.uint8(new_img))
    # Return Nothing
    return None


# Equalization
image, cdf = plot_Hist([g_img, 'Image Histogram Before Equalization'])
image, new_hist = Eq_Hist([image, cdf])
plot_HistFxn([new_hist, 'Histogram Equalization Function'])
new_img = new_Img([image, new_hist])
new_img, new_cdf = plot_Hist([new_img, 'Image Histogram After Equalization'])
show_NewImg([new_img, 'Image After Histogram Equalization'])

# Clipping
image, cdf = plot_Hist([g_img, 'Image Histogram Before Clipping'])
hist_clip = Clip_Hist([cdf])
plot_HistFxn([hist_clip, 'Histogram Clipping Function'])
new_img = new_Img([image, hist_clip])
new_img, new_cdf = plot_Hist([new_img, 'Image Histogram After Clipping'])
show_NewImg([new_img, 'Image After Histogram Clipping'])

# Range Compression
C_vals = [1, 10, 100, 1000, 10000]
for c in C_vals:
    image, cdf = plot_Hist([g_img, 'Image Histogram Before Range Compression C = ' + str(c)])
    hist_rc = Rng_Comp([cdf, c])
    plot_HistFxn([hist_rc, 'Histogram Range Compression Function C = ' + str(c)])
    new_img = new_Img([image, hist_rc])
    new_img, new_cdf = plot_Hist([new_img, 'Image Histogram After Range Compression C = ' + str(c)])
    show_NewImg([new_img, 'Image After Histogram Range Compression C = ' + str(c)])


end = time.time()
time_taken = (end - start)/60

# End of File