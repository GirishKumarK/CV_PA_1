import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import cv2 as cv
import time

start = time.time()

# Open and Load the Images
ip_img_1 = cv.imread('susan_input1.png')
ip_img_2 = cv.imread('susan_input2.png')
cv.imshow('Original Image 1', ip_img_1)
cv.waitKey(1)
cv.imshow('Original Image 2', ip_img_2)
cv.waitKey(1)

# Convert the Three Channel Image to Single Channel
g_img_1 = np.float64((0.30 * ip_img_1[:,:,0]) + (0.59 * ip_img_1[:,:,1]) + (0.11 * ip_img_1[:,:,2]))
g_img_2 = np.float64((0.30 * ip_img_2[:,:,0]) + (0.59 * ip_img_2[:,:,1]) + (0.11 * ip_img_2[:,:,2]))
cv.imshow('Gray Image 1', np.uint8(g_img_1))
cv.waitKey(1)
cv.imshow('Gray Image 2', np.uint8(g_img_2))
cv.waitKey(1)
I1_size = g_img_1.shape
I2_size = g_img_2.shape

def filt_Median(params):
    # Median Filter Odd Dimension Kernel
    img = params[0]
    dim = int(params[1])
    pad_img = np.lib.pad(img, ((int((dim-1)/2), int((dim-1)/2)), (int((dim-1)/2), int((dim-1)/2))), 'constant', constant_values = 0)
    for r in range(int((dim-1)/2), int(pad_img.shape[0]-(int((dim-1)/2)))):
        for c in range(int((dim-1)/2), int(pad_img.shape[1]-(int((dim-1)/2)))):
            vals = list()
            for i in range(int(r-(int((dim-1)/2))), int(r+(int((dim-1)/2)))+1):
                for j in range(int(c-(int((dim-1)/2))), int(c+(int((dim-1)/2)))+1):
                    vals.append(pad_img[i][j])
            med = np.median(np.array(vals))
            pad_img[r, c] = med
    filt_img = pad_img[(int((dim-1)/2)):int(pad_img.shape[0]-(int((dim-1)/2))), (int((dim-1)/2)):int(pad_img.shape[1]-(int((dim-1)/2)))]
    return filt_img

def filt_Mean(params):
    # Mean Filter Odd Dimension Kernel
    img = params[0]
    dim = int(params[1])
    pad_img = np.lib.pad(img, ((int((dim-1)/2), int((dim-1)/2)), (int((dim-1)/2), int((dim-1)/2))), 'edge')
    for r in range(int((dim-1)/2), int(pad_img.shape[0]-(int((dim-1)/2)))):
        for c in range(int((dim-1)/2), int(pad_img.shape[1]-(int((dim-1)/2)))):
            vals = list()
            for i in range(int(r-(int((dim-1)/2))), int(r+(int((dim-1)/2)))+1):
                for j in range(int(c-(int((dim-1)/2))), int(c+(int((dim-1)/2)))+1):
                    vals.append(pad_img[i][j])
            mean = np.mean(np.array(vals))
            pad_img[r, c] = mean
    filt_img = pad_img[(int((dim-1)/2)):int(pad_img.shape[0]-(int((dim-1)/2))), (int((dim-1)/2)):int(pad_img.shape[1]-(int((dim-1)/2)))]
    return filt_img

def filt_Box(params):
    # Box Blur Kernel
    img = params[0]
    dim = int(params[1])
    mul = params[2]
    box = np.ones([dim, dim], dtype='float').reshape(dim, dim)
    box = np.divide(box, mul**2, dtype='float')
    box = np.array(box).reshape(dim, dim)
    filt_img = ndi.convolve(img, box)
    return filt_img

def filt_Gaussian(params):
    # Gaussian Smoothing Odd Value Kernel
    img = params[0]
    sigma = params[1]
    # Notes : Sigma Matters! Don't take whole numbers Eg: 2 messes up, but 1.999 is awesome!
    g = np.array([1, 2, 1, 2, 4, 2, 1, 2, 1]).reshape(3, 3)
    g_m = np.exp(-(g**2)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
    g_m = np.array(g_m).reshape(3, 3)
    # Take Derivative Mask Here If Needed
    filt_img = ndi.convolve(img, g_m)
    return filt_img

def filt_Sharp(params):
    # Image Sharpening Kernel
    img = params[0]
    mul = params[1]
    sharp = np.array([0, -1, 0, -1, 5, -1, 0, -1, 0]).reshape(3, 3)
    sharp = mul * sharp
    filt_img = ndi.convolve(img, sharp)
    return filt_img

def supr_8conn(img):
    # Suppression Function based on 8 Connectivity
    img = np.lib.pad(img, ((1, 1), (1, 1)), 'constant', constant_values = 0)
    for r in range(1, img.shape[0]-1):
        for c in range(1, img.shape[1]-1):
            if ((img[r, c] >= img[r, c+1]) and (img[r, c-1] == 0)):
                img[r, c+1] = 0
            if ((img[r, c] >= img[r+1, c]) and (img[r-1, c] == 0)):
                img[r+1, c] = 0
            if ((img[r, c] >= img[r+1, c+1]) and (img[r-1, c-1] == 0)):
                img[r+1, c+1] = 0
    supr_img = img[1:img.shape[0]-1, 1:img.shape[1]-1]
    return supr_img

def do_Susan(params):
    # Obtain the Parameters from the Function
    image = params[0]
    normz = params[1]
    gm_th = params[2]
    br_th = params[3]
    cd_th = params[4]
    # Upscale Image with Normalization Value
    image = image + normz
    # Get Image Size
    m, n = image.shape
    # Kernal Size Square
    k_size = (7, 7)
    # Kernal Side Measure
    k_side = 7.0
    # Create a circular Kernel Mask
    k_mask = np.ones(k_size).reshape(k_size)
    for i in range(-3, 4):
        for j in range(-3, 4):
            if (round(np.sqrt((i ** 2) + (j ** 2))) > 3):
                k_mask[i+3, j+3] = 0
    # Get Count of Non Zero Elements
    k_area = np.count_nonzero(k_mask)
    # Calculate the Geometric Threshold
    geo_th = k_area * (gm_th / (k_side ** 2))
    # Padding Borders for Simpler Calculations
    pad_image = np.lib.pad(image, ((3, 3), (3, 3)), 'constant', constant_values = 0)
    # Initialize an Edge Strength Image
    corners = np.zeros([m, n]).reshape(m, n)
    # For Each Pixel in the Image get Edges
    for i in range(3, m+3):
        for j in range(3, n+3):
            # Get USAN Region
            # Calculate the number of pixels within the circular mask of similar brightness to the nucleus
            usan = k_mask * np.exp(-((pad_image[i-3:i+4, j-3:j+4] - pad_image[i, j]) / br_th) ** 6)
            # Calculate USAN Area
            usan_area = np.sum(usan)
            # Subtract USAN from Geometric Threshold to produce Edge Strength Image
            if (usan_area < geo_th):
                # This Pixel is an Edge Pixel
                corners[i-3, j-3] = (geo_th - usan_area)
    # Convert Corners Map to float64
    corners = np.float64(corners)
    # For Each Pixel in Edges get Corners
    for r in range(0, m):
        for c in range(0, n):
            if (corners[r, c] < cd_th):
                corners[r, c] = 0
    corners = np.float64(np.round(corners * (normz / np.amax(corners))))
    # Non-Maximum Suppression - Getting the Higher Value - 11x11 Grid
    corners = np.lib.pad(corners, ((5, 5), (5, 5)), 'constant', constant_values = 0)
    for r in range(5, corners.shape[0]-5):
        for c in range(5, corners.shape[1]-5):
            for i in range(-5, 6):
                for j in range(-5, 6):
                    if corners[r, c] > corners[r+i, c+j]:
                        corners[r+i, c+j] = 0
    corners = corners[5:corners.shape[0]-5, 5:corners.shape[1]-5]
    # 8 Connectivity Suppression
    corners = supr_8conn(corners)
    # Make a List of all Corner Points for Plot
    Cx = list()
    Cy = list()
    for r in range(0, corners.shape[0]):
        for c in range(0, corners.shape[1]):
            if (corners[r, c] > 0):
                Cx.append(c)
                Cy.append(r)
    # Return the Corners Image
    return [corners, Cx, Cy]


# Perform SUSAN and get the Corners
# Parameters for SUSAN Function = Image, Normalization Value, 
# Geometric Threshold, Brightness Threshold, Corner Detect Threshold

Corners_1, Cx_1, Cy_1 = do_Susan([g_img_1, 255.0, 35.76, 10.0, 10.0])
# Best Values - 255.0, 35.76, 10.0, 10.0
plt.figure('Corners Image 1')
plt.imshow(ip_img_1, cmap='gray')
plt.plot(Cx_1, Cy_1, 'r.')

Corners_2, Cx_2, Cy_2 = do_Susan([g_img_2, 1023.0, 15.0, 0.1, 10.25])
# Best Values - 1023.0, 15.0, 0.1, 10.25
plt.figure('Corners Image 2')
plt.imshow(ip_img_2, cmap='gray')
plt.plot(Cx_2, Cy_2, 'r.')

# DO NOT MESS WITH THIS FILTER ORDER !!!
filt_img_2 = filt_Median([g_img_2, 3]) # 3x3 Median Filter
filt_img_2 = filt_Mean([filt_img_2, 5]) # 5x5 Mean Filter
filt_img_2 = filt_Gaussian([filt_img_2, 2.5]) # Gaussian Filter Sigma = 2.5
filt_img_2 = filt_Sharp([filt_img_2, 1.25]) # Sharpening Filter Gain = 1.25
Corners_3, Cx_3, Cy_3 = do_Susan([filt_img_2, 1023.0, 33.88, 7.0, 10.5])
# Best Values - 1023.0, 33.88, 7.0, 10.5
plt.figure('Corners Denoised Image 2')
plt.imshow(filt_img_2, cmap='gray')
plt.plot(Cx_3, Cy_3, 'r.')


end = time.time()
time_taken = (end - start)/60

# End of File