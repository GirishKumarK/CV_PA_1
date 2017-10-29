'''
READ ME FIRST :
    PLEASE COMMENT/UNCOMMENT THE RESPECTIVE IMAGES BEFORE RUNNING
    FOR A VALID REASON THAT I DO NOT WISH TO PICTURE BOMB THE SCREEN
'''

import numpy as np
import scipy.ndimage as ndi
import cv2 as cv
import time

start = time.time()

# Open and Load the Image
'''COMMENT/UNCOMMENT THE RESPECTIVE IMAGES BEFORE RUNNING HERE'''
col_img = cv.imread('test1.jpg')
#col_img = cv.imread('test2.jpg')
#col_img = cv.imread('test3.jpg')
#col_img = cv.imread('test4.jpg')
#col_img = cv.imread('test5.jpg')
cv.imshow('Original Image', col_img)

# Convert the Three Channel Image to Single Channel
I = np.uint8((0.30 * col_img[:,:,0]) + (0.59 * col_img[:,:,1]) + (0.11 * col_img[:,:,2]))
cv.imshow('Gray Image', I)
I_size = I.shape

# Gaussian Mask gx and gy to convolve with I
n = 3
sigma = 1.22 # If sigma increases, blur increases
# Optimum Values for n = 3 and sigma = 1.22 - For All Images
h = np.array([i for i in range(-(n-1)/2,((n-1)/2)+1)])
hg = np.exp(-(h**2)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
g_x = np.array(hg).reshape(1, -1)
g_y = np.array(hg).reshape(-1, 1)

# Derivative of Gaussian Mask Gx and Gy to convolve with Ix and Iy
h = np.array(-(h)/(sigma**2)).reshape(1, -1)
dhg = np.multiply(h, g_x)
G_x = np.array(dhg).reshape(1, -1)
G_x = np.fliplr(G_x)
G_y = np.array(dhg).reshape(-1, 1)
G_y = np.flipud(G_y)

# Convolution I A - Right and Top
I_x1 = ndi.convolve(I, g_x)
cv.imshow('Ix1', np.uint8(I_x1))
I_y1 = ndi.convolve(I, g_y)
cv.imshow('Iy1', np.uint8(I_y1))

# Convolution I B - Left and Bottom
I_x2 = ndi.convolve(I, np.fliplr(g_x))
cv.imshow('Ix2', np.uint8(I_x2))
I_y2 = ndi.convolve(I, np.flipud(g_y))
cv.imshow('Iy2', np.uint8(I_y2))

# Convolution II A - Right and Top
I_x1_prime = ndi.convolve(I_x1, G_x)
cv.imshow('I\'x1', np.uint8(I_x1_prime))
I_y1_prime = ndi.convolve(I_y1, G_y)
cv.imshow('I\'y1', np.uint8(I_y1_prime))

# Convolution II B - Left and Bottom
I_x2_prime = ndi.convolve(I_x2, np.fliplr(G_x))
cv.imshow('I\'x2', np.uint8(I_x2_prime))
I_y2_prime = ndi.convolve(I_y2, np.flipud(G_y))
cv.imshow('I\'y2', np.uint8(I_y2_prime))

# Magnitude A - Right and Top
mag1 = np.sqrt(I_x1_prime**2 + I_y1_prime**2)
cv.imshow('Magnitude 1', np.uint8(mag1))
# Magnitude B - Left and Bottom
mag2 = np.sqrt(I_x2_prime**2 + I_y2_prime**2)
cv.imshow('Magnitude 2', np.uint8(mag2))

# Direction
theta1 = np.arctan2(I_y1_prime, I_x1_prime)
theta1 = np.degrees(theta1)
theta2 = np.arctan2(I_y2_prime, I_x2_prime)
theta2 = np.degrees(theta2)

def NonMax_Supr(mag, theta):
    # Non Maximum Suppression
    Supr_Mag = np.zeros(mag.shape)
    for r in xrange(Supr_Mag.shape[0]):
        for c in xrange(Supr_Mag.shape[1]):
            if theta[r, c] < 0:
                theta[r, c] += 360
            if ((c+1) < Supr_Mag.shape[1]) and ((c-1) >= 0) and ((r+1) < Supr_Mag.shape[0]) and ((r-1) >= 0):
                # 4 Directions
                # 0 degrees
                if (theta[r, c] >= 337.5 and theta[r, c] < 22.5) or (theta[r, c] >= 157.5 and theta[r, c] < 202.5):
                    if mag[r, c] >= mag[r, c + 1] and mag[r, c] >= mag[r, c - 1]:
                        Supr_Mag[r, c] = mag[r, c]
                # 45 degrees
                if (theta[r, c] >= 22.5 and theta[r, c] < 67.5) or (theta[r, c] >= 202.5 and theta[r, c] < 247.5):
                    if mag[r, c] >= mag[r - 1, c + 1] and mag[r, c] >= mag[r + 1, c - 1]:
                        Supr_Mag[r, c] = mag[r, c]
                # 90 degrees
                if (theta[r, c] >= 67.5 and theta[r, c] < 112.5) or (theta[r, c] >= 247.5 and theta[r, c] < 292.5):
                    if mag[r, c] >= mag[r - 1, c] and mag[r, c] >= mag[r + 1, c]:
                        Supr_Mag[r, c] = mag[r, c]
                # 135 degrees
                if (theta[r, c] >= 112.5 and theta[r, c] < 157.5) or (theta[r, c] >= 292.5 and theta[r, c] < 337.5):
                    if mag[r, c] >= mag[r - 1, c - 1] and mag[r, c] >= mag[r + 1, c + 1]:
                        Supr_Mag[r, c] = mag[r, c]
    return Supr_Mag
 
# Non Maximum Suppression - Right and Top
Supr_Mag1 = NonMax_Supr(mag1, theta1)
cv.imshow('Non-Maximum Suppression 1', np.uint8(Supr_Mag1))
# Non Maximum Suppression - Left and Bottom
Supr_Mag2 = NonMax_Supr(mag2, theta2)
cv.imshow('Non-Maximum Suppression 2', np.uint8(Supr_Mag2))

def Double_Thresh(Supr_Mag):
    # Double Thresholding
    threshold = np.zeros(Supr_Mag.shape)
    high_Threshold = 0.80 * np.amax(Supr_Mag)
    low_Threshold = 0.15 * np.amax(Supr_Mag)
    strong_Val = 200
    weak_Val = 100
    # strong_Edges = np.uint8(Supr_Mag >= high_Threshold)
    # strong_Edges = 255 * strong_Edges
    # threshold = np.add(threshold, strong_Edges)
    # weak_Edges = np.uint8(Supr_Mag >= low_Threshold)
    # weak_Edges = 127 * weak_Edges
    # threshold = np.add(threshold, weak_Edges)
    strong_Pixels = list()
    for r in xrange(Supr_Mag.shape[0]):
        for c in xrange(Supr_Mag.shape[1]):
            pixel = Supr_Mag[r, c]
            if pixel >= high_Threshold:
                threshold[r, c] = strong_Val
                strong_Pixels.append((r, c))
            if pixel >= low_Threshold:
                threshold[r, c] = weak_Val
    return threshold, strong_Pixels

# Double Thresholding - Right and Top
thresh_img1, strong_pix1 = Double_Thresh(Supr_Mag1)
cv.imshow('Threshold 1', np.uint8(thresh_img1))
# Double Thresholding - Left and Bottom
thresh_img2, strong_pix2 = Double_Thresh(Supr_Mag2)
cv.imshow('Threshold 2', np.uint8(thresh_img2))

def Edge_Track(threshold, strong_pixels):
    # Edge Tracking
    edges = threshold
    strongs = strong_pixels
    edges = edges.astype(bool)
    
    # Create new image matrix of Edges shape
    pix = np.zeros(edges.shape, bool)
    dx = [1, 0, -1,  0, -1, -1, 1,  1]
    dy = [0, 1,  0, -1,  1, -1, 1, -1]
    
    # For Every Pixel in Stong Pixel list, 
    # check if the neighboring pixels are connected and with similar intensity
    for s in strongs:
        if not pix[s]:
            q = [s]
            while len(q) > 0:
                s = q.pop()
                pix[s] = True
                edges[s] = 1
                for k in xrange(len(dx)):
                    for c in range(1, 16):
                        nx = s[0] + c * dx[k]
                        ny = s[1] + c * dy[k]
                        # If the below conditions are all True, append that pixel to list
                        if (nx >= 0 and nx < edges.shape[0] and ny >= 0 and ny < edges.shape[1]) and (edges[nx, ny] >= 0.5) and (not pix[nx, ny]):
                            q.append((nx, ny))
    # Append Tracked Pixel points to the original Edges List
    for i in xrange(edges.shape[0]):
        for j in xrange(edges.shape[1]):
            edges[i, j] = 1.0 if pix[i, j] else 0.0
    edges = np.uint8(edges)
    edges[edges == 1] = 255
    return edges

# Edge Tracking - Right and Top
edges1 = Edge_Track(thresh_img1, strong_pix1)
cv.imshow('Egdes 1', np.uint8(edges1))
# Edge Tracking - Left and Bottom
edges2 = Edge_Track(thresh_img2, strong_pix2)
cv.imshow('Egdes 2', np.uint8(edges2))

def Compare_Edges(edges1, edges2):
    final_edges = np.zeros(edges1.shape)
    # From the Edges obtained from above methods, map both edgemap matrices into one final matrix
    for r in range(0, edges1.shape[0]):
        for c in range(0, edges1.shape[1]):
            if ((edges1[r, c] == 255) or (edges2[r, c] == 255)):
                final_edges[r, c] = 255
            else:
                final_edges[r, c] = 0
    return final_edges

final_edges = Compare_Edges(edges1, edges2)
cv.imshow('Final Egdes', np.uint8(final_edges))


end = time.time()
time_taken = (end - start)/60

# Comparison with Built-In Canny (uses Sobel Edge Filters for Best Edges !!! - Something I am not allowed to use !)
canny = cv.Canny(I,100,200)
cv.imshow('Canny Method', np.uint8(canny))

# End of File