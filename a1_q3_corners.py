import numpy as np
from numpy import linalg as la
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import cv2 as cv
import time

start = time.time()

def load_Img(filename):
    # Open and Load the Images
    ip_img = cv.imread(filename)
    cv.imshow('Original Image ' + filename, ip_img)
    return [filename, ip_img]

def get_GrayImg(params):
    filename, image = params[0], params[1]
    # Convert the Three Channel Image to Single Channel
    g_img = np.uint8((0.30 * image[:,:,0]) + (0.59 * image[:,:,1]) + (0.11 * image[:,:,2]))
    cv.imshow('Gray Image ' + filename, image)
    return [filename, g_img]


def get_Hessian_Corners(params):
    # Get the Image Parameters
    filename, image = params[0], params[1]
    
    # Box Blur Kernel
    n = params[2]
    box = list()
    for i in range(n**2):
        box.append(np.divide(1, n**2, dtype='float'))
    box = np.array(box).reshape(n, n)
    filt_image = ndi.convolve(image, box)
    #cv.imshow('Gray Image Box Filter Smoothed ' + filename, image)
    
    # Image Gradients
    self_x = np.array([-1, 0, 1]).reshape(1, -1)
    self_y = np.array([-1, 0, 1]).reshape(-1, 1)
    Ix = ndi.convolve(filt_image, self_x)
    #cv.imshow('Ix ' + filename, Ix)
    Iy = ndi.convolve(filt_image, self_y)
    #cv.imshow('Iy ' + filename, Iy)
    Ixx = Ix * Ix
    #cv.imshow('Ixx ' + filename, Ixx)
    Iyy = Iy * Iy
    #cv.imshow('Iyy ' + filename, Iyy)
    Ixy = Ix * Iy
    #cv.imshow('Ixy ' + filename, Ixy)
    Iyx = Iy * Ix
    #cv.imshow('Iyx ' + filename, Iyx)
    
    Ixx = np.float64(Ixx)
    Iyy = np.float64(Iyy)
    Ixy = np.float64(Ixy)
    Iyx = np.float64(Iyx)
    
    # Duplicate Image
    corners = np.zeros(image.shape)
    
    # Cornerness - Hessian
    thresh = params[3] # Increase Decreases
    
    # For Every Pixel in Image
    lambda_1 = list()
    lambda_2 = list()
    t, tt = 0.0, 0.0
    for r in range(0, image.shape[0]):
        for c in range(0, image.shape[1]):
            # Get Hessian Matrix of that Pixel
            Hessian = np.array([Ixx[r, c], Ixy[r, c], Iyx[r, c], Iyy[r, c]]).reshape(2, 2)
            
            # Get Eigen Values and Vectors of the Hessian Matrix
            t = time.time()
            eig_val, _ = la.eigh(Hessian) # eig_vec ignored using "_"
            #eig_val = np.roots([1, -(Hessian[0, 0] + Hessian[1, 1]), ((Hessian[0, 0] * Hessian[1, 1]) - (Hessian[0, 1] * Hessian[1, 0]))]) # Takes More Time
            tt = tt + (time.time() - t)
            
            # Record the Eigen Values into a List and get Corner Points
            lambda_1.append(eig_val[0])
            lambda_2.append(eig_val[1])
            if (eig_val[0] > eig_val[1]): # Edge Case
                corners[r, c] = 0
            if (eig_val[0] < eig_val[1]): # Edge Case
                corners[r, c] = 0
            if ((abs(eig_val[0] - eig_val[1])) < thresh): # Flat Case
                corners[r, c] = 0
            if ((abs(eig_val[0] - eig_val[1])) >= thresh): # Corner Case
                corners[r, c] = 255
    
    print 'Time Taken by Hessian Corner Detection Method for ' + filename + ' (Secs) : ' + str(tt)
    lambda_1 = np.array(lambda_1).reshape(image.shape)
    lambda_2 = np.array(lambda_2).reshape(image.shape)
    cv.imshow('Hessian Corners ' + filename, corners)
    
    # Eigen Values Statistics - For Analysis - Not Used
    lambda_1_max  = np.amax(lambda_1)
    lambda_1_min  = np.amin(lambda_1)
    lambda_1_mean  = np.mean(lambda_1)
    lambda_2_max  = np.amax(lambda_2)
    lambda_2_min  = np.amin(lambda_2)
    lambda_2_mean  = np.mean(lambda_2)
    
    # Plot Corners over Image
    X = list()
    Y = list()
    for r in range(0, image.shape[0]):
        for c in range(0, image.shape[1]):
            if (corners[r, c] == 255):
                # Append the Corner Pixel Values into X, Y Lists
                X.append(c)
                Y.append(r)
    plt.figure('Hessian Corners ' + filename)
    plt.imshow(image, cmap='gray')
    plt.plot(X, Y, 'r.')
    
    # Return Nothing
    return None


def get_Harris_Corners(params):
    # Get the Image Parameters
    filename, image = params[0], params[1]
    
    # Gaussian Mask
    n = params[2] # Kernel Size
    sigma = params[3] # If sigma increases, blur increases
    # Notes : Sigma Matters! Don't take whole numbers!
    # Obtain the Gaussian Kernel using the Formula
    h = np.array([i for i in range(-(n-1)/2,((n-1)/2)+1)])
    hg = np.exp(-(h**2)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
    g = np.array(hg).reshape(1, -1)
    
    # Derivative of Gaussian Mask
    h = np.array(-(h)/(sigma**2)).reshape(1, -1)
    dhg = np.multiply(h, g)
    G_x = np.array(dhg).reshape(1, -1)
    G_x = np.fliplr(G_x)
    G_y = np.array(dhg).reshape(-1, 1)
    G_y = np.flipud(G_y)
    
    # Image Gradients
    Lx = ndi.convolve(image, G_x)
    #cv.imshow('Lx ' + filename, Lx)
    Ly = ndi.convolve(image, G_y)
    #cv.imshow('Ly ' + filename, Ly)
    Lxx = Lx * Lx
    #cv.imshow('Lxx ' + filename, Lxx)
    Lyy = Ly * Ly
    #cv.imshow('Lyy ' + filename, Lyy)
    Lxy = Lx * Ly
    #cv.imshow('Lxy ' + filename, Lxy)
    Lyx = Ly * Lx
    #cv.imshow('Lyx ' + filename, Lyx)
    
    Lxx = np.float64(Lxx)
    Lyy = np.float64(Lyy)
    Lxy = np.float64(Lxy)
    Lyx = np.float64(Lyx)
    
    # Duplicate Image
    H_corners_1 = np.zeros(image.shape)
    H_corners_2 = np.zeros(image.shape)
    
    # Cornerness - Harris-Stephen
    H_thresh = params[4]
    H_thresh_1 = params[5] # More Negative Less Points
    H_thresh_2 = params[6] # More Negative Less Points
    alpha = np.float64(params[7]) # Increase Increases Points
    # alpha 0 to 0.25
    
    # Harris Method 1 - Det and Trace
    # For Every Pixel in the Image
    H_lambda_1 = list()
    H_lambda_2 = list()
    H_C_1 = list()
    H_C_2 = list()
    t1, tt1 = 0.0, 0.0
    t2, tt2 = 0.0, 0.0
    for r in range(0, image.shape[0]):
        for c in range(0, image.shape[1]):
            # Get the Harris Matrix
            Harris = np.array([Lxx[r, c], Lxy[r, c], Lyx[r, c], Lyy[r, c]]).reshape(2, 2)
            
            # Cornerness using First Method of Harris - Determinant and Trace
            # This line below Calculates Cornerness directly without using NumPy Methods - Time Taken is EXTREMELY LESS
            #H_cornerness_1 = (np.float64((Harris[0, 0] * Harris[1, 1]) - (Harris[0, 1] * Harris[1, 0])) - (alpha * np.float64((Harris[0, 0] + Harris[1, 1]) ** 2)))
            # This line below Calculates Cornerness using NumPy Methods - Time Taken is MORE than the Above Method
            # Using below method for Consistency and Time Comparison since NumPy method is used for Eigen Values Calculation for Harris Second Method
            t1 = time.time()
            H_cornerness_1 = (la.det(Harris) - (alpha * (np.trace(Harris) ** 2)))
            tt1 = tt1 + (time.time() - t1)
            H_C_1.append(H_cornerness_1)
            
            # Calculate Eigen Values and Vectors of Harris Matrix
            # Cornerness using Second Method of Harris - Eigen Values
            t2 = time.time()
            H_eig_val, _ = la.eigh(Harris) # H_eig_vec ignored using "_"
            #H_eig_val = np.roots([1, -(Harris[0, 0] + Harris[1, 1]), ((Harris[0, 0] * Harris[1, 1]) - (Harris[0, 1] * Harris[1, 0]))]) # Takes More Time
            H_cornerness_2 = ((H_eig_val[0] * H_eig_val[1]) - (alpha * ((H_eig_val[0] + H_eig_val[1]) ** 2)))
            tt2 = tt2 + (time.time() - t2)
            H_C_2.append(H_cornerness_2)
            
            # Record the Eigen Values into a List and get Corner Points
            H_lambda_1.append(H_eig_val[0])
            H_lambda_2.append(H_eig_val[1])
            if (H_eig_val[0] > H_eig_val[1]): # Edge Case
                H_corners_1[r, c] = 0
                H_corners_2[r, c] = 0
            if (H_eig_val[0] < H_eig_val[1]): # Edge Case
                H_corners_1[r, c] = 0
                H_corners_2[r, c] = 0
            if ((abs(H_eig_val[0] - H_eig_val[1])) < H_thresh): # Flat Case
                H_corners_1[r, c] = 0
                H_corners_2[r, c] = 0
            if ((abs(H_eig_val[0] - H_eig_val[1])) >= H_thresh): # Corner Case
                if (H_cornerness_1 < H_thresh_1):
                    H_corners_1[r, c] = 255
                if (H_cornerness_1 >= H_thresh_1):
                    H_corners_1[r, c] = 0
                if (H_cornerness_2 < H_thresh_2):
                    H_corners_2[r, c] = 255
                if (H_cornerness_2 >= H_thresh_2):
                    H_corners_2[r, c] = 0
    
    print 'Time Taken by Harris Corner Detection Method 1 for ' + filename + ' (Secs) : ' + str(tt1)
    print 'Time Taken by Harris Corner Detection Method 2 for ' + filename + ' (Secs) : ' + str(tt2)
    cv.imshow('Harris Corners Method 1 ' + filename, H_corners_1)
    cv.imshow('Harris Corners Method 2 ' + filename, H_corners_2)
    
    # Eigen Value Statistics - - For Analysis - Not Used
    H_lambda_1 = np.array(H_lambda_1).reshape(image.shape)
    H_lambda_2 = np.array(H_lambda_2).reshape(image.shape)
    H_lambda_1_max  = np.amax(H_lambda_1)
    H_lambda_1_min  = np.amin(H_lambda_1)
    H_lambda_1_mean  = np.mean(H_lambda_1)
    H_lambda_2_max  = np.amax(H_lambda_2)
    H_lambda_2_min  = np.amin(H_lambda_2)
    H_lambda_2_mean  = np.mean(H_lambda_2)
    H_C_1 = np.array(H_C_1).reshape(image.shape)
    H_C_2 = np.array(H_C_2).reshape(image.shape)
    H_C_1_max  = np.amax(H_C_1)
    H_C_1_min  = np.amin(H_C_1)
    H_C_1_mean  = np.mean(H_C_1)
    H_C_2_max  = np.amax(H_C_2)
    H_C_2_min  = np.amin(H_C_2)
    H_C_2_mean  = np.mean(H_C_2)
    
    # Plot Corners
    # Harris Method 1
    HX1 = list()
    HY1 = list()
    for r in range(0, image.shape[0]):
        for c in range(0, image.shape[1]):
            # Append the Corner Pixel Values into X, Y Lists
            if (H_corners_1[r, c] == 255):
                HX1.append(c)
                HY1.append(r)
    plt.figure('Harris Corners Method 1 ' + filename)
    plt.imshow(image, cmap='gray')
    plt.plot(HX1, HY1, 'r.')
    # Harris Method 2
    HX2 = list()
    HY2 = list()
    for r in range(0, image.shape[0]):
        for c in range(0, image.shape[1]):
            # Append the Corner Pixel Values into X, Y Lists
            if (H_corners_2[r, c] == 255):
                HX2.append(c)
                HY2.append(r)
    plt.figure('Harris Corners Method 2 ' + filename)
    plt.imshow(image, cmap='gray')
    plt.plot(HX2, HY2, 'r.')
    
    # Return Nothing
    return None


# Load Image Filenames as a List
images = ['input1.png', 'input2.png', 'input3.png']

# Hessian Thresholds List for the Images
hessian_th = [[3, 515], [3, 510], [5, 530]]
# Hessian Default - [[3, 515],[3, 510], [3, 530]]

# Harris Thresholds List for the Images
harris_th = [[5, 4.41, 510, 0, -65024, 0.04], 
             [11, 6.25, 510, 0, -65024, 0.04], 
             [5, 4.25, 510, 0, -65024, 0.04]]
# Harris Default - [[5, 4.41, 510, 0, -65024, 0.04], [11, 6.25, 510, 0, -65024, 0.04], [5, 4.25, 510, 0, -65024, 0.04]]

# Execute the Functions for Corner Detection
for i in range(0, 3):
    filename, image = load_Img(images[i])
    filename, gray_image = get_GrayImg([filename, image])
    get_Hessian_Corners([filename, gray_image, 
                         hessian_th[i][0], hessian_th[i][1]])
    get_Harris_Corners([filename, gray_image, harris_th[i][0], 
                        harris_th[i][1], harris_th[i][2], 
                        harris_th[i][3], harris_th[i][4], harris_th[i][5]])


end = time.time()
time_taken = (end - start)/60

# End of File