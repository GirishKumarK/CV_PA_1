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
col_img = cv.imread('test1.jpg')
#col_img = cv.imread('test2.jpg')
#col_img = cv.imread('test3.jpg')
#col_img = cv.imread('test4.jpg')
#col_img = cv.imread('test5.jpg')
col_img_size = col_img.shape
cv.imshow('Original Image', col_img)
cv.waitKey(1)

# Convert the Three Channel Image to Single Channel
I = np.uint8((0.30 * col_img[:,:,0]) + (0.59 * col_img[:,:,1]) + (0.11 * col_img[:,:,2]))
cv.imshow('Gray Image', I)
cv.waitKey(1)
I_size = I.shape

# Variables
# L = gray_levels, M = width, N = height
L, M, N = 256, I_size[1], I_size[0]

# Histogram
hvals = np.array([i for i in range(0, L)]).reshape(1, -1)
hist = np.zeros([1, L]).reshape(1, -1)
pixels = I_size[0] * I_size[1]
for n in range(0, L):
    hist[:, n] = np.count_nonzero(I == n)
plt.figure('Image Histogram')
plt.step(hvals[0, :], hist[0, :], 'r')
plt.xlabel('Gray Values')
plt.ylabel('No. of Gray Value Occurences')
plt.title('Image Histogram')

# Probability of Occurence of Gray Pixel
P_gray = np.zeros([1, L]).reshape(1, -1)
for n in range(0, L):
    P_gray[:, n] = hist[:, n]/pixels

# Cumulative Density Function
cdf = np.cumsum(hist)

# Find the Entropy Ranges
hist_lo = np.zeros([1, L]).reshape(1, -1)
hist_hi = np.zeros([1, L]).reshape(1, -1)
for t in range(0, L):
    # Low Range Entropy
    cdf_lo = cdf[t]
    if cdf_lo > 0:
        for n in range(1, t):
            if P_gray[:, n] > 0:
                hist_lo[:, t] = (hist_lo[:, t]) - ((P_gray[:, n]/cdf_lo) * np.log(P_gray[:, n]/cdf_lo));
    # High Range Entropy
    cdf_hi = (1 - cdf_lo); # cdf_lo + cdf_hi = 1
    if cdf_hi > 0:
        for n in range((t + 1), L):
            if P_gray[:, n] > 0:
                hist_hi[:, t] = (hist_hi[:, t]) - ((P_gray[:, n]/cdf_hi) * np.log(P_gray[:, n]/cdf_hi));

# Choose the Best Threshold Value based on Entropy Values
entropy = list()
entropy = np.array(entropy).reshape(1, -1)
threshold = 0
hist_max = hist_lo[:, 0] + hist_hi[:, 0]
entropy = np.append(entropy, hist_max)
for n in range(1, L):
    entropy = np.append(entropy, (hist_lo[:, n] + hist_hi[:, n]))
#    if entropy[n] > hist_max:
#        hist_max = entropy[n]
#        threshold = (n - 1)

# A Simple and Faster Method
entropy_max = max(e for e in entropy if e > 0)
entropy_ind = np.where(entropy == entropy_max)
thresh_hi = entropy_ind[0][0]
entropy_min = min(e for e in entropy if e > 0)
entropy_ind = np.where(entropy == entropy_min)
thresh_lo = entropy_ind[0][0]
if thresh_hi < thresh_lo:
    threshold = (L - thresh_lo)
else:
    threshold = int((thresh_lo + thresh_hi)/2)

# Display New Image
new_I = np.zeros(I_size)
new_I[I < threshold] = 0
new_I[I >= threshold] = 255
cv.imshow('Image After Entropy Thresholding', np.uint8(new_I))
cv.waitKey(1)


# Another Thresholding Method Based on Histogram
# Divide Image Histogram to 16 parts and analyse the min, max, mean and stdev in each part.
h_parts = np.array([i for i in range(0, int(np.sqrt(L)))]).reshape(1, -1)
h_min = np.zeros([1, int(np.sqrt(L))]).reshape(1, -1)
h_max = np.zeros([1, int(np.sqrt(L))]).reshape(1, -1)
h_mean = np.zeros([1, int(np.sqrt(L))]).reshape(1, -1)
h_std = np.zeros([1, int(np.sqrt(L))]).reshape(1, -1)
for n in range(0, int(np.sqrt(L))):
    h_min[0, n] = np.amin(hist[0, (n*(int(np.sqrt(L)))):((n+1)*(int(np.sqrt(L))))])
    h_max[0, n] = np.amax(hist[0, (n*(int(np.sqrt(L)))):((n+1)*(int(np.sqrt(L))))])
    h_mean[0, n] = np.mean(hist[0, (n*(int(np.sqrt(L)))):((n+1)*(int(np.sqrt(L))))])
    h_std[0, n] = np.std(hist[0, (n*(int(np.sqrt(L)))):((n+1)*(int(np.sqrt(L))))])
plt.figure('Image Analysis - Based on Histogram')
plt.plot(h_parts[0, :], h_min[0, :], 'g-')
plt.plot(h_parts[0, :], h_max[0, :], 'r-')
plt.plot(h_parts[0, :], h_mean[0, :], 'b-')
plt.plot(h_parts[0, :], h_std[0, :], 'y-')
plt.xlabel('Image Parts')
plt.ylabel('Min, Max, Mean, StDev Values')
plt.title('Image Analysis - Based on Histogram')
plt.legend(['Min', 'Max', 'Mean', 'StDev'])

# Get the Best Part of Image Histogram based on StDev Values
h_std_vals = h_std[h_std < np.mean(h_std)]
h_std_val = np.amax(h_std_vals)
h_std_ind = np.where(h_std == h_std_val)
h_std_ind = h_std_ind[1][0]

# Choose the Best Threshold Value based on Analysis
h_threshold = min(h for h in hist[0, ((h_std_ind+0)*(int(np.sqrt(L)))):((h_std_ind+1)*(int(np.sqrt(L))))] if h > 0)
h_threshold = np.where(hist == h_threshold)
h_threshold = h_threshold[1][0]
if (h_threshold < 31):
    h_threshold = 127 + h_threshold
elif (h_threshold > 223):
    h_threshold = 127 - (h_threshold - 223)

# Display New Image
new_I = np.zeros(I_size)
new_I[I < h_threshold] = 0
new_I[I >= h_threshold] = 255
cv.imshow('Image After Histogram Analytical Thresholding', np.uint8(new_I))
cv.waitKey(1)


# Another Thresholding Method Based on Entropy
# Divide Image Entropy to 16 parts and analyse the min, max, mean and stdev in each part.
e_parts = np.array([i for i in range(0, int(np.sqrt(L)))]).reshape(1, -1)
e_min = np.zeros([1, int(np.sqrt(L))]).reshape(1, -1)
e_max = np.zeros([1, int(np.sqrt(L))]).reshape(1, -1)
e_mean = np.zeros([1, int(np.sqrt(L))]).reshape(1, -1)
e_std = np.zeros([1, int(np.sqrt(L))]).reshape(1, -1)
for n in range(0, int(np.sqrt(L))):
    e_min[0, n] = np.amin(entropy[(n*(int(np.sqrt(L)))):((n+1)*(int(np.sqrt(L))))])
    e_max[0, n] = np.amax(entropy[(n*(int(np.sqrt(L)))):((n+1)*(int(np.sqrt(L))))])
    e_mean[0, n] = np.mean(entropy[(n*(int(np.sqrt(L)))):((n+1)*(int(np.sqrt(L))))])
    e_std[0, n] = np.std(entropy[(n*(int(np.sqrt(L)))):((n+1)*(int(np.sqrt(L))))])
plt.figure('Image Analysis - Based on Entropy')
plt.plot(h_parts[0, :], e_min[0, :], 'g-')
plt.plot(h_parts[0, :], e_max[0, :], 'r-')
plt.plot(h_parts[0, :], e_mean[0, :], 'b-')
plt.plot(h_parts[0, :], e_std[0, :], 'y-')
plt.xlabel('Image Parts')
plt.ylabel('Min, Max, Mean, StDev Values')
plt.title('Image Analysis - Based on Entropy')
plt.legend(['Min', 'Max', 'Mean', 'StDev'])

# Get the Best Part of Image Histogram based on StDev Values
e_std_vals = e_std[e_std < np.mean(e_std)]
e_std_val = np.amax(e_std_vals)
e_std_ind = np.where(e_std == e_std_val)
e_std_ind = e_std_ind[1][0]

# Choose the Best Threshold Value based on Analysis
e_threshold = min(e for e in entropy[((e_std_ind+0)*(int(np.sqrt(L)))):((e_std_ind+1)*(int(np.sqrt(L))))] if e > 0)
e_threshold = np.where(entropy == e_threshold)
e_threshold = e_threshold[0][0]
if (e_threshold < 31):
    e_threshold = 127 + e_threshold
elif (e_threshold > 223):
    e_threshold = 127 - (e_threshold - 223)

# Display New Image
new_I = np.zeros(I_size)
new_I[I < e_threshold] = 0
new_I[I >= e_threshold] = 255
cv.imshow('Image After Entropy Analytical Thresholding', np.uint8(new_I))
cv.waitKey(1)


end = time.time()
time_taken = (end - start)/60

# End of File