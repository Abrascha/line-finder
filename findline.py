import os
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# import improcess
import cv2
import math


def fit_coordinate(image, fit_parameters):
    slope,intercept = fit_parameters
    y1 = image.shape[0]
    y2 = int(y1*3/5)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])




def average_slope_intercept(image, lines):
    fit_right = [] 
    fit_left = [] 
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        slope,intercept = np.polyfit((x1,x2),(y1,y2),1)
        if slope > 0:
            fit_right.append((slope, intercept))
        else:
            fit_left.append((slope, intercept))

    left_line = fit_coordinate(image, np.average(fit_left, axis=0))
    right_line = fit_coordinate(image, np.average(fit_right, axis=0))
    return np.array([left_line, right_line])
    


def takeElem(elem,num):
    return elem[num]

def polarTocartesian(r,theta):
    return [int(r/math.cos(theta)), 0, 0, int(r/math.sin(theta))]  ## x1,y1,x2,y2


def findline(image_input):
    # Read in and grayscale the image
    # image = cv2.convertScaleAbs(image_input,alpha=255)
    image = image_input
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 180

    edges = cv2.Canny(blur_gray, low_threshold, high_threshold,L2gradient= True) 
    mask = np.zeros_like(edges)
    ignore_color = 255
    imshape = image.shape
    # print(imshape)
    y_high = imshape[0]//2
    
    # for y in range(imshape[0],0,-1):
    #     if edges[imshape[1]//2, y] != 0:
    #         y_high = y
    #         break
    y_high += 50
    vertices = np.array([[(0,imshape[0]),(imshape[1]/2 - imshape[1]/64,y_high), (imshape[1]/2 + imshape[1]/64,y_high), (imshape[1],imshape[0])]],dtype=np.int32)
    cv2.fillPoly(mask,vertices,ignore_color)
    masked_edges = cv2.bitwise_and(edges,mask)
    
    # print(y_high)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1
    theta = (np.pi/180)
    threshold = 20
    min_line_length = 10
    max_line_gap = 2
    line_image = np.copy(image)*0 #creating a blank to draw lines on

    # Run Hough on edge detected image
    # lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    # trans_thetas = [[round((y1-y2)/(x1-x2),0) for x1,y1,x2,y2 in line][0] for line in lines]
    # trans_thetas_count = [trans_thetas.count(x) for x in trans_thetas]
    # # print(trans_thetas_count)
    # for i, line in enumerate(lines):
    #     if trans_thetas_count[i] > 1:
    #         for x1,y1,x2,y2 in line:
    #             cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),2)
    
    
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    # ployfit() -- slope,b -- fit_right(+),fit_left(-) -- find points in vertices -- cv2.line
    fit_lines = average_slope_intercept(image,lines)

    for line in fit_lines:
        x1,y1,x2,y2 = line
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)


    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 
    
    # Draw the lines on the edge image
    image_output = cv2.addWeighted(image, 0.8, line_image, 1, 0) 
    return image_output



def main():
    # test_images = os.listdir("copy/")
    test_images = os.listdir("test_images/")
    print(test_images)
    for i,x in enumerate(test_images):
        # image_input = mpimg.imread("test_images/"+x)
        image_input = cv2.imread("test_images/"+x)
        image_output = findline(image_input)
        plt.subplot(3,3,i+1)
        plt.imshow(image_output,aspect='auto')

    plt.show()

if __name__ == "__main__":
    main()
