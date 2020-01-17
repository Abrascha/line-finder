import os
# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# import improcess
import cv2
import math


class line_finder:
    def __init__(self):
        ## global variante
        # self.fit_left_avg_old = np.array([-1.0, 700])
        # self.fit_right_avg_old = np.array([1.0, 0])

        self.fit_left_avg_old = np.array([])
        self.fit_right_avg_old = np.array([])

    def fit_coordinate(self, image, fit_parameters):
        slope, intercept = fit_parameters
        y1 = image.shape[0]
        y2 = int(y1*0.6)
        x1 = int((y1-intercept)//slope)
        x2 = int((y2-intercept)//slope)
        return np.array([x1, y1, x2, y2])

    def average_slope_intercept(self, image, lines):
        fit_right = []
        fit_left = []
        fit_left_avg = np.array([])
        fit_right_avg = np.array([])

        # use np.polyfit to find slope and intercept
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
                # slope filter
                slope_threthold = 0.4
                if slope > slope_threthold:
                    fit_right.append((slope, intercept))
                elif slope < -slope_threthold:
                    fit_left.append((slope, intercept))
        if fit_left:
            fit_left_avg = np.average(fit_left, axis=0)
        if fit_right:
            fit_right_avg = np.average(fit_right, axis=0)
        # smooth  the move with fit_xx_avg_old
        """ if there's new and old, return avg(new,old) line
            if there's only new, return new line
            if there's only old, return old line
            if there's nothing, return default line
        """
        if (fit_left_avg.size == 0) and self.fit_left_avg_old.size != 0:
            fit_left_avg = self.fit_left_avg_old.copy()
            left_line = self.fit_coordinate(image, fit_left_avg)
        elif fit_left_avg.size != 0:
            if self.fit_left_avg_old.size != 0:
                fit_left_avg = np.average(
                    (fit_left_avg, self.fit_left_avg_old), axis=0, weights=[2, 8])
            self.fit_left_avg_old = fit_left_avg.copy()
            left_line = self.fit_coordinate(image, fit_left_avg)
        else:
            left_line = self.fit_coordinate(image, np.array([-1.0, 700]))

        if fit_right_avg.size == 0 and self.fit_right_avg_old.size != 0:
            fit_right_avg = self.fit_right_avg_old.copy()
            right_line = self.fit_coordinate(image, fit_right_avg)
        elif fit_right_avg.size != 0:
            if self.fit_right_avg_old.size != 0:
                fit_right_avg = np.average(
                    (fit_right_avg, self.fit_right_avg_old), axis=0, weights=[2, 8])
            self.fit_right_avg_old = fit_right_avg.copy()
            right_line = self.fit_coordinate(image, fit_right_avg)
        else:
            right_line = self.fit_coordinate(image, np.array([1.0, 0]))

        return np.array([left_line, right_line])

    def draw_lines(self, lines, image, color, weight):
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(image, (x1, y1), (x2, y2), color, weight)
        return image

    def traffic_line(self, image_input):
        # Read in and grayscale the image
        image = image_input
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Define a kernel size and apply Gaussian smoothing
        kernel_size = 3
        blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # Define our parameters for Canny and apply
        low_threshold = 50  # 50
        high_threshold = 180  # 180
        edges = cv2.Canny(blur_gray, low_threshold,
                          high_threshold, L2gradient=True)

        mask = np.zeros_like(edges)
        ignore_color = 255
        imshape = image.shape
        y_high = int(imshape[0]*0.6) + 10
        vertices = np.array([[(0+int(imshape[1]*0.05), int(imshape[0])),
                              (imshape[1]/2 - imshape[1]/16, y_high),
                              (imshape[1]/2 + imshape[1]/16, y_high),
                              (imshape[1]-int(imshape[1]*0.05), int(imshape[0]))]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_color)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 1
        theta = (np.pi/180)
        threshold = 20
        min_line_length = 5
        max_line_gap = 3
        line_image = np.copy(image)*0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array(
            []), min_line_length, max_line_gap)

        # ployfit() -- slope,b -- fit_right(+),fit_left(-) -- find points in vertices -- cv2.line
        fit_lines = self.average_slope_intercept(image, lines)

        line_image = self.draw_lines(fit_lines, line_image, (255,0,0), 10) # traffic line
        # line_image = self.draw_lines([x[0] for x in lines], line_image,(0,255,0), 3) # Houghline
        # line_image = self.draw_lines((vertices.reshape(2,4)[1],vertices.reshape(2,4)[0]),line_image,(255,255,0), 3) # vertices

        # Create a "color" binary image to combine with line image
        color_edges = np.dstack((edges, edges, edges))

        # Draw the lines on the edge image
        color_image = True
        if color_image:
            image_output = cv2.addWeighted(
                image, 0.8, line_image, 1, 0)  # color image
        else:
            image_output = cv2.addWeighted(
                color_edges, 0.8, line_image, 1, 0)  # no color image
        return image_output


def main():
    # test_images = os.listdir("copy/")
    test_images = os.listdir("test_images/")
    print(test_images)
    for i, x in enumerate(test_images):
        # image_input = mpimg.imread("test_images/"+x)
        image_input = cv2.imread("test_images/"+x)
        image_processing = line_finder()
        image_output = image_processing.traffic_line(image_input)
        plt.subplot(3, 3, i+1)
        plt.imshow(image_output, aspect='auto')

    # image_input = cv2.imread("test_images/"+"challenge.png")
    # image_processing = line_finder()
    # image_output = image_processing.traffic_line(image_input)
    # plt.imshow(image_output, aspect='auto')

    plt.show()


if __name__ == "__main__":
    main()
