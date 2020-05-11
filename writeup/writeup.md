# **Finding Lane Lines on the Road** 

**Description**
This is the writeup report for the *Project 1* of [Self-Driving Car Engineer Nanodegree Program](https://classroom.udacity.com/nanodegrees/nd013/dashboard/overview).
The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

## Reflection

### 1. Import Package
```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
### 2. Class and Functions
```python
class line_finder:
    def __init__(self)
    def fit_coordinate(self, image, fit_parameters)
    def average_slope_intercept(self, image, lines)
    def draw_lines(self, lines, image, color, weight)
    def traffic_line(self, image_input)

```

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

1. convert the images to grayscale as image_input
2. porcessing the gray-image with GaussianBlur
3. find edges with Canny and select the Region of Interest Mask
4. detect lines with Hough transform
5. find two fited lines with slope and intercept
6. draw the two lines on orignal image


In order to draw a single line on the left and right lanes, I add the function average_slope_intercept(), which search the average of slope and intercept of lines. And I have also compare the result with last image and set the weights to reduce the influence of noise.

### 2. Identify potential shortcomings with your current pipeline

By result, can man easy to find:
**challenge.mp4:** the lines in 5 second of the video have not clearly detected.

One potential shortcoming would be what would happen when ... 
1. too many noise between two lines
2. the car not in the middle of the two lines
3. By change direction(the slope is nearly to 0)
4. the neighbosrs line appear in the interesting area

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to:
1. more exactly edge-detection (preprocessing)
2. dynamic slope-threthold
3. better noise filter "relative line detection"

