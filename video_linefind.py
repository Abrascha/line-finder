# Import everything needed to edit/save/watch video clips
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from findline import line_finder

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below  print(len(lines))
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    result = line_finder().traffic_line(image)
    return result   

if __name__ == "__main__":
    for x in [i for i in os.listdir("test_videos/") if i[-3::]=='mp4']:
        white_output = 'test_videos_output/' + x
        ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
        ## To do so add .subclip(start_second,end_second) to the end of the line below
        ## Where start_second and end_second are integer values representing the start and end of the subclip
        ## You may also uncomment the following line for a subclip of the first 5 seconds
        ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
        clip1 = VideoFileClip("test_videos/" + x)
        white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)
    os.system("firefox test_videos_output/*.mp4")