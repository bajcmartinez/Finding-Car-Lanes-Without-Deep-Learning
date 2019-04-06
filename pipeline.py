import cv2
import glob
import matplotlib.pyplot as plt
from lib.camera import Camera
from lib.line_finder import LineFinder
import os
from moviepy.editor import VideoFileClip

# sample_camera controls whether we should recalibrate the camera from the sample pictures or just used the saved
# matrix from previous execution
sample_camera = False
camera = Camera()

if sample_camera:
    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')

    fig, axs = plt.subplots(5, 4, figsize=(16, 11))
    axs = axs.ravel()

    # Go through the images one by one
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        img = camera.sample_image(img)

        axs[i].axis('off')
        axs[i].imshow(img)

    plt.show()

    camera.calibrate()
    camera.save()

    fig, axs = plt.subplots(len(images), 2, figsize=(20, 100))
    fig.subplots_adjust(hspace=.2, wspace=.05)
    axo = axs[:, 0]
    axm = axs[:, 1]
    axo[0].set_title('Original', fontsize=30)
    axm[0].set_title('Undistorted', fontsize=30)

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        axo[i].axis('off')
        axo[i].imshow(img)

        # camera calibration happening here
        dst = camera.undistort(img)

        axm[i].axis('off')
        axm[i].imshow(dst)

    plt.show()
else:
    camera.load()


debug = os.getenv("debug", False)

# Test on images
images = glob.glob('./test_images/*.jpg')

for i, fname in enumerate(images):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    line_finder = LineFinder(camera, debug=debug)
    img = line_finder.process(img)

    oname = fname.replace("/test_images/", "/output_images/")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(oname, img)

# Test on project video
line_finder = LineFinder(camera, debug=debug, is_video=True)
video_output = 'project_video_output.mp4'
video_input = VideoFileClip('project_video.mp4')
processed_video = video_input.fl_image(line_finder.process)
processed_video.write_videofile(video_output, audio=False)

# Test on challenge
# line_finder = LineFinder(camera, debug=debug, is_video=True)
# video_output = 'challenge_video_output.mp4'
# video_input = VideoFileClip('challenge_video.mp4')
# processed_video = video_input.fl_image(line_finder.process).subclip(0, 10)
# processed_video.write_videofile(video_output, audio=False)

# Test on harder challenge
# line_finder = LineFinder(camera, debug=debug, is_video=True)
# video_output = 'harder_challenge_video_output.mp4'
# video_input = VideoFileClip('harder_challenge_video.mp4')
# processed_video = video_input.fl_image(line_finder.process)
# processed_video.write_videofile(video_output, audio=False)