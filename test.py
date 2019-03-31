import cv2
import glob
import matplotlib.pyplot as plt
from lib.camera import Camera
from lib.line_finder import LineFinder

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


images = glob.glob('./test_images/*.jpg')
line_finder = LineFinder(camera, debug=True)

for i, fname in enumerate(images):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    line_finder.process(img)