# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[p_image_processor_1]: ./examples/p_image_processor_1.jpg "Image Processor 1"
[p_image_processor_2]: ./examples/p_image_processor_2.jpg "Image Processor 2"
[r_image_1]: ./output_images/straight_lines1.jpg "Result"
[r_image_2]: ./output_images/straight_lines2.jpg "Result"
[r_image_3]: ./output_images/test1.jpg "Result"
[r_image_4]: ./output_images/test2.jpg "Result"
[r_image_5]: ./output_images/test3.jpg "Result"
[r_image_6]: ./output_images/test4.jpg "Result"
[r_image_7]: ./output_images/test5.jpg "Result"
[r_image_8]: ./output_images/test6.jpg "Result"

---

## Set up

1. Clone from [Github Repository](https://github.com/bajcmartinez/CarND-Advanced-Lane-Lines)
```bash
git clone git@github.com:bajcmartinez/CarND-Advanced-Lane-Lines.git
```  

2. Installing requirements
```bash
pipenv install
```

3. Run the code
```bash
pipenv run python pipeline.py
```

Run in debug mode

```bash
debug=True; pipenv run python pipeline.py
```

## Code Structure

The file `pipeline.py` is an example that executes the pipeline for individual images as well as videos.

All the code executed is separated by individual modules which we will explain next:

### camera.py

This module is responsible for the camera calibration.

How it works?

1. We need to use the method `camera.sample_image(img)` to load all the images that will work as a basis for the camera calibration.
2. We call `camera.calibrate()` to calibrate the camera.

Optional, as this process can take some time we provided some methods to avoid doing this step on each execution.

1. After `camera.calibrate()` method, we can call `camera.save()`. This will save to disk the distortion matrix for later usage.
2. If we want to load the saved camera details call `camera.load()`.

3. Either after `camera.calibrate()` or `camera.load()` call `camera.undistort(img)` to undistort the given image. 

### image_processor.py

This module is responsible for preparing the images for lane finding. The main function to be used is `image_processor.prepare_image(img)` which will execute the image pipeline as follows:

1. Perspective transformation:

    The points where selected from a sample image, using lane points and extrapolating them to be a rect all the way to the top of the image as follows: 

    ```python
    src = np.float32([
        [210, height],
        [1110, height],
        [580, 460],
        [700, 460]
    ])
    
    dst = np.float32([
        [210, height],
        [1110, height],
        [210, 0],
        [1110, 0]
    ])
    ```
    
2. Threshold the image:
    
    This processes is composed of many steps described as follow:
    
    1. Convert to greyscale.
    2. Enhance image using Gaussian Blur.
    3. Threshold on the horizontal gradient using Sobel.
    4. Gradient direction threshold so that only edges closer to vertical are detected, using Sobel.
    5. Color threshold
    6. HSL threshold on L layer
    7. HSL threshold on S layer

    Example 1                        |  Example 2
    :-------------------------------:|:-------------------------------:
    ![alt text][p_image_processor_1] | ![alt text][p_image_processor_2]


### lane.py

This module describes all the properties and method of a lane.

Properties

```python
# was the line detected in the last iteration?
self.detected = False
# x values of the last n fits of the line
self.recent_x_fitted = []
# average x values of the fitted line over the last n iterations
self.best_x = None
# polynomial coefficients averaged over the last n iterations
self.best_fit = None
# polynomial coefficients for the most recent fit
self.current_fit = [np.array([False])]
# polynomial coefficients for the recent fits
self.history_fit = []
# max count for elements in the history, 1 second approx
self.max_history = 30
# weights used to calculate the history average
self.history_weights = [x//2+1 for x in range(self.max_history)]
# radius of curvature of the line in some units
self.radius_of_curvature = None
# distance in meters of vehicle center from the line
self.line_base_pos = None
# difference in fit coefficients between last and new fits
self.diffs = np.array([0, 0, 0], dtype='float')

# x values for detected line pixels
self.all_x = None
# y values for detected line pixels
self.all_y = None

# meters per pixel in dimension
self._xm_per_pix = xm_per_pix
self._ym_per_pix = ym_per_pix
```

### line_finder.py

This module is the principal responsible for finding lanes in the pictures and has all the logic for it.

The main method is `line_finder.process(img)` which expects an image and returns the processed image

Here are all the steps that are required:

1. Undistort the given image using the camera module
2. Prepare the image for analysis using the image_processor module
3. Find lanes
    3.1 Use the histogram method is no lanes where found on the previous frame
    3.2 If lanes were found on previous frame, just look for lines on the nearby
4. Update the lane information on the lane module
    4.1 Calculate curvature
    ```python
    fit_cr = np.polyfit(self.all_y * self._ym_per_pix, self.all_x * self._xm_per_pix, 2)
    plot_y = np.linspace(0, 720 - 1, 720)
    y_eval = np.max(plot_y)

    curve = ((1 + (2 * fit_cr[0] * y_eval * self._ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
    ```
    4.2 Run sanity check
    ```python
    R0 = self.radius_of_curvature
    self._insanity = abs(R - R0) / R0
    return self._insanity <= 0.5
    ```
    
5. Restore perspective on the lanes so it can be over imposed the original image
6. Draw the curvature and distance from the center of the lane into the image

Here are some examples

![alt text][r_image_1]           |  ![alt text][r_image_2]
:-------------------------------:|:-------------------------------:
![alt text][r_image_3] | ![alt text][r_image_4]
![alt text][r_image_5] | ![alt text][r_image_6]
![alt text][r_image_7] | ![alt text][r_image_8]