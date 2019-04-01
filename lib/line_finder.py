import numpy as np
import cv2
from lib.image_processor import ImageProcessor
import matplotlib.pyplot as plt
from lib.lane import Lane


class LineFinder:
    """
    This class is responsible for finding the lanes on a particular image
    """

    def __init__(self, camera, debug=False, is_video=False):
        """

        :param camera: Camera
        """
        self._camera = camera
        self._debug = debug
        self._is_video = is_video
        self._image_processor = ImageProcessor(self._debug)

        self._xm_per_pix = 3.048 / 100
        self._ym_per_pix = 3.7 / 378

        self.left_lane = Lane(xm_per_pix=self._xm_per_pix, ym_per_pix=self._ym_per_pix)
        self.right_lane = Lane(xm_per_pix=self._xm_per_pix, ym_per_pix=self._ym_per_pix)

    def _find_lane_pixels(self, binary_warped):
        """
        Finds the lane pixels in a binary warped image

        :param binary_warped:
        :return: (left_x, left_y, right_x, right_y, out_image: only on debug mode)
        """
        if self._debug:
            # Create an output image to draw on and visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255.
        else:
            out_img = None

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        if not self.left_lane.detected or not self.right_lane.detected:
            # We are starting our search from scratch, with no previous reference

            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0] // 2)
            left_x_base = np.argmax(histogram[:midpoint])
            right_x_base = np.argmax(histogram[midpoint:]) + midpoint

            # Choose the number of sliding windows
            num_windows = 9
            # Set the width of the windows +/- margin
            margin = 50
            # Set minimum number of pixels found to recenter window
            min_pix = 100

            # Set height of windows - based on num_windows above and image shape
            window_height = np.int(binary_warped.shape[0] // num_windows)

            # Current positions to be updated later for each window in nwindows
            left_x_current = left_x_base
            right_x_current = right_x_base

            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(num_windows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_x_left_low = left_x_current - margin
                win_x_left_high = left_x_current + margin
                win_x_right_low = right_x_current - margin
                win_x_right_high = right_x_current + margin

                if self._debug:
                    # Draw the windows on the visualization image
                    cv2.rectangle(out_img, (win_x_left_low, win_y_low),
                                  (win_x_left_high, win_y_high), (0, 255, 0), 2)
                    cv2.rectangle(out_img, (win_x_right_low, win_y_low),
                                  (win_x_right_high, win_y_high), (0, 255, 0), 2)

                # Identify the nonzero pixels in x and y within the window #
                good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                                  (nonzero_x >= win_x_left_low) & (nonzero_x < win_x_left_high)).nonzero()[0]
                good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                                   (nonzero_x >= win_x_right_low) & (nonzero_x < win_x_right_high)).nonzero()[0]

                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)

                # If you found > min_pix pixels, recenter next window on their mean position
                if len(good_left_inds) > min_pix:
                    left_x_current = np.int(np.mean(nonzero_x[good_left_inds]))
                if len(good_right_inds) > min_pix:
                    right_x_current = np.int(np.mean(nonzero_x[good_right_inds]))

            # Concatenate the arrays of indices (previously was a list of lists of pixels)
            try:
                left_lane_inds = np.concatenate(left_lane_inds)
                right_lane_inds = np.concatenate(right_lane_inds)
            except ValueError:
                # Avoids an error if the above is not implemented fully
                pass

        else:
            # Set the width of the windows +/- margin
            margin = 50

            # In here we now search for the pixels based on previous lane information
            left_fit_prev = self.left_lane.current_fit
            right_fit_prev = self.right_lane.current_fit

            left_lane_inds = ((nonzero_x > (left_fit_prev[0] * (nonzero_y ** 2) +
                                            left_fit_prev[1] * nonzero_y +
                                            left_fit_prev[2] - margin)) &
                              (nonzero_x < (left_fit_prev[0] * (nonzero_y ** 2) +
                                            left_fit_prev[1] * nonzero_y +
                                            left_fit_prev[2] + margin)))

            right_lane_inds = ((nonzero_x > (right_fit_prev[0] * (nonzero_y ** 2) +
                                             right_fit_prev[1] * nonzero_y +
                                             right_fit_prev[2] - margin)) &
                               (nonzero_x < (right_fit_prev[0] * (nonzero_y ** 2) +
                                             right_fit_prev[1] * nonzero_y +
                                             right_fit_prev[2] + margin)))

        # Extract left and right line pixel positions
        left_x = nonzero_x[left_lane_inds]
        left_y = nonzero_y[left_lane_inds]
        right_x = nonzero_x[right_lane_inds]
        right_y = nonzero_y[right_lane_inds]

        return left_x, left_y, right_x, right_y, out_img

    def _find_lanes(self, binary_warped):
        """
        Uses polyfit to find the the lanes

        :return: (, processed image only on image mode)
        """

        # Find our lane pixels first
        left_x, left_y, right_x, right_y, out_img = self._find_lane_pixels(binary_warped)

        # Fit a second order polynomial to each using `np.polyfit`
        if len(left_y) > 0 and len(left_x) > 0:
            left_fit = np.polyfit(left_y, left_x, 2)
        else:
            left_fit = None

        if len(right_y) > 0 and len(right_x) > 0:
            right_fit = np.polyfit(right_y, right_x, 2)
        else:
            right_fit = None

        if self._debug:
            # Visualization
            # Colors in the left and right lane regions
            out_img[left_y, left_x] = [255, 0, 0]
            out_img[right_y, right_x] = [0, 0, 255]

        return left_fit, right_fit, (left_x, left_y, right_x, right_y), out_img

    def draw_lanes(self, img):
        """
        Renders the lanes on top of the given images

        :param img:
        :return: updated image
        """
        if self.left_lane.best_fit is None or self.right_lane.best_fit is None:
            return img

        height, width, color = img.shape

        # Create an image to draw the lines on
        lanes_img = np.zeros_like(img).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        # for that we take the lanes best fits
        plot_y = np.linspace(0, height - 1, num=height)  # to cover same y-range as image
        l_fit = self.left_lane.best_fit
        r_fit = self.right_lane.best_fit

        # Build the 2nd order polynomial
        left_fit_x = l_fit[0] * plot_y**2 + l_fit[1]*plot_y + l_fit[2]
        right_fit_x = r_fit[0] * plot_y**2 + r_fit[1] * plot_y + r_fit[2]

        # Calculate all the points
        pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw them into the blank image
        cv2.fillPoly(lanes_img, np.int_([pts]), (0, 255, 0))

        return lanes_img

    def draw_info(self, img):
        info_img = np.copy(img)

        font = cv2.FONT_HERSHEY_COMPLEX

        text = 'Left curvature: ' + '{:04.2f}'.format(self.left_lane.radius_of_curvature) + 'm'
        cv2.putText(info_img, text, (40, 70), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)

        text = 'Right curvature: ' + '{:04.2f}'.format(self.right_lane.radius_of_curvature) + 'm'
        cv2.putText(info_img, text, (40, 120), font, 1.5, (200, 255, 155), 2, cv2.LINE_AA)

        return info_img


    def process(self, img):
        """
        Finds lanes on an image

        :param img:
        :return: boolean
        """
        undistorted = self._camera.undistort(img)
        prepared = self._image_processor.prepare_image(undistorted)

        left_fit, right_fit, points, img_polyfit = self._find_lanes(prepared)

        self.left_lane.add_fit(left_fit, points[0], points[1])
        self.right_lane.add_fit(right_fit, points[2], points[3])

        # Now with all done we need to change our original image to draw the lanes we just matched
        lanes_warped = self.draw_lanes(img)
        lanes = self._image_processor.restore_perspective(lanes_warped)
        result = cv2.addWeighted(img, 1, lanes, 0.3, 0)

        # now draw lane information into it
        result = self.draw_info(result)

        if self._debug:
            if self._is_video:
                img_debug = np.zeros(result.shape, dtype=np.uint8)

                # result (top left)
                img_debug[0:360, 0:640, :] = cv2.resize(result, (640, 360))

                # prepared (top right)
                thresholded = self._image_processor.thresholded_image(undistorted)
                thresholded = np.dstack((thresholded*255., thresholded*255., thresholded*255.))
                img_debug[0:360, 640:1280, :] = cv2.resize(thresholded, (640, 360))

                # polyfit (bottom right)
                img_debug[360:720, 640:1280, :] = cv2.resize(img_polyfit, (640, 360))

                result = img_debug


            else:
                fig, axs = plt.subplots(1, 3, figsize=(16, 10))
                fig.subplots_adjust(hspace=.2, wspace=.05)
                axs[0].set_title('Original', fontsize=30)
                axs[1].set_title('Pre-Processed', fontsize=30)
                axs[2].set_title('Final', fontsize=30)

                axs[0].imshow(img)
                axs[1].imshow(prepared, cmap='gray')
                axs[2].imshow(result)

                plt.show()

        return result
