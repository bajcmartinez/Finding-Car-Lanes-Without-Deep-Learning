import cv2
import numpy as np

class ImageProcessor:
    """
    This class is responsible for all the image transformations that are required
    """
    def __init__(self, debug=False):
        self._debug = debug
        self._M = None

    def _to_greyscale(self, img):
        """
        Applies color transformation to the image, in preparation for lane finding

        :param img:
        :return: color transformed image
        """
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def _enhance(self, img):
        dst = cv2.GaussianBlur(img, (0, 0), 3)
        out = cv2.addWeighted(img, 1.5, dst, -0.5, 0)
        return out

    def _sobel_gradient_condition(self, img, orient='x', thresh_min=0, thresh_max=255):
        sobel = cv2.Sobel(img, cv2.CV_64F, orient == 'x', orient == 'y')
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        return (scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)

    def _directional_condition(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Calculate the x and y gradients
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Take the absolute value of the x and y gradients
        gradient_direction = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
        gradient_direction = np.absolute(gradient_direction)

        return (gradient_direction >= thresh[0]) & (gradient_direction <= thresh[1])

    def _hls_condition(self, img, channel, thresh=(220, 255)):
        channels = {
            "h": 0,
            "l": 1,
            "s": 2
        }
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        hls = hls[:, :, channels[channel]]

        return (hls > thresh[0]) & (hls <= thresh[1])

    def _color_condition(self, img, thresh=150):
        r_channel = img[:, :, 0]
        g_channel = img[:, :, 1]
        return (r_channel > thresh) & (g_channel > thresh)

    def thresholded_image(self, img):
        grey = self._to_greyscale(img)
        grey = self._enhance(grey)

        # apply gradient threshold on the horizontal gradient
        sx_condition = self._sobel_gradient_condition(grey, 'x', 20, 220)

        # apply gradient direction threshold so that only edges closer to vertical are detected.
        dir_condition = self._directional_condition(grey, thresh=(np.pi/6, np.pi*5/6))

        # combine the gradient and direction thresholds.
        gradient_condition = ((sx_condition == 1) & (dir_condition == 1))

        # and color threshold
        color_condition = self._color_condition(img, thresh=200)

        # now let's take the HSL threshold
        l_hls_condition = self._hls_condition(img, channel='l', thresh=(120, 255))
        s_hls_condition = self._hls_condition(img, channel='s', thresh=(100, 255))

        combined_condition = (l_hls_condition | color_condition) & (s_hls_condition | gradient_condition)
        result = np.zeros_like(color_condition)
        result[combined_condition] = 1

        return result

    def _calc_warp_points(self, img):
        """
        Calculates the points for the wrapping

        :return: Source and Destination pointts
        """
        height, width, color = img.shape

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
        return src, dst

    def transform_perspective(self, img):
        height, width, color = img.shape

        src, dst = self._calc_warp_points(img)

        if self._M is None:
            self._M = cv2.getPerspectiveTransform(src, dst)
            self._M_inv = cv2.getPerspectiveTransform(dst, src)

        return cv2.warpPerspective(img, self._M, (width, height), flags=cv2.INTER_LINEAR)

    def restore_perspective(self, img):
        height, width, color = img.shape

        if self._M_inv is None:
            raise Exception("[Image Processor]: You must first calculate the perspective transformation before "
                            "restoring it.")

        return cv2.warpPerspective(img, self._M_inv, (width, height), flags=cv2.INTER_LINEAR)

    def prepare_image(self, img):
        transformed = self.transform_perspective(img)
        thresholded = self.thresholded_image(transformed)

        return thresholded
