import numpy as np
import cv2

class Lane():
    """
    Define a class to receive the characteristics of each line detection
    """

    def __init__(self, xm_per_pix, ym_per_pix):
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

    def calculate_curvature(self):
        fit_cr = np.polyfit(self.all_y * self._ym_per_pix, self.all_x * self._xm_per_pix, 2)
        plot_y = np.linspace(0, 720 - 1, 720)
        y_eval = np.max(plot_y)

        curve = ((1 + (2 * fit_cr[0] * y_eval * self._ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

        return curve

    def add_fit(self, fit, points_x, points_y):
        """
        Adds a fit to the current lane

        :param fit: Second order polynomial that represents the lane
        """
        if fit is not None:
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit - self.best_fit)

            self.detected = True

            # update points
            self.all_x = points_x
            self.all_y = points_y
            self.radius_of_curvature = self.calculate_curvature()

            # if we detected a good fit then we store in current_fit
            self.current_fit = fit
            self.history_fit.append(fit)
            # keep only last N items
            self.history_fit = self.history_fit[-self.max_history:]

            # calculate the average
            self.best_fit = np.average(self.history_fit, axis=0, weights=self.history_weights[:len(self.history_fit)])
        else:
            self.detected = False
            self.current_fit = [np.array([False])]
