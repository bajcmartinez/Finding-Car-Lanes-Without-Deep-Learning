import numpy as np
import cv2

class Lane():
    """
    Define a class to receive the characteristics of each line detection
    """

    def __init__(self):
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

    def add_fit(self, fit):
        """
        Adds a fit to the current lane

        :param fit: Second order polynomial that represents the lane
        """
        if fit is not None:
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit - self.best_fit)

            self.detected = True

            # if we detected a good fit then we store in current_fit
            self.current_fit = fit
            self.history_fit.append(fit)
            # keep only last 5 items
            self.history_fit = self.history_fit[-5:]

            # calculate the average
            self.best_fit = np.average(self.history_fit, axis=0)
        else:
            self.detected = False
            self.current_fit = [np.array([False])]
            if len(self.history_fit) > 0:
                # throw out oldest fit
                self.history_fit = self.history_fit[:len(self.history_fit) - 1]

            if len(self.history_fit) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.history_fit, axis=0)
