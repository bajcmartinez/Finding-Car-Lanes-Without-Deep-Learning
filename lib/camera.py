import numpy as np
import cv2
import pickle


class Camera:
    """
    Class responsible for all camera manipulations to an image
    """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    _objp = np.zeros((6 * 9, 3), np.float32)
    _objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    _valid_images = []  # images that contain a 9 by 6 grid
    _obj_points = []  # 3d points in real world space
    _img_points = []  # 2d points in image plane.

    _mtx = [] # camera calibration mtx
    _dist = [] # camera calibration dist

    def sample_image(self, img):
        """
        Load an image to be used as a sample for the camera calibration

        :param img:
        :return: image with corners
        """

        # first we convert the  image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            self._valid_images.append(img)
            self._obj_points.append(self._objp)
            self._img_points.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)

        return img


    def calibrate(self):
        """
        Calibrate the camera

        :return: None
        """
        img_size = (self._valid_images[0].shape[1], self._valid_images[0].shape[0])
        ret, self._mtx, self._dist, t, t2 = cv2.calibrateCamera(self._obj_points, self._img_points, img_size, None, None)

    def save(self, filename="camera.p"):
        """
        Saves the camera settings to disk
        """
        dist_pickle = {}
        dist_pickle["mtx"] = self._mtx
        dist_pickle["dist"] = self._dist
        pickle.dump(dist_pickle, open(filename, "wb"))

    def load(self, filename="camera.p"):
        """
        Loads the camera settings from disk

        :param filename:
        """
        dist_pickle = pickle.load(open(filename, "rb"))
        self._mtx = dist_pickle["mtx"]
        self._dist = dist_pickle["dist"]

    def undistort(self, img):
        """
        Once the camera is calibrated, use this method to undistort images

        :param img:
        :return: undistorted image
        """
        return cv2.undistort(img, self._mtx, self._dist, None, self._mtx)
