from lib.image_processor import ImageProcessor

class LineFinder:
    """
    This class is responsible for finding the lanes on a particular image
    """
    def __init__(self, camera, debug = False):
        """

        :param camera: Camera
        """
        self._camera = camera
        self._debug = debug
        self._image_processor = ImageProcessor(self._debug)


    def process(self, img):
        """
        Finds lanes on an image

        :param img:
        :return: boolean
        """
        img = self._camera.undistort(img)

        img = self._image_processor.prepare_image(img)



        return True