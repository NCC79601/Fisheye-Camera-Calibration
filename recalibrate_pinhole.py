from callibrator import Callibrator

callibrator = Callibrator()

callibrator.load_calibration()

callibrator.recalibrate_pinhole('./images_pinhole')