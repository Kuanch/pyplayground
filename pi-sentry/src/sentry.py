import io
from queue import Empty
import multiprocessing as mp

import cv2
import picamera
import picamera.array
import numpy as np


class MotionDetector(object):
    def __init__(self):
        self.avg_frame = None

    def _detect(self, image):
        diff = cv2.absdiff(self.avg_frame, image)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        print(np.sum(np.sum(thresh)))
        if np.sum(np.sum(thresh)) > 100:
            return True
        return False

    def detect(self, image):
        motive = False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('float')
        if self.avg_frame is None:
            self.avg_frame = cv2.blur(gray, (21, 21))

        else:
            blur_image = cv2.blur(gray, (21, 21))
            cv2.accumulateWeighted(blur_image, self.avg_frame, 0.5)
            motive = self._detect(blur_image)

        return motive


class ImageProcessor(mp.Process):
    def __init__(self, queue, finished, motion_analysis):
        super(ImageProcessor, self).__init__()
        self.finished = finished
        self.queue = queue
        self.motion_detector = None
        if motion_analysis:
            self.motion_detector = MotionDetector()

        self.start()

    def _process(self, image):

        return image

    def process(self, image):
            self._process(image)
            if self.motion_detector is not None:
                motive = self.motion_detector.detect(image)
                if motive:
                    print('Motion Detected!')

    def run(self):
        while not self.finished.is_set():
            try:
                bytes_image = self.queue.get(False)
            except Empty:
                continue
            else:
                image = cv2.imdecode(np.frombuffer(bytes_image, dtype=np.uint8), -1)
                self.process(image)


class QueueOutput(object):
    def __init__(self, queue, finished, num_process, motion_analysis=True):
        self.queue = queue
        self.finished = finished
        self.stream = io.BytesIO()
        self.process_pool = [ImageProcessor(self.queue, self.finished, motion_analysis)
                             for i in range(num_process)]

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, put the last frame's data in the queue
            size = self.stream.tell()
            if size:
                self.stream.seek(0)
                self.queue.put(self.stream.read(size))
                self.stream.seek(0)
        self.stream.write(buf)

    def stop_processor(self):
        for p in self.process_pool:
            p.terminate()
            p.join()

    def save_image(self):
        bytes_image = self.queue.get(False)
        arr_image = np.frombuffer(bytes_image, np.uint8)
        cv2.imwrite('test_image.png', arr_image)

    def flush(self):
        self.queue.close()
        self.queue.join_thread()
        self.finished.set()
        self.stop_processor()
        print('end of recording')


class Capture(mp.Process):
    def __init__(self, queue, finished, minutes, num_postprocess_process=0):
        super(Capture, self).__init__()
        self.queue = queue
        self.finished = finished
        self.minutes = minutes

    def run(self):
        self.do_capture()

    def do_capture(self):
        with picamera.PiCamera(resolution='VGA', framerate=5) as camera:
            output = QueueOutput(self.queue, self.finished, 0)
            camera.start_recording(output, format='mjpeg')
            camera.wait_recording(60 * self.minutes)
            camera.stop_recording()
