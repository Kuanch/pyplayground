import io
import argparse
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
        image = cv2.blur(image, (3, 3))
        diff = cv2.absdiff(self.avg_frame, image)
        grey = np.sum(diff, axis=2)
        ret, thresh = cv2.threshold(grey, 30, 1, cv2.THRESH_BINARY)
        if np.sum(np.sum(thresh)) > 10:
            return True
        return False

    def detect(self, image):
        motive = False
        if self.avg_frame is None:
            self.avg_frame = cv2.blur(image, (3, 3))

        else:
            motive = self._detect(image)

        return motive


class ImageProcessor(mp.Process):
    def __init__(self, queue, finished, motion_analysis=True):
        super(ImageProcessor, self).__init__()
        self.finished = finished
        self.queue = queue
        self.motion_detector = None
        if motion_analysis:
            self.motion_detector = MotionDetector()

        self.start()

    def _process(self, image):
        pass

    def process(self):
        try:
            stream = io.BytesIO(self.queue.get(False))
        except Empty:
            pass
        else:
            image = np.frombuffer(stream, dtype=np.uint8)
            if self.motion_detector is not None:
                self.motion_detector.detect(image, self.avg_frame)
            self._process(image)
            stream.seek(0)

    def run(self):
        pass


class QueueOutput(object):
    def __init__(self, queue, finished, num_process, motion_analysis=True):
        self.queue = queue
        self.finished = finished
        self.stream = io.BytesIO()
        self.process_pool = [ImageProcessor(queue, finished, motion_analysis)
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
        print('write into queue')

    def flush(self):
        print('end of recording')
        self.queue.close()
        self.queue.join_thread()
        self.finished.set()


class Capture(object):
    def __init__(self, ):
        pass

    def do_capture(queue, finished, minutes):
        with picamera.PiCamera(resolution='VGA', framerate=30) as camera:
            output = QueueOutput(queue, finished, args.num_postprocess_process)
            camera.start_recording(output, format='mjpeg')
            camera.wait_recording(60 * minutes)
            camera.stop_recording()


def main(args):
    queue = mp.Queue()
    finished = mp.Event()
    capture = Capture()
    processor = ImageProcessor()
    capture_proc = mp.Process(target=capture.do_capture, args=(queue, finished, args.minutes))
    processing_procs = None

    # multi-process for postprocessing
    if args.num_postprocess_process:
        processing_procs = [mp.Process(target=processor.do_processing, args=(queue, finished))
                            for i in range(args.num_postprocess_process)]
        for proc in processing_procs:
            proc.start()

    # main process
    capture_proc.start()

    # if postprocessing processors exist
    if processing_procs is not None:
        for proc in processing_procs:
            proc.join()

    capture_proc.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minutes', default=1, type=int)
    parser.add_argument('--num_postprocess_process', default=0, type=int)
    args = parser.parse_args()
    main(args)
