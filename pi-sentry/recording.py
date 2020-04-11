import os
import io
import time
import argparse
from queue import Empty
import multiprocessing as mp

import picamera
from PIL import Image


class QueueOutput(object):
    def __init__(self, queue, finished):
        self.queue = queue
        self.finished = finished
        self.stream = io.BytesIO()

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


def do_capture(queue, finished, minutes):
    with picamera.PiCamera(resolution='VGA', framerate=30) as camera:
        output = QueueOutput(queue, finished)
        camera.start_recording(output, format='mjpeg')
        camera.wait_recording(60 * minutes)
        camera.stop_recording()


def do_processing(queue, finished):
    while not finished.wait(0.1):
        try:
            stream = io.BytesIO(queue.get(False))
        except Empty:
            pass
        else:
            stream.seek(0)
            image = Image.open(stream)
            # Pretend it takes 0.1 seconds to process the frame; on a quad-core
            # Pi this gives a maximum processing throughput of 40fps
            time.sleep(0.1)
            print('%d: Processing image with size %dx%d' % (
                  os.getpid(), image.size[0], image.size[1]))


def main(args):
    queue = mp.Queue()
    finished = mp.Event()
    capture_proc = mp.Process(target=do_capture, args=(queue, finished, args.minutes))
    processing_procs = None

    # multi-process for postprocessing
    if args.num_postprocess_process:
        processing_procs = [mp.Process(target=do_processing, args=(queue, finished))
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
