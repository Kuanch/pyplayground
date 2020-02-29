"""
Concurrency for video
"""
import queue
import threading
from multiprocessing import Process
from multiprocessing import Queue


import cv2


class VideoCaptureThreading(object):
    def __init__(self, device):
        self._cam = cv2.VideoCapture(device)
        self._queue = queue.Queue()
        self._stop = True

    def _get_frame(self):
        while not self._stop:
            frame = self._cam.read()
            self._queue.put(frame, block=False)

        self._cam.release()

    def start(self):
        threading.Thread(target=self._get_frame, daemon=True, args=()).start()

    def read(self):
        return self._queue.get(block=False)
