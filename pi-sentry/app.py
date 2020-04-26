import time
from queue import Empty

import multiprocessing as mp
from flask import Flask, render_template, Response

from recording import Capture
from controller import controller

app = Flask(__name__)
service = 'sentry_api'


@app.route('/')
def index():

    return render_template('index.html')


def get_frame(queue, finished):
    while not finished.is_set():
        try:
            frame = queue.get(False)
        except Empty:
            continue
        else:
            print('got a frame')
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    queue = mp.Queue()
    finished = mp.Event()
    capture = Capture(queue, finished, 1)
    capture.start()
    # p = mp.Process(target=do_capture, args=(queue, finished))
    # p.start()

    return Response(get_frame(capture.queue, finished), mimetype='multipart/x-mixed-replace; boundary=frame')


'''
@app.route('/<string:mode>/<string:action>', methods=['GET', 'POST'])
def callback(mode, action):

    return getattr(controller, service)(mode, action)
'''


if __name__ == '__main__':
    app.run(host='0.0.0.0')
