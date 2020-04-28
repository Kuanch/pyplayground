import importlib
from queue import Empty

import multiprocessing as mp


def agent_api(agent_name, action, *arg, **karg):
    queue = mp.Queue()
    finished = mp.Event()

    agent_path = 'src.' + agent_name
    agent = importlib.import_module(agent_path)
    agent_action = getattr(agent, action)(queue, finished, minutes=1)
    agent_action.start()

    while not finished.is_set():
        try:
            frame = queue.get(False)
        except Empty:
            continue
        else:
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
