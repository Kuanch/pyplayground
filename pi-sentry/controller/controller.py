import importlib
from queue import Empty

import multiprocessing as mp


class Agent(object):
    def __init__(self, agent_name, action, *arg, **karg):
        self.queue = mp.Queue()
        self.finished = mp.Event()

        agent_path = 'src.' + agent_name
        agent = importlib.import_module(agent_path)
        self.agent_action = getattr(agent, action)(self.queue, self.finished, minutes=1)
        self.agent_action.start()

    def stream(self):
        try:
            while not self.finished.is_set():
                try:
                    frame = self.queue.get(False)
                except Empty:
                    continue
                yield(b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except GeneratorExit:
            self.agent_action.terminate()
            if not self.agent_action.is_alive():
                self.agent_action.join()
