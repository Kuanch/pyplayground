from flask import Flask, render_template, Response

from controller import controller

app = Flask(__name__)
api = 'agent_api'
agent_name = 'sentry'
action = 'Capture'


@app.route('/')
def index():

    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    stream = getattr(controller, api)(agent_name, action)

    return Response(stream, mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/<string:agent_name>/<string:action>', methods=['GET', 'POST'])
def callback(agent_name, action):

    return getattr(controller, api)(agent_name, action)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
