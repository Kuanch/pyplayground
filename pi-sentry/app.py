from flask import Flask, render_template, Response, url_for, request, redirect
from flask_login import LoginManager, UserMixin, login_user
from flask_login import current_user, login_required, logout_user

from controller import controller

app = Flask(__name__)
app.secret_key = 'test'
api = 'agent_api'
agent_name = 'sentry'
action = 'Capture'

login_manager = LoginManager()
login_manager.init_app(app)
users = {'test': {'password': 'test'}}


class User(UserMixin):
    pass


@login_manager.user_loader
def user_loader(email):
    if email not in users:
        return
    user = User()
    user.id = email

    return user


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    name = request.form['user_name']
    if request.form['password'] == users[name]['password']:
        user = User()
        user.id = name
        login_user(user)

        return redirect(url_for('protected'))

    return 'Bad login'


@app.route('/protected')
@login_required
def protected():
    if current_user.is_active:
        return render_template('index.html')


@app.route('/logout')
def logout():
    logout_user()
    return 'Logged out'


@app.route('/')
def index():
    if not current_user.is_active:
        return redirect(url_for('protected'))
    else:
        return redirect(url_for('login'))


@app.route('/video_feed')
def video_feed():
    stream = getattr(controller, api)(agent_name, action)

    return Response(stream, mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/<string:agent_name>/<string:action>', methods=['GET', 'POST'])
def callback(agent_name, action):

    return getattr(controller, api)(agent_name, action)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
