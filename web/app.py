import os
from flask import Flask, render_template

app = Flask(__name__, static_url_path='', static_folder='../examples')

@app.route('/', methods=['GET'])
def showProgress():
    paths = os.listdir('./examples')

    def epoch_no(e):
        return -int(e.split('.png')[0].split('epoch')[-1])

    paths.sort(key=epoch_no)

    return render_template('index.html', imgs=paths)
