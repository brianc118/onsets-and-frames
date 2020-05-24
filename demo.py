import io
import json

from transcribe import *

from flask import Flask, jsonify, request


ALLOWED_EXTENSIONS = {'mp3', 'wav'}

app = Flask(__name__)

def getfiletype(filename):
    return None if '.' not in filename else filename.rsplit('.', 1)[1].lower()


@app.route('/', methods=['GET'])
def upload_file():
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="/transcribe" method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if request.method == 'POST':
        audio_file = request.files['file']
        # convert to 16 kHz sr
        if getfiletype(audio_file) not in ALLOWED_EXTENSIONS:
            # error
            return
        

        img_bytes = audio_file.read()
        return jsonify({'yeet': 'hey'})


if __name__ == '__main__':
    app.run(port=8080)