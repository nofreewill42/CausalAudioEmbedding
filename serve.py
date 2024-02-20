from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return "Streaming Server"

@socketio.on('audio_chunk')
def handle_audio_chunk(audio_data: bytes):
    # Here you would process the binary data
    print('Received audio chunk of size:', len(audio_data))

if __name__ == '__main__':
    socketio.run(app, debug=True)
