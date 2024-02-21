from flask import Flask
from flask_socketio import SocketIO
import numpy as np
import torch

from audio_processor import AudioProcessor


device = 'cuda' if torch.cuda.is_available() else 'cpu'
audio_processor = AudioProcessor()
audio_processor.to(device)
audio_processor.eval()


app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return "Streaming Server"

@socketio.on('audio_chunk')
def handle_audio_chunk(audio_data: bytes):
    # Process the audio
    with torch.no_grad():
        results = audio_processor.process_audio(audio_data)  # results is a tensor (b, t, d)
    print(results.shape)
    results = results.cpu().detach().numpy().tolist()
    # Send the results
    socketio.emit('audio_results', results)


if __name__ == '__main__':
    socketio.run(app, debug=True)