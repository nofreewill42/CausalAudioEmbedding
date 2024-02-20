import wave
import socketio

sio = socketio.Client()

sio.connect('http://127.0.0.1:5000')

# Open an audio file in binary mode
with wave.open('common_voice_en_79882.wav', 'rb') as wf:
    # Loop over the audio chunks
    while True:
        # Read a chunk of the audio file
        audio_chunk = wf.readframes(1024)  # Define the chunk size you want
        if not audio_chunk:
            break
        # Send the audio chunk to the server
        sio.emit('audio_chunk', audio_chunk)

# Disconnect from the server
sio.disconnect()
