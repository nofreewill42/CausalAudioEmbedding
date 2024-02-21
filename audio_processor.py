import wave
import pathlib
import numpy as np
import librosa
import torch
import torch.nn as nn


def audio_bytes_to_numpy(audio_bytes: bytes) -> np.ndarray:
    '''
    Convert raw audio bytes to a numpy array.
    '''
    audio_numpy = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_numpy

def load_audio(audio_path: pathlib.Path, start_time: float=0.0, duration: float=None) -> np.ndarray:
    '''
    Load audio from file starting at start_time and lasting duration seconds.
    '''
    with wave.open(str(audio_path), 'rb') as wf:
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        start_frame = int(start_time * frame_rate)
        if duration is None:
            num_frames = wf.getnframes()
        else:
            num_frames = int(duration * frame_rate)
        wf.setpos(start_frame)
        audio_bytes = wf.readframes(num_frames)
    #audio_numpy = audio_bytes_to_numpy(audio_bytes)
    return audio_bytes


class Audio2LogMelSpectrogram:
    def __init__(self):
        self.sr = 16000
        self.n_fft = 400
        self.hop_length = 160
        self.n_mels = 128

        filters = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels)
        self.mel_basis = torch.tensor(filters, dtype=torch.float32).unsqueeze(0)
        self.window = torch.hann_window(self.n_fft)
    
    def __call__(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        # Compute the mel spectrogram
        stft = torch.stft(audio_tensor, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=False, return_complex=True)
        power_spectrogram = stft[..., :-1].abs() ** 2  # torch.Size([1, 257, 637]), self.mel_basis.shape = torch.Size([1, 128, 257])
        mel_spectrogram = torch.matmul(self.mel_basis, power_spectrogram)
        logmel_spectrogram = torch.clamp(mel_spectrogram, min=1e-10).log10()
        logmel_spectrogram = torch.maximum(logmel_spectrogram, logmel_spectrogram.max() - 8.0)
        logmel_spectrogram = (logmel_spectrogram + 4.0) / 4.0
        return logmel_spectrogram
    
    def to(self, device: str):
        self.mel_basis = self.mel_basis.to(device)
        self.window = self.window.to(device)
        return self
    

class CausalAudioEncoder(nn.Module):
    def __init__(self, d_model: int=512, n_q: int=8, n_layer: int=8, d_ff: int=2048, dropout: float=0.1, activation: str='relu'):
        super().__init__()
        self.d_model = d_model
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_q, dim_feedforward=d_ff, dropout=dropout,
                                                   activation=activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

    def forward(self, x: torch.Tensor, kv_cache: torch.Tensor=None) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        src_mask = torch.triu(
            torch.full((x.size(1), x.size(1)), float('-inf'), device=x.device, dtype=x.dtype),
            diagonal=1)
        x = self.encoder(x, mask=src_mask)
        return x, kv_cache


class AudioProcessor:
    def __init__(self):
        self.audio2mel_spectrogram = Audio2LogMelSpectrogram()
        self.causal_audio_encoder = CausalAudioEncoder()
        self.kv_cache = None
        self.device = 'cpu'
    
    def to(self, device: str):
        self.audio2mel_spectrogram.to(device)
        self.causal_audio_encoder.to(device)
        self.device = device
        return self
    
    def eval(self):
        self.causal_audio_encoder.eval()
        return self

    def process_audio(self, audio_data: bytes) -> dict:
        '''
        Process audio and return results.
        '''
        audio_numpy = audio_bytes_to_numpy(audio_data)
        audio_tensor = torch.tensor(audio_numpy, dtype=torch.float32, device=self.device)
        audio_tensor = audio_tensor.unsqueeze(0) / 32768.0
        logmel_spectrogram = self.audio2mel_spectrogram(audio_tensor)
        results, self.kv_cache = self.causal_audio_encoder(logmel_spectrogram, kv_cache=self.kv_cache)
        return results

    def __del__(self):
        pass


if __name__ == '__main__':
    audio_processor = AudioProcessor()
    audio_path = pathlib.Path('common_voice_en_79882.wav')
    audio_bytes = load_audio(audio_path)
    results = audio_processor.process_audio(audio_bytes)