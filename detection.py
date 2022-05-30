import torch, torchaudio, numpy as np
import torch.nn.functional as F
import streamlit as st
from models import ASTModel
from scipy.io import wavfile
import sounddevice as sd

n_class = 3
fshape = 128
tshape = 2
fstride=128
tstride=2
num_mel_bins=128
target_length=1024
model_size='small'

def make_features(wav_name, mel_bins, target_length=1024):
    wav_name = f"files/{wav_name}"
    waveform, sr = torchaudio.load(wav_name)
    waveform = waveform - waveform.mean()

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    #fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


@st.cache()
def load_model(filename):
    audio_model = ASTModel(label_dim=n_class, fshape=fshape, tshape=tshape, fstride=fstride, tstride=tstride,
                       input_fdim=num_mel_bins, input_tdim=target_length, model_size=model_size, pretrain_stage=False,
                       load_pretrained_mdl_path=filename)
    return audio_model


def predict(audio_path, ast_mdl):
    feats = make_features(audio_path, mel_bins=128)
    input_tdim = feats.shape[0]
    feats_data = feats.expand(1, input_tdim, 128)

    ast_mdl = ast_mdl.to('cpu')
    ast_mdl.eval()
    with torch.no_grad():
        prediction = F.softmax(ast_mdl(feats_data, task='ft_avgtok'), dim=1).to('cpu').detach()
        #prediction = torch.sigmoid().to('cpu').detach()
        prediction = prediction.data.cpu().numpy()[0]
        #print(prediction)
        #prediction = np.multiply(prediction, [0.4839, 1.8876, 2.4761])
        #print(prediction)
        sorted_indexes = np.argsort(prediction)[::-1]
    return prediction

def record_and_predict(sr=16000, channels=1, duration=5, filename='temp.wav'):
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=channels).reshape(-1)
    sd.wait()
    wavfile.write("temp.wav", sr, recording)
    return predict("temp.wav")
