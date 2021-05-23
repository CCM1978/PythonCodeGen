import os
import torch
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import load_model
import numpy as np
from scipy.io.wavfile import write

am_path = "./downloads/tts_train_fastspeech2_raw_phn_pypinyin_g2p_phone"
vocoder_path = "./downloads/csmsc.parallel_wavegan.v1/checkpoint-400000steps.pkl"

device = "cpu"
text2speech = Text2Speech(
    train_config=os.path.join(am_path, "config.yaml"),
    model_file=os.path.join(am_path, "train.loss.ave_5best.pth"),
    device=device,
    speed_control_alpha=1.0,
)
text2speech.spc2wav = None

fs = 24000
vocoder = load_model(vocoder_path).to(device).eval()
vocoder.remove_weight_norm()


def gen_tts(text, path):
    with torch.no_grad():
        wav, c, *_ = text2speech(text)
        wav = vocoder.inference(c)

    wav = wav.view(-1).cpu().numpy()
    pad_len = int(0.15 * fs)
    wav = np.pad(wav, (pad_len, pad_len), 'constant', constant_values=(0, 0))
    scaled = np.int16(wav / np.max(np.abs(wav)) * 32767)
    write(path, fs, scaled)
