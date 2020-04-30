import os
import torchaudio
import subprocess

from ..transform import *

AUDIO_TRANSFORM_PIPELINE_TORCH = [
    # Pitch shift.
    ("pitch", {"n_semitones": (0.1, 0.1, "linear"),}),
    # Contrast (simple form of compression).
    ("contrast", {"amount": (100.0, 100.0, "linear"),}),
    # Two independent EQ modifications.
    (
        "equalizer",
        {
            "frequency": (4096.0, 4096.0, "log"),
            "width_q": (2.0, 2.0, "linear"),
            "gain_db": (-10.0, -10.0, "linear"),
        },
    ),
    (
        "equalizer",
        {
            "frequency": (32.0, 32.0, "log"),
            "width_q": (2.0, 2.0, "linear"),
            "gain_db": (5, 5.0, "linear"),
        },
    ),
    # Reverb (for now just single-parameter).
    # ("reverb", {"reverberance": (70.0, 70.0, "linear"),}),
]

FILENAMES = [
    # "onsets_and_frames/test/assets/sinewave.wav",
    # "onsets_and_frames/test/assets/steam-train-whistle-daniel_simon.wav",
    # "onsets_and_frames/test/assets/440Hz_44100Hz_16bit_05sec.wav",
    "onsets_and_frames/test/assets/MAPS_MUS-alb_esp2_AkPnCGdD.wav",
]

def spec(f):
    subprocess.call(['sox', f'{f}.wav', '-n', 'spectrogram', '-o', f'{f}_spec.png'])

if __name__ == "__main__":
    for filename in FILENAMES:
        x, sr = torchaudio.load(filename)
        f = os.path.splitext(filename)[0]
        y = transform_wav_audio(
            x, sr, pipeline=AUDIO_TRANSFORM_PIPELINE_TORCH, sox_only=False
        )
        torchaudio.save(f"{f}_tr.wav", y, sr)
        y = transform_wav_audio(
            x, sr, pipeline=AUDIO_TRANSFORM_PIPELINE_TORCH, sox_only=True
        )
        torchaudio.save(f"{f}_tr_sox.wav", y, sr)
        spec(f)
        spec(f'{f}_tr')
        spec(f'{f}_tr_sox')
