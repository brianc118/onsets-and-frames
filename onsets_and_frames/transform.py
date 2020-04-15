"""Audio transform"""

import math
import random
import sox
import subprocess
import tempfile
import torch
import torchaudio
import logging

from .constants import *

logging.getLogger("sox").setLevel(logging.ERROR)

SHARED_PIPELINE = [
    # Reverb (for now just single-parameter).
    ("reverb", {"reverberance": (0.0, 70.0, "linear"),}),
]

AUDIO_TRANSFORM_PIPELINE_TORCH = [
    # Pitch shift.
    ("pitch", {"n_semitones": (-0.1, 0.1, "linear"),}),
    # Contrast (simple form of compression).
    ("contrast", {"amount": (0.0, 100.0, "linear"),}),
    # Two independent EQ modifications.
    (
        "equalizer",
        {
            "frequency": (32.0, 4096.0, "log"),
            "width_q": (2.0, 2.0, "linear"),
            "gain_db": (-10.0, 5.0, "linear"),
        },
    ),
    (
        "equalizer",
        {
            "frequency": (32.0, 4096.0, "log"),
            "width_q": (2.0, 2.0, "linear"),
            "gain_db": (-10.0, 5.0, "linear"),
        },
    ),
]


class AudioTransformParameter:
    def __init__(self, name, min_value, max_value, scale):
        if scale not in ("linear", "log"):
            raise ValueError("invalid parameter scale: %s" % scale)

        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.scale = scale

    def sample(self):
        if self.scale == "linear":
            return random.uniform(self.min_value, self.max_value)
        else:
            log_min_value = math.log(self.min_value)
            log_max_value = math.log(self.max_value)
            return math.exp(random.uniform(log_min_value, log_max_value))


class AudioTransformStage:
    def __init__(self, name, params):
        self.name = name
        self.params = params

    def apply(self, transformer):
        args = dict((param.name, param.sample()) for param in self.params)
        getattr(transformer, self.name)(**args)


def construct_pipeline(pipeline):
    return [
        AudioTransformStage(
            name=stage_name,
            params=[
                AudioTransformParameter(param_name, l, r, scale)
                for param_name, (l, r, scale) in params_dict.items()
            ],
        )
        for stage_name, params_dict in pipeline
    ]


def run_pipeline(pipeline, input_filename, output_filename):
    transformer = sox.Transformer()
    transformer.set_globals(guard=True)
    for stage in pipeline:
        stage.apply(transformer)
    transformer.build(input_filename, output_filename)


def transform_wav_audio(wav_audio, sample_rate, pipeline, sox_only=False):
    prev_tempdir = tempfile.tempdir
    tempfile.tempdir = "/dev/shm"

    wav_audio_dtype = wav_audio.dtype
    wav_audio_size = wav_audio.size()
    if wav_audio_dtype == torch.int16:
        wav_audio = wav_audio.to(torch.float32) / (2 ** 15)
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_input:
        torchaudio.save(temp_input.name, wav_audio, sample_rate)

        if not sox_only:
            ec = torchaudio.sox_effects.SoxEffectsChain()
            ec.set_input_file(temp_input.name)
            # TODO: Allow volume of noise to be adjusted
            # si_in, _ = torchaudio.info(temp_input.name)
            # len_in_seconds = si_in.length / si_in.channels / si_in.rate
            # Add noise before all other pipeline steps.
            # noise_vol = random.uniform(
            #     audio_transform_min_noise_vol, audio_transform_max_noise_vol
            # )
            # ec.append_effect_to_chain(
            #     "synth", [len_in_seconds, audio_transform_noise_type, "mix"]
            # )
            # ec.append_effect_to_chain("vol", [noise_vol])
            for feature, params_dict in pipeline:
                ec.append_effect_to_chain(
                    feature,
                    [
                        AudioTransformParameter(param_name, l, r, scale).sample()
                        for param_name, (l, r, scale) in params_dict.items()
                    ],
                )
            wav_audio, sr = ec.sox_build_flow_effects()
            assert sr == sample_rate

        if RUN_SHARED_PIPELINE or sox_only:
            torchaudio.save(temp_input.name, wav_audio, sample_rate)
            # Running pysox transformation pushes it to >5s/it
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_output:
                combined_pipeline = pipeline if sox_only else []
                combined_pipeline += SHARED_PIPELINE if RUN_SHARED_PIPELINE else []
                run_pipeline(
                    construct_pipeline(combined_pipeline),
                    temp_input.name,
                    temp_output.name,
                )
                wav_audio, sr = torchaudio.load(temp_output.name)
                assert sr == sample_rate
    # interpolate if size wrong
    wav_audio = wav_audio.squeeze()
    if (
        wav_audio.size()[0] != wav_audio_size[0]
        and abs(1 - wav_audio.size()[0] / wav_audio_size[0]) < 0.01
    ):
        # scale wav_audio to wav_audio
        if wav_audio.size()[0] > wav_audio_size[0]:
            wav_audio = wav_audio[: wav_audio_size[0]]
        else:
            m = torch.nn.ConstantPad1d((0, wav_audio_size[0] - wav_audio.size()[0]), 0)
            wav_audio = m(wav_audio)
    if len(wav_audio_size) > len(wav_audio.size()):
        wav_audio = wav_audio.unsqueeze(0)

    assert (
        wav_audio_size == wav_audio.size()
    ), "Transformed audio size is different. Original: {}. Transformed: {}".format(
        wav_audio_size, wav_audio.size()
    )
    if wav_audio_dtype == torch.int16:
        wav_audio = wav_audio * (2 ** 15)
        wav_audio = wav_audio.to(torch.int16)
    tempfile.tempdir = prev_tempdir
    return wav_audio
