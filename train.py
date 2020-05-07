import os
from datetime import datetime

import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate
from onsets_and_frames import *

ex = Experiment('train_transcriber')


@ex.config
def config():
    logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 500000
    resume_iteration = None
    checkpoint_interval = 1000
    train_on = 'MAESTRO'

    batch_size = 8
    sequence_length = 327680
    model_complexity = 48

    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    learning_rate = 0.0006
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    validation_interval = 500

    ex.observers.append(FileStorageObserver.create(logdir))

    maestro_paths = ['data/MAESTRO']
    maestro_probabilities = [1]

    evaluate_maestro_path = 'data/MAESTRO'

    transform_audio_pipeline = [
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

    transform_audio = False
    sox_only = False


@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval, maestro_paths, maestro_probabilities, evaluate_maestro_path,
          transform_audio_pipeline, transform_audio, sox_only):
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_groups, validation_groups = ['train'], ['validation']

    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    transform = None
    if transform_audio:
        transform = lambda x: transform_wav_audio(x, SAMPLE_RATE, transform_audio_pipeline, sox_only=sox_only)
    if train_on == 'MAESTRO':
        dataset = ShuffledDataset([MAESTRO(maestro_path, groups=train_groups, sequence_length=sequence_length, transform=transform) for maestro_path in maestro_paths], maestro_probabilities)
        validation_datasets = [MAESTRO(evaluate_maestro_path, groups=validation_groups, sequence_length=validation_length), GuitarSet(sequence_length=None)]
    else:
        dataset = MAPS(groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'], sequence_length=sequence_length, transform=transform)
        validation_datasets = [MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length)]

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True)

    if resume_iteration is None:
        model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    for i, batch in zip(loop, cycle(loader)):
        batch['audio'] = batch['audio'].to(device)
        batch['label'] = batch['label'].to(device)
        batch['velocity'] = batch['velocity'].to(device)
        batch['onset'] = batch['onset'].to(device)
        batch['offset'] = batch['offset'].to(device)
        batch['frame'] = batch['frame'].to(device)
        predictions, losses = model.run_on_batch(batch)

        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        for key, value in {'loss': loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=i)

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                for j, validation_dataset in enumerate(validation_datasets):
                    for key, value in evaluate(validation_dataset, model, device).items():
                        writer.add_scalar(f'val/{type(validation_dataset).__name__}_{j}/' + key.replace(' ', '_'), np.mean(value), global_step=i)
            model.train()

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
