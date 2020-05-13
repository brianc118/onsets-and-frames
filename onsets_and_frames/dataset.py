import json
import os
from abc import abstractmethod
from glob import glob

import numpy as np
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import *
from .midi import parse_midi


class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, gpu_tensor=False, in_memory=False, transform=None):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.gpu_tensor = gpu_tensor
        self.in_memory = in_memory
        self.transform = transform
        self.random = np.random.RandomState(seed)

        self.data = []
        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc=f"Loading group {group} from {path}"):
                if in_memory:
                    self.data.append(self.load(*input_files))
                else:
                    self.data.append(tuple(input_files))

    def __getitem__(self, index):
        data = self.data[index] if self.in_memory else self.load(*self.data[index])
        result = dict(path=data['path'])

        if self.sequence_length is not None:
            audio_length = len(data['audio'])
            if audio_length < self.sequence_length:
                raise Exception(str(index) + " " + str(audio_length) + " " + str(self.sequence_length))
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length
            
            if self.transform:
                result['audio'] = self.transform(data['audio'][begin:end]).to(self.device) if self.gpu_tensor else self.transform(data['audio'][begin:end])
            else:
                result['audio'] = data['audio'][begin:end].to(self.device) if self.gpu_tensor else data['audio'][begin:end]
            result['label'] = data['label'][step_begin:step_end, :].to(self.device) if self.gpu_tensor else data['label'][step_begin:step_end, :]
            result['velocity'] = data['velocity'][step_begin:step_end, :].to(self.device) if self.gpu_tensor else data['velocity'][step_begin:step_end, :]
        else:
            if self.transform:
                result['audio'] = self.transform(data['audio']).to(self.device) if self.gpu_tensor else self.transform(data['audio'])
            else:
                result['audio'] = data['audio'].to(self.device) if self.gpu_tensor else data['audio']
            result['label'] = data['label'].to(self.device) if self.gpu_tensor else data['label']
            result['velocity'] = data['velocity'].to(self.device).float() if self.gpu_tensor else data['velocity']

        result['audio'] = result['audio'].float().div_(32768.0)
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)

        return result

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            path: str
                the path to the audio file

            audio: torch.ShortTensor, shape = [num_samples]
                the raw waveform

            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """
        saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt')
        if os.path.exists(saved_data_path):
            return torch.load(saved_data_path)

        audio, sr = soundfile.read(audio_path, dtype='int16')
        assert sr == SAMPLE_RATE

        audio = torch.ShortTensor(audio)
        audio_length = len(audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        tsv_path = tsv_path
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

        for onset, offset, note, vel in midi:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            onset_right = min(n_steps, left + HOPS_IN_ONSET)
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

            f = int(note) - MIN_MIDI
            label[left:onset_right, f] = 3
            label[onset_right:frame_right, f] = 2
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel

        data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)
        torch.save(data, saved_data_path)
        return data


class MAESTRO(PianoRollAudioDataset):

    def __init__(self, path, groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, transform=None):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, device, transform)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        if group not in self.available_groups():
            # year-based grouping
            flacs = sorted(glob(os.path.join(self.path, group, '*.flac')))
            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(self.path, group, '*.wav')))

            midis = sorted(glob(os.path.join(self.path, group, '*.midi')))
            files = list(zip(flacs, midis))
            if len(files) == 0:
                raise RuntimeError(f'Group {group} is empty')
        else:
            metadata = json.load(open(os.path.join(self.path, 'maestro-v1.0.0.json')))
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for row in metadata if row['split'] == group])

            files = [(audio if os.path.exists(audio) else audio.replace('.flac', '.wav'), midi) for audio, midi in files]

        result = []
        for audio_path, midi_path in files:
            tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result


class MAPS(PianoRollAudioDataset):
    def __init__(self, path='data/MAPS', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, transform=None):
        super().__init__(path, groups if groups is not None else ['ENSTDkAm', 'ENSTDkCl'], sequence_length, seed, device, transform)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        flacs = glob(os.path.join(self.path, 'flac', '*_%s.flac' % group))
        tsvs = [f.replace('/flac/', '/tsv/matched/').replace('.flac', '.tsv') for f in flacs]
        midis = [f.replace('/flac/', '/midi/').replace('.flac', '.mid') for f in flacs]
        for midi_path, tsv_filename in zip(sorted(midis), sorted(tsvs)):
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        return sorted(zip(flacs, tsvs))

class GuitarSet(PianoRollAudioDataset):
    def __init__(self, path='data/guitarset', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, transform=None):
        super().__init__(path, groups if groups is not None else ['default'], sequence_length, seed, device, transform)

    @classmethod
    def available_groups(cls):
        return ['default']
    
    def files(self, group):
        if group not in self.available_groups():
            return []
        
        flacs = sorted(glob(os.path.join(self.path, '*.flac')))
        if len(flacs) == 0:
            flacs = sorted(glob(os.path.join(self.path, '*.wav')))

        midis = sorted(glob(os.path.join(self.path, '*.mid')))
        files = list(zip(flacs, midis))
        if len(files) == 0:
            raise RuntimeError(f'Group {group} is empty')
        
        result = []
        for audio_path, midi_path in files:
            tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result


class TraditionalFluteDataset(PianoRollAudioDataset):
    def __init__(self, path='data/traditional_flute_dataset', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, gpu_tensor=False, in_memory=False, transform=None):
        super().__init__(path, groups if groups is not None else ['default'], sequence_length, seed, device, gpu_tensor, in_memory, transform)

    @classmethod
    def available_groups(cls):
        return ['default']
    
    def files(self, group):
        if group not in self.available_groups():
            return []
        
        flacs = sorted(glob(os.path.join(self.path, 'audio', '*.flac')))
        if len(flacs) == 0:
            flacs = sorted(glob(os.path.join(self.path, 'audio', '*.wav')))

        midis = sorted(glob(os.path.join(self.path, 'ground_truth', '*.mid')))
        files = list(zip(flacs, midis))
        if len(files) == 0:
            raise RuntimeError(f'Group {group} is empty')
        
        result = []
        for audio_path, midi_path in files:
            tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
            result.append((audio_path, tsv_filename))
        return result 

class ShuffledDataset(Dataset):
    def __init__(self, datasets, probabilities):
        assert(sum(probabilities) == 1)
        assert(all(p >= 0 for p in probabilities))
        assert(len(datasets) > 0)
        assert(all(len(dataset) == len(datasets[0]) for dataset in datasets))
        self.datasets = datasets
        self.probabilities = probabilities

    def __getitem__(self, index):
        i = np.random.choice(len(self.datasets), 1, p=self.probabilities)[0].item()
        return self.datasets[i][index]

    def __len__(self):
        return len(self.datasets[0])