# Automatic Music Transcription with Synthetic Data

This repository introduces the following features to the original PyTorch implementation of Onsets and Frames:

* Weighted sampling of datasets (which we call shuffling)
* Online audio augmentation with SoX and TorchAudio

It also contains `mds.py` and `tsne.py` which were used in visualising the timbre space of instruments.

See my thesis for details :)

# PyTorch Implementation of Onsets and Frames

This is a [PyTorch](https://pytorch.org/) implementation of Google's [Onsets and Frames](https://magenta.tensorflow.org/onsets-frames) model, using the [Maestro dataset](https://magenta.tensorflow.org/datasets/maestro) for training and the Disklavier portion of the [MAPS database](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) for testing.

## Instructions

This project is quite resource-intensive; 32 GB or larger system memory and 8 GB or larger GPU memory is recommended. 

### Downloading Datasets

The `data` subdirectory already contains the MAPS database. To download the MAESTRO dataset, GuitarSet dataset and Traditional Flute Dataset, first make sure that you have `ffmpeg` executable. Then run the following:

```bash
ffmpeg -version
cd data
./prepare_maestro.sh
./prepare_guitarset.sh
./prepare_traditional_flute_dataset.sh
```

This will require >200 GB of intermediate storage.

### Training

All package requirements are contained in `requirements.txt`. To train the model, run:

```bash
pip install -r requirements.txt
python train.py
```

`train.py` is written using [sacred](https://sacred.readthedocs.io/), and accepts configuration options such as:

```bash
python train.py with logdir=runs/model iterations=1000000
```

Trained models will be saved in the specified `logdir`, otherwise at a timestamped directory under `runs/`.

### Testing

To evaluate the trained model using the MAPS database, run the following command to calculate the note and frame metrics:

```bash
python evaluate.py runs/model/model-100000.pt
```

Specifying `--save-path` will output the transcribed MIDI file along with the piano roll images:

```bash
python evaluate.py runs/model/model-100000.pt --save-path output/
```

In order to test on the Maestro dataset's test split instead of the MAPS database, run:

```bash
python evaluate.py runs/model/model-100000.pt Maestro test
```

