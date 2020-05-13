import argparse
import csv
import jams
import librosa
import os
import pretty_midi

def jams_to_midi(jam, q=1, shift=0):
    # q = 1: with pitch bend. q = 0: without pitch bend.
    midi = pretty_midi.PrettyMIDI()
    annos = jam.search(namespace='note_midi')
    if len(annos) == 0:
        annos = jam.search(namespace='pitch_midi')
    midi_ch = pretty_midi.Instrument(program=25)  # steel string guitar
    for anno in annos:
        for note in anno:
            pitch = int(round(note.value)) + shift
            bend_amount = int(round((note.value - pitch) * 4096))
            st = note.time
            dur = note.duration
            n = pretty_midi.Note(
                velocity=100,
                pitch=pitch, start=st,
                end=st + dur
            )
            pb = pretty_midi.PitchBend(pitch=bend_amount * q, time=st)
            midi_ch.notes.append(n)
            midi_ch.pitch_bends.append(pb)
    if len(midi_ch.notes) != 0:
        midi.instruments.append(midi_ch)
    return midi

def gt_to_midi(gt, shift=0):
    midi = pretty_midi.PrettyMIDI()
    cr = csv.reader(open(gt))
    midi_ch = pretty_midi.Instrument(program=73)  # flute
    for i, row in enumerate(cr):
        st, freq, dur = float(row[0]), float(row[1]), float(row[2])
        if freq > 0:
            pitch = int(round(librosa.hz_to_midi(freq))) + shift
            n = pretty_midi.Note(
                velocity=100,
                pitch=pitch, start=st,
                end=st + dur
            )
            midi_ch.notes.append(n)
    if len(midi_ch.notes) != 0:
        midi.instruments.append(midi_ch)
    return midi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', type=str)
    parser.add_argument('midi_file', type=str)
    parser.add_argument('--pitch_shift', type=int, default=0)
    args = parser.parse_args()

    _, ext = os.path.splitext(args.in_file)
    if ext in [".jam", ".jams"]:
        jam = jams.load(args.in_file)
        midi = jams_to_midi(jam, q=0)
        midi.write(args.midi_file)
    else:
        midi = gt_to_midi(args.in_file)
        midi.write(args.midi_file)
