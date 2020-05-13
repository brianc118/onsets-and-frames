import argparse
import jams
import pretty_midi

def jams_to_midi(jam, q=1):
    # q = 1: with pitch bend. q = 0: without pitch bend.
    midi = pretty_midi.PrettyMIDI()
    annos = jam.search(namespace='note_midi')
    if len(annos) == 0:
        annos = jam.search(namespace='pitch_midi')
    midi_ch = pretty_midi.Instrument(program=25)
    for anno in annos:
        for note in anno:
            pitch = int(round(note.value))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('jams_file', type=str)
    parser.add_argument('midi_file', type=str)
    args = parser.parse_args()
    
    jam = jams.load(args.jams_file)
    midi = jams_to_midi(jam, q=0)
    midi.write(args.midi_file)
