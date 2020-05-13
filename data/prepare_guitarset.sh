#! /bin/bash

wget -O audio_hex-pickup_debleeded.zip https://zenodo.org/record/3371780/files/audio_hex-pickup_debleeded.zip?download=1
wget -O annotation.zip https://zenodo.org/record/3371780/files/annotation.zip?download=1
mkdir guitarset
unzip -o annotation.zip -d guitarset | awk 'BEGIN{ORS=""} {print "\rExtracting " NR "/361 ..."; system("")} END {print "\ndone\n"}'
unzip -o audio_hex-pickup_debleeded.zip -d guitarset | awk 'BEGIN{ORS=""} {print "\rExtracting " NR "/361 ..."; system("")} END {print "\ndone\n"}'

conda activate pytorch

echo Converting annotation JAMS to MIDI
COUNTER=0
for f in guitarset/*.jams; do
    COUNTER=$((COUNTER + 1))
    echo -ne "\rConverting ($COUNTER/361) ..."
    python3 ../to_midi.py $f ${f/\.jams/.mid}
done

echo
echo Converting the audio files to FLAC ...
COUNTER=0
for f in guitarset/*.wav; do
    COUNTER=$((COUNTER + 1))
    echo -ne "\rConverting ($COUNTER/361) ..."
    ffmpeg -y -loglevel fatal -i $f -ac 1 -ar 16000 ${f/\.wav/.flac}
done

echo
echo Preparation complete!