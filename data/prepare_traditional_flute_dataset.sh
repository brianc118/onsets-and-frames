#! /bin/bash

# Assumes flute dataset is located in ./traditional_flute_dataset/

# rm traditional_flute_dataset/audio/allemande_fifth_fragment_preston.*
# rm traditional_flute_dataset/audio/allemande_second_fragment_preston.*
# mv traditional_flute_dataset/audio/allemande_fifth_fragment_preston_resampled.wav \
#    traditional_flute_dataset/audio/allemande_fifth_fragment_preston.wav
# mv traditional_flute_dataset/audio/allemande_second_fragment_preston_resampled.wav \
#    traditional_flute_dataset/audio/allemande_second_fragment_preston.wav

conda activate pytorch

echo Converting annotation GT to MIDI
COUNTER=0
for f in traditional_flute_dataset/ground_truth/*.gt; do
    COUNTER=$((COUNTER + 1))
    echo -ne "\rConverting ($COUNTER/30) ..."
    if [[ "$f" == "traditional_flute_dataset/audio/allemande_fifth_fragment_preston" || 
          "$f" == "traditional_flute_dataset/audio/allemande_second_fragment_preston.wav" ]]
    then
        python3 ../to_midi.py $f ${f/\.gt/.mid} --pitch_shift -2
    else
        python3 ../to_midi.py $f ${f/\.gt/.mid}
    fi  
done

echo
echo Converting the audio files to FLAC ...
COUNTER=0
for f in traditional_flute_dataset/audio/*.wav; do
    COUNTER=$((COUNTER + 1))
    echo -ne "\rConverting ($COUNTER/30) ..."
    ffmpeg -y -loglevel fatal -i $f -ac 1 -ar 16000 ${f/\.wav/.flac}
done

echo
echo Preparation complete!

