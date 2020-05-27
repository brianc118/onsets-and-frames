// Soundfont players.
let playerInput, playerSample, playerMelody;

// MIDI Visualizers.
let vizInput, vizMelody, vizSample;


// The melodies for each of the players/visualizer pairs.
let input, melody, currentSample;
let playerSaidStop = false;  // So that we can loop.

init();

function init() {
    transcribeBtn.addEventListener('click', transcribeRequest);
    btnPlaySample.addEventListener('click', (e) => play(e, 2));

    playerSample = new mm.SoundFontPlayer('https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus');
    playerSample.callbackObject = {
        run: (note) => vizSample.redraw(note, true),
        stop: () => { }
    };
}

function _base64ToArrayBuffer(base64) {
    var binary_string = window.atob(base64);
    var len = binary_string.length;
    var bytes = new Uint8Array(len);
    for (var i = 0; i < len; i++) {
        bytes[i] = binary_string.charCodeAt(i);
    }
    return bytes.buffer;
}

async function transcribeRequest() {
    beforeEvaluating.hidden = true;
    whileEvaluating.hidden = false;
    afterEvaluating.hidden = true;

    const modelName = document.getElementById("modelName").value;
    const audioFile = document.getElementById("fileSelector").files[0];

    let formData = new FormData();
    formData.append("model_name", modelName)
    formData.append("audio_file", audioFile)

    try {
        console.log('try fetch')
        fetch('/transcribe', { method: "POST", body: formData })
            .then(response => response.json())
            .then(data => {
                console.log(data);
                bytes = _base64ToArrayBuffer(data["b64_midi"]);
                console.log(bytes);
                currentSample = mm.midiToSequenceProto(bytes);
                showInput(currentSample);
                play()
                beforeEvaluating.hidden = true;
                whileEvaluating.hidden = true;
                afterEvaluating.hidden = false;
            });
    } catch (e) {
        console.log('Houston we have problem...:', e);
    }
}


async function play() {
    let player, mel;
    player = playerSample;
    mel = currentSample;

    const btn = document.getElementById("btnPlaySample");
    if (player.isPlaying()) {
        stopPlayer(player, btn);
    } else {
        startPlayer(player, btn);
        player.loadSamples(mel).then(() => loopMelody(player, mel, btn))
    }
}

function loopMelody(player, mel, btn) {
    player.start(mel).then(() => {
        if (!playerSaidStop) {
            loopMelody(player, mel, btn);
        } else {
            stopPlayer(player, btn);
        }
    });
}

function stopPlayer(player, btn) {
    player.stop();
    playerSaidStop = true;
    btn.querySelector('.iconPlay').removeAttribute('hidden');
    btn.querySelector('.iconStop').setAttribute('hidden', true);
}

function startPlayer(player, btn) {
    playerSaidStop = false;
    btn.querySelector('.iconStop').removeAttribute('hidden');
    btn.querySelector('.iconPlay').setAttribute('hidden', true);
}

async function showInput(ns) {
    trimSilence(ns);
    vizSample = new mm.PianoRollSVGVisualizer(
        ns,
        document.getElementById('vizSample'),
        { noteRGB: '35,70,90', activeNoteRGB: '157, 229, 184', noteHeight: 3 });
}


function trimSilence(ns) {
    for (let i = 0; i < ns.length; i++) {
        const notes = ns[i].notes.sort((n1, n2) => n1.startTime - n2.startTime);
        const silence = notes[0].startTime;
        for (let j = 0; j < ns[i].notes.length; j++) {
            ns[i].notes[j].startTime -= silence;
            ns[i].notes[j].endTime -= silence;
        }
    }
}