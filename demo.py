import ffmpeg
import io
import json
import os
import tempfile
from base64 import b64encode
from tqdm import tqdm

from onsets_and_frames import *
from transcribe import *

from flask import Flask, jsonify, request, render_template, send_from_directory, abort


DUMMY = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLOWED_EXTENSIONS = {"flac", "mp3", "wav"}
ONSET_THRESHOLD = 0.5
FRAME_THRESHOLD = 0.5


def getfiletype(filename):
    return None if "." not in filename else filename.rsplit(".", 1)[1].lower()


if __name__ == "__main__":
    with torch.no_grad():
        print("Loading models")
        models = {}
        model_names = list(os.listdir("models"))
        for model_name in tqdm(model_names):
            models[model_name] = torch.load(
                os.path.join("models", model_name), map_location=DEVICE
            ).eval()

        app = Flask(__name__)
        # app.config["CLIENT_MIDIS"] = ""

        @app.route("/", methods=["GET"])
        def upload_file():
            return render_template("main.html", model_names=model_names)

        @app.route("/transcribe", methods=["POST"])
        def get_transcription():
            if request.method == "POST":
                print(request)
                if "audio_file" not in request.files:
                    print("error did not upload a file")
                    return jsonify({"error": "Did not upload a file"})
                audio_file = request.files["audio_file"]
                if "model_name" not in request.values:
                    print("error no model name")
                    return jsonify({"error": "Did not specify model_name"})
                model_name = request.values["model_name"]
                model = models[model_name]
                ext = getfiletype(audio_file.filename)
                if ext not in ALLOWED_EXTENSIONS:
                    # error
                    return jsonify(
                        {
                            "Error": f"Invalid file. Does not have extension from {str(ALLOWED_EXTENSIONS)}. Extension is {ext}."
                        }
                    )
                audio_bytes = audio_file.read()
                prev_tempdir = tempfile.tempdir
                contents = {"error": "failed to transcribe"}
                tempfile.tempdir = "/dev/shm"
                with tempfile.NamedTemporaryFile(
                    suffix="midi"
                ) as midi_file, tempfile.NamedTemporaryFile(
                    suffix=ext
                ) as raw_audio_file, tempfile.NamedTemporaryFile(
                    suffix="wav"
                ) as audio_file:
                    raw_audio_file.write(audio_bytes)
                    ffmpeg.input(raw_audio_file.name).output(
                        audio_file.name, ac=1, ar=16000, format="wav"
                    ).overwrite_output().run()
                    if not DUMMY:
                        audio = load_and_process_audio(audio_file.name, None, DEVICE)
                        predictions = transcribe(model, audio)
                        p_est, i_est, v_est = extract_notes(
                            predictions["onset"],
                            predictions["frame"],
                            predictions["velocity"],
                            ONSET_THRESHOLD,
                            FRAME_THRESHOLD,
                        )

                        scaling = HOP_LENGTH / SAMPLE_RATE

                        i_est = (i_est * scaling).reshape(-1, 2)
                        p_est = np.array(
                            [midi_to_hz(MIN_MIDI + midi) for midi in p_est]
                        )

                        save_midi(midi_file.name, p_est, i_est, v_est)
                        contents = {"b64_midi": b64encode(midi_file.read().decode())}
                    else:
                        with open("dummy.midi", "rb") as dummy_midi:
                            contents = {"b64_midi": b64encode(dummy_midi.read()).decode()}

                tempfile.tempdir = prev_tempdir

                return jsonify(contents)

        @app.route("/get-midi/<path:midi_name>")
        def get_midi(midi_name):
            try:
                root_dir = os.path.dirname(os.getcwd())
                return send_from_directory(
                    os.path.join(root_dir, "static", "midis"),
                    filename=midi_name,
                    as_attachment=True,
                )
            except FileNotFoundError:
                abort(404)

        app.run(port=8080)
