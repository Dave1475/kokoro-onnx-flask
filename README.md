# kokoro-onnx-flask

A flask server for Kokoro-onnx

## Features:
Generates MP3 files from text input where paragraphs are separated by a blank line (an empty line between each paragraph), and combines them into a single MP3 file.

Each paragraph's text can be individually edited and updated after audio is generated in case mistakes are found, instead of having to regenerate the entire chapter.

Each paragraph can be spoken using a different voice by adding <v=number> at the start of it.

## Setup:
pip install git+https://github.com/Dave1475/kokoro-onnx-flask.git@main#egg=kokoro-onnx-flask

## Use
python -m kokoro_onnx_flask.server  --model kokoro-v0_19.onnx --voices voices.bin

http://127.0.0.1:5000
