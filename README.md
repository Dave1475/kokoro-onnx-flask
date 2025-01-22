# kokoro-onnx-flask

A flask server for Kokoro-onnx

## Features:
Generates MP3 files from text input where sentences are separated by a blank line (an empty line between each sentence), and combines them into a single MP3 file.

## Setup:
pip install git+https://github.com/Dave1475/kokoro-onnx-flask.git@main#egg=kokoro-onnx-flask

## Use
python -m kokoro_onnx_flask.server  --model kokoro-v0_19.onnx --voices voices.bin
