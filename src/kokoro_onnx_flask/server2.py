from vosk import Model, KaldiRecognizer
import wave
import json
import soundfile as sf
import logging
import io
import numpy as np
from flask import Flask, request, send_file, render_template
import kokoro_onnx
from kokoro_onnx import Kokoro
from scipy.io.wavfile import write as wav_write
from scipy import signal
import spacy
import argparse

# USING THIS WITH MY OWN TEXT TO SPEECH PROGRAM AND PASSING IT BOUNDARY DATA TO HIGHLIGHT EACH WORD

# uses vosk-model to detect words and sends them as a newline separated string with the wav file followed by "SIZE" 
# my software breaks it into parts using that info
 

# Set up logging
# logging.getLogger(kokoro_onnx.__name__).setLevel("DEBUG")

# Initialize Flask application
app = Flask(__name__)

# Set up argument parser
parser = argparse.ArgumentParser(description="Initialize Kokoro with specified ONNX model and voices file.")
parser.add_argument("--model", required=True, help="Path to the ONNX file.")
parser.add_argument("--voices", required=True, help="Path to the voices file.")

# Parse arguments
args = parser.parse_args()

# Initialize Kokoro with arguments
kokoro = Kokoro(args.model, args.voices)

global_voices_list = kokoro.get_voices()

for voice in kokoro.get_voices():
    print(voice)

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Initialize Vosk
model_path = "vosk-model-small-en-us-0.15"
model = Model(model_path)
rec = KaldiRecognizer(model, 16000)
rec.SetWords(True)  # Enable word timing

def get_voice_from_hash_key(voice_hash):      
    index = voice_hash % len(global_voices_list)
    return global_voices_list[index]

def resample_audio(audio_data, orig_sr, target_sr):
    """
    Resample audio data to target sample rate
    """
    # Calculate resampling ratio
    ratio = target_sr / orig_sr
    
    # Calculate new length
    new_length = int(len(audio_data) * ratio)
    
    # Resample audio
    resampled = signal.resample(audio_data, new_length)
    
    return resampled.astype(np.int16)

def get_results(wav_data):
    """
    Process the WAV data using Vosk and return speech recognition results.
    This function uses wave.open to skip the WAV header and read raw PCM frames.
    """
    results = []
    with wave.open(io.BytesIO(wav_data), "rb") as wf:
        sample_rate = wf.getframerate()
        rec = KaldiRecognizer(model, sample_rate)
        rec.SetWords(True)  # Enable word timing

        while True:
            # Read raw PCM frames from the WAV file
            data = wf.readframes(4000)
            if len(data) == 0:
                break

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if 'result' in result:
                    results.extend(result['result'])

        final_result = json.loads(rec.FinalResult())
        if 'result' in final_result:
            results.extend(final_result['result'])

    return results

def chunk_text(text, max_chunk_size=220):
    doc = nlp(text)
    chunks = []
    
    def get_break_priority(token):
        """Returns priority score for break points (higher = better break)"""
        if token.text in ['.', '!', '?']:  # Highest priority
            return 4
        if token.text in [',', ';', ':']:  # Medium priority
            return 3
        if token.pos_ == "SCONJ":  # Subordinating conjunctions
            return 2
        if token.pos_ == "CCONJ":  # Coordinating conjunctions
            return 1
        if token.pos_ == "ADP" and token.dep_ == "prep":
            return 1
        if token.pos_ == "PRON" and token.dep_ == "relcl":
            return 1
        return 0
    
    def add_chunk(chunk_tokens):
        if chunk_tokens:
            chunk_start = chunk_tokens[0].idx
            chunk_end = chunk_tokens[-1].idx + len(chunk_tokens[-1].text)
            chunks.append(text[chunk_start:chunk_end].strip())
    
    for sent in doc.sents:
        sentence_tokens = list(sent)
        
        if len(sent.text) <= max_chunk_size:
            chunks.append(sent.text.strip())
            continue
            
        current_chunk = []
        current_length = 0
        
        for token in sentence_tokens:
            token_text = text[token.idx:token.idx + len(token.text)]
            token_length = len(token_text) + 1  # +1 for space
            
            if current_length + token_length > max_chunk_size:
                best_break = -1
                highest_priority = 0
                
                # Scan forwards to find best break point
                for i, t in enumerate(current_chunk):
                    priority = get_break_priority(t)
                    if priority > highest_priority:
                        highest_priority = priority
                        best_break = i
                    elif priority == highest_priority and i > best_break:
                        best_break = i
                
                # Split logic
                if best_break != -1 and highest_priority > 0:
                    split_index = best_break + 1
                    add_chunk(current_chunk[:split_index])
                    current_chunk = current_chunk[split_index:]
                    current_length = sum(len(text[t.idx:t.idx + len(t.text)]) + 1 
                                      for t in current_chunk)
                else:
                    # Fallback: Split at max size if no break points found
                    add_chunk(current_chunk)
                    current_chunk = []
                    current_length = 0
                    
            current_chunk.append(token)
            current_length += token_length
        
        if current_chunk:
            add_chunk(current_chunk)

    # Improved merging with buffer
    merged_chunks = []
    buffer = ""
    
    for chunk in chunks:
        if len(buffer) + len(chunk) + 1 <= max_chunk_size:
            buffer = f"{buffer} {chunk}".strip()
        else:
            if buffer:
                merged_chunks.append(buffer)
            buffer = chunk
    if buffer:
        merged_chunks.append(buffer)
        
    # Print final chunks for debugging
    print()
    for i, chunk in enumerate(merged_chunks):
        print(f"Chunk {i + 1}: {chunk}")
    print()

    return merged_chunks

# Define the routes
@app.route('/')
def home():
    voices_list = kokoro.get_voices()  # Get the list of voices
    return render_template('index.html', voices=voices_list)
    
@app.route("/generate_audio", methods=["GET"])
def generate_audio():
    text = request.args.get('text', default="", type=str)
    voice = request.args.get('voice', default="af_sarah", type=str) or "af_sarah"
    speed = request.args.get('kokoro_speed', default=1.0, type=float) or 1.0
    boundary = request.args.get('boundary', default="true", type=str) 
    voice_hash = request.args.get('voice_hash', default=1, type=int)
    
    if voice_hash > 1:
        voice = get_voice_from_hash_key(voice_hash)
    
    if not text:
        return "Text parameter is required", 400

    # Chunk text into smaller pieces while preserving words
    text = text.strip()  # Remove leading and trailing spaces
    chunks = chunk_text(text)
    
    # Initialize lists to store audio samples and ensure consistent sample rate
    all_samples = []
    sample_rate = None

    phonemes = None
    # Process each chunk and collect the audio samples
    for chunk in chunks:
        samples, chunk_sample_rate = kokoro.create(
            chunk, voice=voice, phonemes=phonemes, speed=speed, lang="en-us",
        )
        
        # Store the sample rate from the first chunk
        if sample_rate is None:
            sample_rate = chunk_sample_rate
        elif sample_rate != chunk_sample_rate:
            return "Inconsistent sample rates across chunks", 500
            
        # Append the samples to our collection
        all_samples.append(samples)

    # Concatenate all audio samples
    combined_samples = np.concatenate(all_samples)
    all_samples.clear()
   
 
        
    if boundary == "true":
    
        # Convert to 16-bit PCM and create in-memory WAV file
        combined_samples_int16 = (combined_samples * 32767).astype(np.int16)
        wav_output = io.BytesIO()
        wav_write(wav_output, sample_rate, combined_samples_int16)
        wav_output.seek(0)  # Reset stream position

        # Get Vosk results
        results = get_results(wav_output.read()) 
    
        # Format Vosk data
        vosk_data = ""
        for word_info in results:
            vosk_data += f"{word_info['start']:.3f} {word_info['word']}\n"

        # Add "KOKORO" at the end
        vosk_data += "KOKORO\n"
        
        wav_output.seek(0)   
        wav_size = len(wav_output.getvalue())
        size_str = "SIZE".encode('utf-8')
        response_stream = io.BytesIO()
        response_stream.write(wav_output.read())
        response_stream.write(vosk_data.encode('utf-8'))
        response_stream.write(wav_size.to_bytes(4, byteorder='big'))
        response_stream.write(size_str)
        response_stream.seek(0)
        return send_file(response_stream, mimetype="audio/wav")
        
    # Create a single WAV file from the combined samples
    output_stream = io.BytesIO()
    wav_write(output_stream, sample_rate, combined_samples)
    output_stream.seek(0)  

    return send_file(output_stream, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
