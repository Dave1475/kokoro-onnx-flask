import logging
import io
import numpy as np
from flask import Flask, request, send_file, render_template
import kokoro_onnx
from kokoro_onnx import Kokoro
from scipy.io.wavfile import write as wav_write
import spacy
import argparse
#from memory_profiler import profile

# Set up logging
#logging.getLogger(kokoro_onnx.__name__).setLevel("DEBUG")


# Initialize Flask application
app = Flask(__name__)


# Initialize Kokoro with the model and voices file
# Set up argument parser
#python -m kokoro_onnx_flask.server --model kokoro-v0_19.onnx --voices voices.bin

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

def get_voice_from_hash_key(voice_hash):      
    index = voice_hash % len(global_voices_list)
    return global_voices_list[index]

def chunk_text(text, max_chunk_size=250):
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
        if token.pos_ == "CCONJ":  # Coordinating conjunctions (your original logic)
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

# Define the route for audio generation
@app.route('/')
def home():
    voices_list = kokoro.get_voices()  # Get the list of voices
    return render_template('index.html', voices=voices_list)
    
@app.route("/generate_audio", methods=["GET"])
#@profile
def generate_audio():
    text = request.args.get('text', default="", type=str)
    voice = request.args.get('voice', default="af_sarah", type=str) or "af_sarah"
    speed = request.args.get('kokoro_speed', default=1.0, type=float) or 1.0
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
    
    # Create a single WAV file from the combined samples
    output_stream = io.BytesIO()
    #print( sample_rate )
    wav_write(output_stream, sample_rate, combined_samples)
    output_stream.seek(0)
    
    return send_file(output_stream, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
