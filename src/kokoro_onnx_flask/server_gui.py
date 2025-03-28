import io
from flask import Flask, request, send_file, render_template, jsonify, Response
from io import BytesIO
from kokoro import KPipeline
import numpy as np
import wave
import time
from flask_cors import CORS

import base64
from pydub import AudioSegment
from scipy.io.wavfile import write as wav_write

from misaki import en

import torch
import random
import re


app = Flask(__name__)
CORS(app) 

# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
pipeline = KPipeline(lang_code='a')  # Ensure lang_code matches voice
global_time = 0

global_voices_list = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", 
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", 
    "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", 
    "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily", 
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis"
]

def get_voice_from_hash_key(voice_hash):
    index1 = voice_hash % len(global_voices_list)
    random.seed(voice_hash)  # Ensures reproducibility
    if random.randint(1, 20) == 1:  # 5% chance (1 out of 20) to return a single voice
        return global_voices_list[index1]
    index2 = (voice_hash // len(global_voices_list)) % len(global_voices_list)
    if index1 == index2:  # Ensure different voices
        index2 = (index2 + 1) % len(global_voices_list)
    return f"{global_voices_list[index1]},{global_voices_list[index2]}"


def generate_audio(text: str, voice: str, speed: float = 1.0):
    #words = text.split()  # Split input into words

    # Use regex to split the text while keeping the spaces after each word
    words_with_spaces = re.findall(r'\S+\s*', text)
    # Remove one trailing space from each word (if present)
    words = [word[:-1] if word[-1].isspace() else word for word in words_with_spaces]

    whitespace_token = en.MToken(text=" ", phonemes="", tag="", whitespace="")


    processed_tokens = []
    current_phrase = []

    for word in words:
        #print(f"'{word}'")
        if is_phoneme(word):
            # Process accumulated words and the current phoneme
            if current_phrase:
                _, phrase_tokens = pipeline.g2p(" ".join(current_phrase))
                processed_tokens.extend(phrase_tokens)
                processed_tokens.append(whitespace_token) 
                current_phrase = []
            # Add the phoneme directly
            token = en.MToken(text=word, phonemes=word, tag="", whitespace=" ")
            processed_tokens.append(token)
        else:
            # Accumulate non-phoneme words
            current_phrase.append(word)

    # Process any remaining words at the end
    if current_phrase:
        _, phrase_tokens = pipeline.g2p(" ".join(current_phrase))
        processed_tokens.extend(phrase_tokens)

    #for word in processed_tokens:
        #print(f"'{word}'")

    # Generate audio from processed tokens
    generator = pipeline.generate_from_tokens(
        tokens=processed_tokens,
        voice=voice,
        speed=speed
    )

    output_stream = BytesIO()
    voice_data = ""
    cumulative_time = 0.0
    token_index = 0
    SAMPLE_RATE = 24000
    MIN_DURATION = 1.5  # Minimum desired duration in seconds

    with wave.open(output_stream, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        has_audio = False
        for result in generator:
            audio = result.audio
            result_tokens = result.tokens

            if result_tokens:

                current_token_end = 0.0
                for token in result_tokens:
                    #print(token.text + token.whitespace, end='')
                    start_time = cumulative_time + (token.start_ts or 0.0)

                    adjustedIndex = token_index + (len(token.text) - len(token.text.lstrip(' ')))
                    
                    if start_time > 0:
                        voice_data += f"{start_time:.3f} {token.text.lstrip(' ')} {adjustedIndex}\n"
                    elif current_token_end > 0:
                        voice_data += f"{current_token_end:.3f} {token.text.lstrip(' ')} {adjustedIndex}\n"
                    
                    if token.end_ts is not None:
                        current_token_end = cumulative_time + token.end_ts
                    
                    #print(len(token.text), "|", len(token.whitespace), "|", f"|{token.text}|")
                    token_index = token_index + len(token.text) + len(token.whitespace)

            if audio is not None:
                has_audio = True
                audio_duration = len(audio) / 24000.0
                cumulative_time += audio_duration
                audio_int16 = np.array(audio * 32767, dtype=np.int16)
                wf.writeframes(audio_int16.tobytes())

        # After processing all audio, check if we need to add silence
        if not has_audio:
            # No audio at all - add 1.5 seconds of silence
            silence_samples = np.zeros(int(MIN_DURATION * SAMPLE_RATE), dtype=np.int16)
            wf.writeframes(silence_samples.tobytes())
            cumulative_time = MIN_DURATION
        elif cumulative_time < MIN_DURATION:
            # Audio is shorter than 1.5 seconds - add enough silence to reach 1.5 seconds
            silence_duration = MIN_DURATION - cumulative_time
            silence_samples = np.zeros(int(silence_duration * SAMPLE_RATE), dtype=np.int16)
            wf.writeframes(silence_samples.tobytes())
            cumulative_time = MIN_DURATION

    wf.close()  # Make sure to close the writer
    output_stream.seek(0)
    voice_data += "WORD BOUNDARY DATA WITH INDEX\n"

    return output_stream, voice_data, cumulative_time



def is_phoneme(token):
    phonemes = [
        "ʊ", "æ", "ɛ", "ɑ", "oʊ", "ɐ", "ɒ", "ɓ", "ʙ", "β", "ɔ", "ɕ", "ç", 
        "ɗ", "ɖ", "ð", "ʤ", "ə", "ɘ", "ɚ", "ɛ", "ɜ", "ɝ", "ɞ", "ɟ", "ʄ", "ɡ", 
        "ɠ", "ɢ", "ʛ", "ɦ", "ɧ", "ħ", "ɥ", "ʜ", "ɨ", "ɪ", "ʝ", "ɭ", "ɬ", "ɫ", 
        "ɮ", "ʟ", "ɱ", "ɯ", "ɰ", "ŋ", "ɳ", "ɲ", "ɴ", "ø", "ɵ", "ɸ", "θ", "œ", 
        "ɶ", "ʘ", "ɹ", "ɺ", "ɾ", "ɻ", "ʀ", "ʁ", "ɽ", "ʂ", "ʃ", "ʈ", "ʧ", "ʉ", 
        "ʊ", "ʋ", "ⱱ", "ʌ", "ɣ", "ɤ", "ʍ", "χ", "ʎ", "ʏ", "ʑ", "ʐ", "ʒ", "ʔ", 
        "ʡ", "ʕ", "ʢ", "ǀ", "ǁ", "ǂ", "ǃ", "ˈ", "ˌ", "ː", "ˑ", "ʼ", "ʴ", "ʰ", 
        "ʱ", "ʲ", "ʷ", "ˠ", "ˤ", "˞", "↓", "↑", "→", "↗", "↘", "'̩", "ᵻ"
    ]
    # Check if any phoneme is in the token
    return any(phoneme in token for phoneme in phonemes)



def create_mp3_response(audio_stream, sample_rate, voice_data):
    # Read the raw audio data from the BytesIO stream
    # Read the raw audio data from the BytesIO stream
    audio_stream.seek(0)
    raw_data = audio_stream.read()

    # Convert the raw bytes back to audio samples
    audio_samples = np.frombuffer(raw_data, dtype=np.int16)

    # Create a temporary WAV file in memory with the exact samples
    wav_stream = io.BytesIO()
    wav_write(wav_stream, sample_rate, audio_samples)
    wav_stream.seek(0)

    # Convert the raw WAV to an AudioSegment (ensuring original duration)
    audio = AudioSegment.from_file(wav_stream, format="wav")
    audio = audio.set_frame_rate(sample_rate)  # Keep original rate

    # Apply a fade-in effect WITHOUT changing length
    fade_duration = 100  # 100ms
    audio = audio.fade_in(fade_duration)

    # Convert to MP3 with minimal processing (matching duration)
    mp3_stream = io.BytesIO()
    audio.export(mp3_stream, format="mp3", bitrate="96k", parameters=["-ar", str(sample_rate)])
    mp3_stream.seek(0)

    # Decide whether to return the MP3 as base64 or as an attachment
    returnAsBase64 = False
    if not returnAsBase64:
        print("Returning audio/mp3 with boundaryData header")
        response = send_file(
            mp3_stream,
            mimetype="audio/mp3",
            download_name="audio.mp3"
        )
        safe_voice_data = base64.b64encode(voice_data.encode("utf-8")).decode("ascii")
        response.headers["boundaryData"] = safe_voice_data
        return response
    else:
        mp3_base64 = base64.b64encode(mp3_stream.read()).decode("utf-8")
        print("JSON")
        response_data = {
            "media": {
                "audio": {
                    "format": "mp3",
                    "data": {
                        "base64": mp3_base64
                    }
                }
            },
            "metadata": {
                "encoding": "base64",
                "size": "large"
            },
            "boundaryData": voice_data
        }
        return jsonify(response_data)

def benchmark(start_time, cumulative_time):
    #global global_time  # Declare global_time as a global variable

    end_time = time.time()
    elapsed_time = end_time - start_time
    print_time( cumulative_time, "", end=" " )
    print_time(elapsed_time, "Generated in: ")
    #global_time = global_time + elapsed_time
    #print_time(global_time, "Total time:")

def print_time(elapsed_time, message, end = "\n"):
    # Convert elapsed time to hours, minutes, seconds, and milliseconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    milliseconds = int((elapsed_time - int(elapsed_time)) * 1000)

    # Print the elapsed time in H M S ms format
    print(f"{message}{hours}h {minutes}m {seconds}s {milliseconds}ms.", end=end)  
    
def generate_response(audio_stream, voice_data):
    # Seek to the beginning of the audio stream
    audio_stream.seek(0)
    
    # Get the size of the audio stream
    wav_size = len(audio_stream.getvalue())
    
    # Create a BytesIO object with the preallocated size
    response_stream = io.BytesIO()
    response_stream.write(audio_stream.read())
    

    # Seek to the beginning of the response stream
    response_stream.seek(0)
    
    # Return the response as a file

    response = Response(response_stream.getvalue(), content_type="audio/wav")

    safe_voice_data = base64.b64encode(voice_data.encode("utf-8")).decode("ascii")
    response.headers["boundaryData"] = safe_voice_data
    return response

# Define the routes
@app.route('/')
def home():
    voices_list  = global_voices_list #["af_heart", "af_sky"]  # Use a Python list
    return render_template('index.html', voices=voices_list)


@app.route("/generate_audio", methods=["GET", "POST"])
def generate_and_stream_audio():

    torch.cuda.empty_cache()
    start_time = time.time()

    text = request.args.get('text', default="", type=str)
    voice = request.args.get('voice', default="af_sarah", type=str) or "af_sarah"
    speed = request.args.get('kokoro_speed', default=1.0, type=float) or 1.0
    boundary = request.args.get('boundary', default="false", type=str) 
    voice_hash = request.args.get('voice_hash', default=1, type=int)

    if request.method == "POST":
        data = request.get_json()
        text = data.get('input', '')
        voice = "af_sarah"
        speed = 1.0
        print( data )

    if voice_hash > 1:
        voice = get_voice_from_hash_key(voice_hash)
        print( voice_hash )
        print( voice )
    
    [audio_stream, voice_data, cumulative_time] = generate_audio(text, voice, speed)

    if request.args.get('format') == 'mp3':
        print("mp3")
        return create_mp3_response( audio_stream, 24000, voice_data )
    
    if boundary == "true":
        response_stream = generate_response(audio_stream, voice_data)
        benchmark( start_time, cumulative_time )
        return response_stream
    
    benchmark( start_time, cumulative_time )
    return send_file(audio_stream, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
