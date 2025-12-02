import os
import base64
from pydub import AudioSegment
from pydub.utils import which
import speech_recognition as sr

# 🔥 Attach ffmpeg paths
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

AUDIO_FOLDER = os.path.join("static", "audio")
os.makedirs(AUDIO_FOLDER, exist_ok=True)


def get_next_audio_filename():
    existing = [
        f for f in os.listdir(AUDIO_FOLDER)
        if f.startswith("speech_complaint_") and f.endswith(".wav")
    ]

    if not existing:
        return "speech_complaint_1.wav"

    numbers = []
    for f in existing:
        try:
            num = int(f.replace("speech_complaint_", "").replace(".wav", ""))
            numbers.append(num)
        except:
            pass

    next_num = max(numbers) + 1 if numbers else 1
    return f"speech_complaint_{next_num}.wav"


def convert_voice_to_text(base64_audio):
    try:
        audio_bytes = base64.b64decode(base64_audio)
        temp_input = os.path.join(AUDIO_FOLDER, "temp_input.webm")

        with open(temp_input, "wb") as f:
            f.write(audio_bytes)

        wav_filename = get_next_audio_filename()
        wav_filepath = os.path.join(AUDIO_FOLDER, wav_filename)

        # Convert WebM → WAV
        audio = AudioSegment.from_file(temp_input, format="webm")
        audio.export(wav_filepath, format="wav")
        os.remove(temp_input)

        # SpeechRecognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_filepath) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        return {
            "success": True,
            "text": text,
            "filename": wav_filename
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error processing audio: {e}"
        }
