import os
import whisper

# Define paths for input and output directories
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
AUDIO_DIR = os.path.join(ROOT_DIR, 'data', 'audio')
TEXT_DIR = os.path.join(ROOT_DIR, 'data', 'text')

# Ensure the output directory exists
os.makedirs(TEXT_DIR, exist_ok=True)

# Load the Whisper model
model = whisper.load_model("base")  # Use "small", "medium", or "large" as needed

def transcribe_audio_files():
    """Transcribes all wav files in the AUDIO_DIR and saves them in the TEXT_DIR."""
    for filename in os.listdir(AUDIO_DIR):
        if filename.endswith(".wav"):
            audio_path = os.path.join(AUDIO_DIR, filename)
            
            # Perform transcription
            print(f"Transcribing {filename}...")
            result = model.transcribe(audio_path)
            transcription = result["text"]
            
            # Save transcription to a .txt file with the same name in TEXT_DIR
            text_filename = os.path.splitext(filename)[0] + ".txt"
            text_path = os.path.join(TEXT_DIR, text_filename)
            with open(text_path, "w", encoding="utf-8") as text_file:
                text_file.write(transcription)
            
            print(f"Saved transcription to {text_path}")

if __name__ == "__main__":
    transcribe_audio_files()
