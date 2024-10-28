import os
import whisper

# Set up paths
AUDIO_DIR = "../../../data/audio"
TEXT_DIR = "../../../data/text"

# Ensure the output directory exists
os.makedirs(TEXT_DIR, exist_ok=True)

# Load the Whisper model
model = whisper.load_model("base")

def transcribe_audio_files():
    # Iterate over all files in the audio directory
    for filename in os.listdir(AUDIO_DIR):
        if filename.endswith(".mp3"):
            audio_path = os.path.join(AUDIO_DIR, filename)
            
            # Perform transcription
            print(f"Transcribing {filename}...")
            result = model.transcribe(audio_path)
            transcription = result["text"]
            
            # Save transcription to text file
            text_filename = os.path.splitext(filename)[0] + ".txt"
            text_path = os.path.join(TEXT_DIR, text_filename)
            with open(text_path, "w", encoding="utf-8") as text_file:
                text_file.write(transcription)
            
            print(f"Saved transcription to {text_path}")

if __name__ == "__main__":
    transcribe_audio_files()
