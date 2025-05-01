from whisper import load_model
import time

model = load_model("large-v3-turbo")

start = time.time()
audio = model.transcribe("audio_samples/magicumbrella_12_cory_128kb.mp3", beam_size=4)
end = time.time()

print(audio['text'])
print(f"Transcription time (CPU): {end - start:.2f} seconds") 