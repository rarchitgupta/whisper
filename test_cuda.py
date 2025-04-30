from whisper import load_model

model = load_model("large-v3-turbo")

audio = model.transcribe("audio_samples/OSR_us_000_0061_8k.wav", beam_size=2, beam_search_device="GPU")

print(audio['text'])