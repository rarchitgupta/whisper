from whisper import load_model
import time

# Pre-load CUDA extensions to exclude build/load time from benchmark
import cuda_beam_search.cuda_beam_search.beam_search_cuda
import cuda_beam_search.cuda_beam_search.logit_filter_cuda
import cuda_beam_search.cuda_beam_search.topk_beam_expansion_cuda

model = load_model("large-v3-turbo")

start = time.time()
audio = model.transcribe("audio_samples/magicumbrella_12_cory_128kb.mp3", beam_size=4, beam_search_device="GPU")
end = time.time()

print(audio['text'])
print(f"Transcription time (excluding module load): {end - start:.2f} seconds")