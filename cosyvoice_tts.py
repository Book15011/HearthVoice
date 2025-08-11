from transformers import AutoProcessor, AutoModel
import soundfile as sf
import torch

# 1. Load model
model_id = "FunAudioLLM/CosyVoice2-0.5B"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

# 2. Load your voice
reference_audio, ref_sr = sf.read("my_voice.wav")  # Must be 16kHz!

# 3. Your custom text
custom_text = "Hello, this is my voice speaking through AI!"  # <-- EDIT THIS

# 4. Process inputs
inputs = processor(
    text=custom_text,
    audio=reference_audio,
    sampling_rate=ref_sr,
    return_tensors="pt"
)

# 5. Generate speech
with torch.no_grad():
    output = model.generate(**inputs)

# 6. Save output
sf.write("output.wav", output["audio"][0].numpy(), output["sampling_rate"])
print("Done! Output saved as output.wav")