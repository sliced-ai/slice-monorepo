import torch
import torchaudio
import IPython
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice

# Define your own voice folder
VOICE_NAME = 'your_name_here'  # Replace 'your_name_here' with your name or desired folder name
text = 'Hello from this tutorial, I hope you enjoy it'

# Load the TTS model
tts = TextToSpeech()

# Load your voice samples
voice_samples, conditioning_latents = load_voice(VOICE_NAME)

# Generate speech with your own voice
gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents)

# Save the generated speech to a file
output_file = f'generated-{VOICE_NAME}.wav'
torchaudio.save(output_file, gen.squeeze(0).cpu(), 24000)

# Play the generated audio
IPython.display.Audio(output_file)
