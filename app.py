import streamlit as st
from transformers import pipeline
import sounddevice as sd
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from keybert import KeyBERT

# Load the Whisper model and specify the device as GPU
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    asr_model = pipeline("automatic-speech-recognition", model="whisper-finetuned", device=0)
except Exception as e:
    st.error(f"Error loading Whisper model: {e}")

# Load the KeyBERT model for keyword extraction
kw_model = KeyBERT()

# Function to record audio
def record_audio(duration=5, samplerate=16000):
    try:
        st.write("Recording...")
        audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()
        st.write("Recording finished.")
        return audio.flatten()
    except Exception as e:
        st.error(f"Error recording audio: {e}")
        return None

# Function to transcribe audio using Whisper
def transcribe_audio(audio):
    try:
        result = asr_model(audio)
        return result['text']
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return ""

# Function to extract keywords
def extract_keywords(text, num_keywords=5):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=num_keywords)
    return ' '.join([keyword for keyword, score in keywords])

# Streamlit app
def main():
    st.title("Audio Transcription and Image Generation with Keyword Extraction")
    
    # Record audio
    if st.button("Record Audio"):
        duration = st.slider("Select recording duration in seconds", 1, 10, 5)
        audio = record_audio(duration)
        if audio is not None:
            text = transcribe_audio(audio)
            st.write(f"Recognized: {text}")
            
            # Extract keywords
            keywords = extract_keywords(text)
            st.write(f"Keywords: {keywords}")
            
            try:
                # Load the stable diffusion model and specify the device as GPU
                model_id = "CompVis/stable-diffusion-v1-4"
                pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
                
                # Generate the image
                with torch.autocast("cuda"):
                    image = pipe(keywords, guidance_scale=7.5).images[0]
                
                # Display the image
                st.image(image, caption="Generated Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error generating image: {e}")

if __name__ == "__main__":
    main()
