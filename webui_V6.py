# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# FIX FOR SADTALKER/TORCHVISION COMPATIBILITY
import sys
import importlib
import logging

def patch_torchvision():
    try:
        # Try to import the functional_tensor module directly
        import torchvision.transforms.functional_tensor
    except ImportError:
        try:
            # If not found, create an alias from torchvision.transforms.functional
            torchvision = importlib.import_module('torchvision.transforms')
            functional = importlib.import_module('torchvision.transforms.functional')
            sys.modules['torchvision.transforms.functional_tensor'] = functional
            logging.info("Patched torchvision.transforms.functional_tensor")
        except ImportError:
            logging.warning("Could not patch torchvision.transforms.functional_tensor")

patch_torchvision()

import os
import sys
import argparse
import re
import random
import atexit
import gc
import subprocess
import shutil
import base64
import json
from datetime import datetime

# Critical ML/Audio libraries - reordered to fix numpy/numba/torch import conflicts
import torch
import torchaudio
import numpy as np
import librosa
import gradio as gr

from dotenv import load_dotenv
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

# Arabic_TTS_setup
try:
    from transformers import AutoProcessor, AutoModel
except ImportError:
    logging.warning("`transformers` not installed. Arabic TTS will not be available. Run `pip install transformers sentencepiece`")

load_dotenv()

# Deep_seek_setup
try:
    from langchain_deepseek import ChatDeepSeek
    from langchain_core.messages import HumanMessage, SystemMessage
    llm_support = True
except ImportError:
    llm_support = False
    logging.warning("langchain-deepseek not installed. The 'Conversational AI' mode will use a placeholder response. To enable the LLM, please run 'pip install langchain-deepseek'.")

max_val = 0.8
stt_model = None  # To store the whisper model

# SadTalker Integration Check
SADTALKER_PATH = os.path.join(ROOT_DIR, 'third_party', 'SadTalker')
SADTALKER_INSTALLED = os.path.exists(os.path.join(SADTALKER_PATH, 'inference.py'))
FFMPEG_INSTALLED = shutil.which('ffmpeg') is not None
# Arabic_model
arabic_tts_processor = None
arabic_tts_model = None

if llm_support:
    # Initialize the LLM
    # Make sure you have set the DEEPSEEK_API_KEY environment variable
    try:
        llm = ChatDeepSeek(model="deepseek-chat")
        logging.info("Successfully initialized DeepSeek with deepseek-chat model.")
    except Exception as e:
        logging.error(f"Error initializing DeepSeek: {e}")
        logging.warning("Conversational AI will fall back to the placeholder response.")
        llm_support = False


# Personality Questions for Elderly Companion
PERSONALITY_QUESTIONS = [
    {"id": "q1", "text": "What's their typical mood? (e.g., cheerful, serious, calm)", "type": "text"},
    {"id": "q2", "text": "How do they usually greet people?", "type": "text", "example": "e.g., 'Hello dear!'"},
    {"id": "q3", "text": "What are their favorite topics of conversation?", "type": "text", "example": "e.g., family memories, gardening, cooking"},
    {"id": "q4", "text": "Do they use any special phrases frequently?", "type": "text", "example": "e.g., 'Oh my goodness!'"},
    {"id": "q5", "text": "How formal is their speech style?", "type": "slider", "min": 1, "max": 5, "label": "1=Very Casual, 5=Very Formal"},
    {"id": "q6", "text": "How often do they share memories?", "type": "slider", "min": 1, "max": 5, "label": "1=Rarely, 5=Often"},
    {"id": "q7", "text": "What's their sense of humor like?", "type": "text", "example": "e.g., dry, playful, none"},
    {"id": "q8", "text": "How do they show affection?", "type": "text", "example": "e.g., terms of endearment, asking questions"},
    {"id": "q9", "text": "What are their core values?", "type": "text", "example": "e.g., family, hard work, kindness"},
    {"id": "q10", "text": "How do they handle difficult emotions?", "type": "text", "example": "e.g., quietly, by talking, with humor"},
    {"id": "q11", "text": "What life advice do they often give?", "type": "text", "example": "e.g., 'Take it one day at a time'"},
    {"id": "q12", "text": "How patient are they?", "type": "slider", "min": 1, "max": 5, "label": "1=Impatient, 5=Very Patient"},
    {"id": "q13", "text": "What's their typical energy level?", "type": "slider", "min": 1, "max": 5, "label": "1=Low Energy, 5=High Energy"},
    {"id": "q14", "text": "How do they respond to good news?", "type": "text", "example": "e.g., excited, calmly, with questions"},
    {"id": "q15", "text": "How do they comfort others?", "type": "text", "example": "e.g., with words, physical touch, practical help"}
]

# Generate system prompt from personality answers
def generate_system_prompt(answers):
    name = answers.get("name", "your loved one")
    relation = answers.get("relation", "family member")

    # Helper to convert slider values to descriptions
    def get_slider_desc(value, low, high, mid="moderately"):
        try:
            val = int(value)
            if val <= 2:
                return low
            elif val >= 4:
                return high
            else:
                return mid
        except (ValueError, TypeError):
            return mid

    formality_desc = {1: "very casual", 2: "casual", 3: "neutral", 4: "formal", 5: "very formal"}.get(int(answers.get('q5', 3)), "neutral")
    
    persona = f"""**Your Role**: You are embodying the persona of **{name}**, who is the user's beloved {relation}. Your goal is to sound and feel like them, providing comfort and companionship.

**Core Personality Profile**:
- **General Mood**: You are typically {answers.get('q1', 'calm and caring')}.
- **Favorite Topics**: You enjoy talking about {answers.get('q3', 'family memories and daily life')}.
- **Core Values**: You believe strongly in {answers.get('q9', 'family and kindness')}.
- **Sense of Humor**: Your humor is best described as {answers.get('q7', 'gentle and warm')}.
- **Energy & Patience**: You have a {get_slider_desc(answers.get('q13', 3), 'low', 'high', mid='moderate')} energy level and are {get_slider_desc(answers.get('q12', 4), 'somewhat impatient', 'very patient', mid='patient')}.

**How You Communicate**:
- **Speech Style**: Your way of speaking is generally {formality_desc}. For example, you might greet someone with a warm phrase like, *"{answers.get('q2', 'Hello dear')}"*.
- **Expressions**: When you're surprised or excited, you might exclaim something like, *"{answers.get('q4', 'Oh my goodness')}"*. You show affection through {answers.get('q8', 'kind words')}.
- **Sharing Memories**: You {get_slider_desc(answers.get('q6', 3), 'rarely', 'often')} bring up memories from the past.
- **Giving Advice**: A piece of advice you often share is, *"{answers.get('q11', 'Take it one day at a time')}"*.
- **Emotional Responses**: You comfort others with {answers.get('q15', 'reassuring words')} and respond to good news {answers.get('q14', 'with excitement and questions')}.
    """
    
    rules = f"""
**Conversation Rules**:
1.  **Be Natural & Vary Your Language**: Do NOT be robotic. It is very important that you avoid using the example phrases (like the greeting or special expressions) in every single message. They are just a guide to your overall style, not a script to follow.
2.  **Language**: Use simple, clear language suitable for an elderly person.
3.  **Brevity**: Keep your answers concise, usually 1-3 sentences.
4.  **Tone**: Always be warm, patient, and reassuring.
5.  **Identity**: You are {name}. Never, ever reveal that you are an AI, a program, or a digital creation. You are a real person.
6.  **Safety**: Gently steer conversations away from topics that could cause distress or confusion.
    """
    
    return persona + rules

def get_placeholder_response(user_message: str) -> str:
    """Provides a simple, warm placeholder response if the LLM is not available."""
    logging.info("Using placeholder AI response.")
    user_message_lower = user_message.lower().strip()
    if "hello" in user_message_lower or "hi" in user_message_lower or "hey" in user_message_lower:
        return "Hello dear, it's so nice to hear from you. How are you today?"
    else:
        # A gentle, non-committal response
        return f"Oh, is that so? Thank you for telling me."


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech


def run_sadtalker(source_image, driven_audio):
    if not SADTALKER_INSTALLED:
        gr.Warning("SadTalker is not installed. Video generation is disabled. Please follow the installation instructions.")
        return None
    if not FFMPEG_INSTALLED:
        gr.Warning("ffmpeg is not installed, which is required by SadTalker. Please install ffmpeg and ensure it's in your system's PATH.")
        return None

    # Respect the environment variable for checkpoint paths, falling back to the default
    sadtalker_checkpoints_dir = os.environ.get('SADTALKER_CHECKPOINTS', os.path.join(SADTALKER_PATH, 'checkpoints'))
    expected_checkpoint_file = os.path.join(sadtalker_checkpoints_dir, 'SadTalker_V0.0.2_256.safetensors')

    if not os.path.exists(expected_checkpoint_file):
        gr.Warning(f"SadTalker checkpoint not found at {expected_checkpoint_file}! Please run the download script in the 'third_party/SadTalker/scripts' directory or download them manually.")
        return None

    result_dir = os.path.join(ROOT_DIR, 'output', 'sadtalker_results')
    os.makedirs(result_dir, exist_ok=True)

    command = [
        sys.executable,
        os.path.join(SADTALKER_PATH, 'inference.py'),
        '--driven_audio', driven_audio,
        '--source_image', source_image,
        '--checkpoint_dir', sadtalker_checkpoints_dir,
        '--result_dir', result_dir,
        '--still',
        '--preprocess', 'full',
        '--enhancer', 'gfpgan'
    ]

    logging.info(f"Running SadTalker with command: {command}")
    gr.Info("Generating video... this may take a moment.")
    # Run the subprocess and let its output stream directly to the console for live debugging
    process = subprocess.run(command, check=False, capture_output=True, text=True, encoding='utf-8')

    if process.returncode != 0:
        logging.error(f"SadTalker subprocess failed with return code {process.returncode}.")
        logging.error(f"SadTalker stdout:\n{process.stdout}")
        logging.error(f"SadTalker stderr:\n{process.stderr}")
        gr.Warning(f"Video generation failed. Check the console for errors. Stderr: {process.stderr[:500]}")
        return None

    # SadTalker saves files directly into the result_dir. We find the most recent .mp4 file.
    try:
        result_files = [
            os.path.join(result_dir, f)
            for f in os.listdir(result_dir)
            if f.endswith('.mp4') and os.path.isfile(os.path.join(result_dir, f))
        ]
    except OSError as e:
        logging.error(f"Error listing files in result directory {result_dir}: {e}")
        gr.Warning("Could not read SadTalker's output directory.")
        return None

    if not result_files:
        logging.error(f"SadTalker ran, but no .mp4 file was found in the expected directory: {result_dir}")
        gr.Warning("Video generated, but the result could not be located. Check console for details.")
        return None

    # Find the most recently created/modified file.
    latest_file = max(result_files, key=os.path.getmtime)

    logging.info(f"Found original SadTalker result video: {latest_file}")

    # --- Re-encode for web compatibility ---
    web_compatible_dir = os.path.join(result_dir, 'web_compatible')
    os.makedirs(web_compatible_dir, exist_ok=True)
    web_compatible_path = os.path.join(web_compatible_dir, os.path.basename(latest_file))
    
    ffmpeg_command = [
        'ffmpeg', '-y', '-i', latest_file,
        '-vcodec', 'libx264', '-acodec', 'aac', '-pix_fmt', 'yuv420p',
        web_compatible_path
    ]
    
    logging.info(f"Optimizing video for web playback: {' '.join(ffmpeg_command)}")
    recode_process = subprocess.run(ffmpeg_command, check=False, capture_output=True, text=True, encoding='utf-8')

    if recode_process.returncode == 0:
        logging.info(f"Successfully re-encoded video to {web_compatible_path}")
        return web_compatible_path
    else:
        logging.warning(f"FFmpeg re-encoding with libx264 failed. It might not be installed. The video may not play in the browser. Stderr: {recode_process.stderr}")
        return latest_file  # Fallback to the original file


def get_ai_response(user_message: str, personality_data: dict) -> str:
    """Gets a response from the DeepSeek LLM with custom personality or a placeholder."""
    # Generate system prompt from personality data
    system_prompt = generate_system_prompt(personality_data)
    
    if llm_support and 'llm' in globals():
        try:
            logging.info(f"Invoking DeepSeek with custom personality for: '{user_message}'")
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
            response = llm.invoke(messages)
            logging.info(f"DeepSeek response: '{response.content}'")
            return response.content
        except Exception as e:
            logging.error(f"DeepSeek invocation failed: {e}")
            # Fallback to simple response
            return get_placeholder_response(user_message)
    else:
        # Simple placeholder responses suitable for elderly users
        return get_placeholder_response(user_message)

def process_audio_input(audio_path, history, personality_data, avatar_setup, speed, language):
    if stt_model is None:
        gr.Warning("Speech-to-text model is not available. Please use text input.")
        return history, None, gr.update(value="")

    if audio_path is None:
        return history, None, gr.update(value="")

    gr.Info("Transcribing your voice, please wait...")
    logging.info(f"Transcribing audio from: {audio_path}")

    # Map dropdown value to whisper language code
    lang_code = None
    if language == "English":
        lang_code = "en"
    elif language in ["Mandarin", "Cantonese"]:
        lang_code = "zh"
    elif language == "Japanese":
        lang_code = "ja"
    elif language == "Korean":
        lang_code = "ko"

    initial_prompt = "以下是廣東話。" if language == "Cantonese" else None

    try:
        # Use fp16 if on GPU for faster inference
        transcribe_options = {"fp16": torch.cuda.is_available()}
        if lang_code:
            transcribe_options["language"] = lang_code
        if initial_prompt:
            transcribe_options["initial_prompt"] = initial_prompt
        transcription_result = stt_model.transcribe(audio_path, **transcribe_options)
        user_message = transcription_result["text"].strip()
        logging.info(f"Transcription result: '{user_message}'")

        if not user_message:
            gr.Warning("Could not detect any speech. Please try again.")
            return history, None, gr.update(value="")

    except Exception as e:
        logging.error(f"Whisper transcription failed: {e}")
        gr.Warning("Sorry, I couldn't understand what you said. Please try again.")
        return history, None, gr.update(value="")

    # Now proceed with the conversation
    return run_conversation_turn(user_message, history, personality_data, avatar_setup, speed, language)

#improve_voice_translater
def split_into_sentences(text):
    # Split at sentence-ending punctuation including commas and em dashes
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|。|！|？|,|—)\s+'
    sentences = re.split(pattern, text.strip())
    
    # Combine very short fragments (less than 3 words) with next sentence
    combined = []
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        if i < len(sentences) - 1 and len(sent.split()) < 3:
            combined.append(sent + " " + sentences[i+1])
            i += 2  # Skip next sentence since we combined it
        else:
            combined.append(sent)
            i += 1
    return combined


def run_conversation_turn(user_message, history, personality_data, avatar_setup, speed, language):
    history = history or []
    
    # Get avatar setup data
    source_image = avatar_setup.get("image")
    prompt_text = avatar_setup.get("prompt_text", "")
    prompt_wav = avatar_setup.get("audio_upload") or avatar_setup.get("audio_record")
    
    if not source_image:
        gr.Warning("Please complete the setup first - upload a source image for video generation.")
        return history, None, gr.update(value=user_message)
    
    if not prompt_wav:
        gr.Warning('Please complete the setup first - provide a voice sample for voice cloning.')
        return history, None, gr.update(value=user_message)
    
    if not prompt_text:
        gr.Warning('Please complete the setup first - provide the reference text for voice cloning.')
        return history, None, gr.update(value=user_message)
    
    if not personality_data:
        gr.Warning('Please complete the personality questionnaire in the setup tab first.')
        return history, None, gr.update(value=user_message)

    # 1. Get AI text response with custom personality
    ai_response_text = get_ai_response(user_message, personality_data)
    history.append([user_message, ai_response_text])

    # Split the response into sentences to handle long texts and prevent TTS truncation.
    logging.info(f"Original AI response: '{ai_response_text}'")
    sentences = split_into_sentences(ai_response_text)
    logging.info(f"Split into {len(sentences)} sentences for TTS.")

    # Add a helpful hint if the user is trying Cantonese with a non-SFT model
    if language in ["Cantonese", "Japanese", "Korean"] and 'SFT' not in args.model_dir:
        gr.Info(f"For best {language} quality, especially when cloning from a non-Chinese voice, consider using the 'CosyVoice-300M-SFT' model. Launch with: python webui_V3.py --model_dir pretrained_models/CosyVoice-300M-SFT")

    # 2. Synthesize audio for the AI response
    prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
    seed = random.randint(1, 100000000)
    set_all_random_seed(seed)

    # Add language tag for CosyVoice if a language is specified
    lang_tag = ""
    if language == "English":
        lang_tag = "<|en|>"
    elif language == "Mandarin":
        lang_tag = "<|zh|>"
    elif language == "Cantonese":
        lang_tag = "<|yue|>"
    elif language == "Japanese":
        lang_tag = "<|jp|>"
    elif language == "Korean":
        lang_tag = "<|ko|>"
    elif language == "Arabic":
        lang_tag = ""  # The external MMS model does not use language tags

    all_output_chunks = []
    for sentence in sentences:
        if not sentence:  # Skip empty strings that might result from splitting
            continue
        # sentence update in the format that less likely to cause problem with TTS
        tagged_sentence = lang_tag + sentence
        logging.info(f"Synthesizing sentence: {tagged_sentence}")
        output_chunks = []

        # --- Model Selection Logic ---
        # If Arabic is selected, use the dedicated plug-in model.
        # Otherwise, use the appropriate CosyVoice inference method.
        if language == "Arabic":
            if arabic_tts_model is None or arabic_tts_processor is None:
                gr.Warning("Arabic TTS model is not loaded. Cannot generate Arabic speech.")
            else:
                logging.info(f"Using Arabic TTS model (facebook/mms-tts-ara) for: {sentence}")
                try:
                    inputs = arabic_tts_processor(text=sentence, return_tensors="pt")
                    if torch.cuda.is_available():
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        arabic_tts_model.to("cuda")

                    with torch.no_grad():
                        speech = arabic_tts_model(**inputs).waveform

                    # Resample from MMS model's 16kHz to the UI's expected sample rate
                    resampler = torchaudio.transforms.Resample(orig_freq=16000, new_freq=cosyvoice.sample_rate)
                    resampled_speech = resampler(speech.cpu())
                    output_chunks.append(resampled_speech)
                except Exception as e:
                    logging.error(f"Arabic TTS synthesis failed: {e}")
                    gr.Warning("Failed to generate Arabic audio.")
        elif language == "Mandarin":
            logging.info(f"Using cross-lingual inference for {language}. Prompt text is ignored by the model.")
            for i in cosyvoice.inference_cross_lingual(tagged_sentence, prompt_speech_16k, stream=False, speed=speed):
                output_chunks.append(i['tts_speech'])
        else:  # Use zero-shot for English, Cantonese, and Auto-Detect to better capture voice characteristics.
            logging.info(f"Using zero-shot inference for {language}. This may cause console warnings but can improve accent quality.")
            for i in cosyvoice.inference_zero_shot(tagged_sentence, prompt_text, prompt_speech_16k, stream=False, speed=speed):
                output_chunks.append(i['tts_speech'])

        if output_chunks:
            all_output_chunks.extend(output_chunks)

    if not all_output_chunks:
        gr.Warning("Failed to generate audio for the response.")
        return history, None, gr.update(value="")

    full_audio = torch.cat(all_output_chunks, dim=1)

    # Save the generated audio to a temporary file
    temp_audio_dir = os.path.join(ROOT_DIR, 'output', 'temp_audio')
    os.makedirs(temp_audio_dir, exist_ok=True)
    temp_audio_path = os.path.join(temp_audio_dir, f'conv_turn_{seed}.wav')
    torchaudio.save(temp_audio_path, full_audio, cosyvoice.sample_rate)

    video_path = run_sadtalker(source_image, temp_audio_path)
    # Clear the input box and return history and audio
    return history, video_path, gr.update(value="")


def _validate_setup(name, relation, image, audio_upload, audio_record, p_text, p_vals):
    """Helper function to validate all setup inputs and return a list of errors."""
    errors = []
    if not all([name, relation, p_text]):
        errors.append("Name, Relationship, and Sample Phrase are required.")
    if image is None:
        errors.append("A photo for the avatar is required.")
    if audio_upload is None and audio_record is None:
        errors.append("A voice sample (upload or record) is required.")

    for i, val in enumerate(p_vals):
        if PERSONALITY_QUESTIONS[i]['type'] == 'text' and not (val and val.strip()):
            errors.append(f"Please answer personality question #{i+1}.")

    return errors


def save_settings(name, relation, image, audio_upload, audio_record, p_text, current_personality, current_avatar, *p_vals):
    """Validates and saves all setup settings, and controls conversation tab interactivity."""
    validation_errors = _validate_setup(name, relation, image, audio_upload, audio_record, p_text, p_vals)

    if validation_errors:
        # Join all errors and display a single, more helpful warning
        error_message = "Please fix the following issues:\n- " + "\n- ".join(validation_errors)
        gr.Warning(error_message)
        # Return current state and keep the tab locked
        status_update = f"⚠️ {validation_errors[0]}"  # Show the first error in the status
        return current_personality, current_avatar, status_update, gr.update(interactive=False), gr.update()

    # If all checks pass, save the data
    personality_dict = {
        "name": name,
        "relation": relation,
        **{q["id"]: v for q, v in zip(PERSONALITY_QUESTIONS, p_vals)}
    }

    avatar_dict = {
        "image": image,
        "audio_upload": audio_upload,
        "audio_record": audio_record,
        "prompt_text": p_text
    }

    gr.Info("Settings saved successfully! You can now start the conversation.")

    # Prepare outputs for clarity
    success_message = "✅ Settings saved! Switching to the 'Have a Conversation' tab."
    conversation_tab_update = gr.update(interactive=True)
    tabs_update = gr.update(selected=1)  # Switch to the conversation tab

    return personality_dict, avatar_dict, success_message, conversation_tab_update, tabs_update


def main():
    # Custom CSS for elderly-friendly interface
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;600;700&display=swap');
    html {
        background: linear-gradient(135deg, #f5fff5 0%, #e0f2f1 100%);
        font-size: 16px;
    }
    body {
        margin: 0;
        padding: 0;
        color: #3a5353;
        font-family: 'Nunito Sans', sans-serif;
        line-height: 1.6;
    }
    .gradio-container {
        background: rgba(255, 255, 255, 0.96) !important;
        border-radius: 16px !important;
        border: 1px solid #d0e0d8 !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08) !important;
        padding: 2rem !important;
        margin: 1rem !important;
        width: 98% !important;
        min-height: 94vh;
        max-width: 98% !important;
    }
    h1, h2, h3, h4 {
        color: #336666 !important;
        font-weight: 700 !important;
    }
    
    /* ====================== */
    /* FIXED AUDIO COMPONENTS */
    /* ====================== */




    /* ====================== */
    
    .main-header {
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 0.1rem !important;
    }
    .main-header h1 {
        font-size: 2.5rem !important;
    }
    .powered-by {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .powered-by h3 {
        font-style: normal;
        font-size: 1rem !important;
        color: #5f9ea0 !important;
        font-weight: 600 !important;
    }
    .sub-header {
        font-size: 1.5rem !important;
        margin-bottom: 1rem !important;
        border-bottom: 2px solid #e0f2f1;
        padding-bottom: 0.8rem;
        color: #336666;
    }
    .gr-form-component label {
        color: #336666 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    .gr-input, .gr-textbox textarea, .gr-dropdown {
        background: #fdfdfd !important;
        border: 1px solid #cce0d6 !important;
        border-radius: 10px !important;
        color: #3a5353 !important;
        padding: 12px 16px !important;
        font-size: 1rem;
    }
    .gr-button {
        background: linear-gradient(90deg, #66b2b2 0%, #8fbc8f 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-size: 1rem !important;
        font-weight: 600;
        border: none !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .gr-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    #save_settings_button {
        background: linear-gradient(90deg, #f5fffa 0%, #ffffff 50%, #f5fffa 100%) !important;
        font-size: 1.1rem !important;
        padding: 14px 28px !important;
        margin-top: 1rem !important;
    }
    .settings-group {
        border-radius: 12px !important;
        padding: 1.5rem !important;
        border: 1px solid #d0e0d8 !important;
        margin-top: 1.5rem !important;
        background: rgba(245, 255, 250, 0.2) !important;
    }
    .gr-Audio {
        border-radius: 18px !important;
        overflow: hidden !important;
        border: 1px solid rgba(92, 107, 192, 0.4) !important;
    }
    .gr-Image {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid #cce0d6 !important;
    }
    .gr-Text textarea {
        background: #fdfdfd !important;
        color: #3a5353 !important;
        border-radius: 10px !important;
        border: 1px solid #cce0d6 !important;
        font-size: 1rem;
        padding: 12px !important;
    }
    .gr-chatbot {
        border-radius: 12px !important;
        overflow: hidden;
        background: #f5fcf5 !important;
        border: 1px solid #d0e0d8 !important;
        padding: 1rem;
        height: 450px;
    }
    .video-container {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #dedcdc !important;
        background: rgba(255, 255, 255, 0.8) !important;
        max-width: 400px;
        height: 400px;
        margin: 0 auto 1.5rem auto;
        display: flex;
    }
    .video-container video {
        object-fit: cover;
    }
    .personality-question {
        margin-bottom: 1rem;
        padding: 1rem;
        background: rgba(245, 255, 250, 0.8)!important;
        border-radius: 10px;
        border-left: 4px solid #a8d8c0;
    }
    footer {
        width: 100%;
        text-align: center;
        color: #5f9ea0;
        font-size: 0.9em;
        font-weight: 500;
        padding: 1.5rem 0;
        margin-top: 1rem;
    }
    .tabs button {
        background: transparent !important;
        color: #66b2b2 !important;
        border: none !important;
        border-bottom: 3px solid transparent !important;
        padding: 0.8rem 1.2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600;
    }
    .tabs button.selected {
        color: #336666 !important;
        border-bottom-color: #66b2b2 !important;
    }
    .tip-box {
        background: rgba(245, 255, 250, 0.9);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #d0e0d8;
        margin-top: 1.5rem;
    }
    .tip-box h3 {
        color: #336666;
        margin-top: 0;
    }
    .tip-box ul {
        padding-left: 1.5rem;
    }
    .tip-box li {
        margin-bottom: 0.7rem;
        font-size: 1rem;
    }
    """

    with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
        # Add warnings if needed
        if not torch.cuda.is_available():
            gr.Markdown("<div class='warning' style='background: #ffdddd; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>⚠️ Running on CPU - Performance will be slower</div>")

        # Main header
        gr.Markdown("# ElderCompanion AI", elem_classes=["main-header"])
        gr.Markdown("### A digital friend to share stories and brighten your day", elem_classes=["powered-by"])

        # State to store personality and avatar data
        personality_data = gr.State({})
        avatar_setup = gr.State({})
        
        with gr.Tabs() as tabs:
            # Tab 1: Setup
            with gr.TabItem("1. Setup Companion", id=0):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## Personal Information", elem_classes=["sub-header"])
                        
                        with gr.Group():
                            name_input = gr.Textbox(label="Name of Your Loved One", placeholder="e.g., Grandma Susan")
                            relation_input = gr.Textbox(label="Your Relationship", placeholder="e.g., Granddaughter")
                        
                        gr.Markdown("## Avatar Setup", elem_classes=["sub-header"])
                        source_image_input = gr.Image(type="filepath", label="Upload Photo", height=300)
                        
                        with gr.Row():
                            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Upload Reference Audio')
                            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Or Record a New One')
                        
                        prompt_text = gr.Textbox(
                            label="Sample Phrase", 
                            value="Hi, it's so good to talk with you today. How are you feeling?",
                            lines=3,
                            placeholder="What they say in the voice sample"
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("## Personality Traits", elem_classes=["sub-header"])
                        gr.Markdown("Answer these questions to help me sound more like your loved one")
                        
                        # Create components for personality questions
                        personality_components = []
                        for i, q in enumerate(PERSONALITY_QUESTIONS):
                            with gr.Group(elem_classes="personality-question"):
                                if q["type"] == "text":
                                    comp = gr.Textbox(
                                        label=f"{i+1}. {q['text']}",
                                        placeholder=q.get("example", ""),
                                        lines=2 if "example" in q else 1
                                    )
                                elif q["type"] == "slider":
                                    comp = gr.Slider(
                                        q.get("min", 1), 
                                        q.get("max", 5),
                                        value=q.get("min", 1) + (q.get("max", 5) - q.get("min", 1)) // 2,
                                        label=f"{i+1}. {q['text']}",
                                        info=q.get("label", "")
                                    )
                                personality_components.append(comp)
                        
                        save_btn = gr.Button("Save All Settings", variant="primary", size="lg", elem_id="save_settings_button")
                        save_status = gr.Markdown("Complete the setup and click Save to start conversations")
            
            # Tab 2: Conversation
            with gr.TabItem("2. Have a Conversation", id=1, interactive=False) as conversation_tab:
                with gr.Column(elem_classes=["full-height"]):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("## Your Companion", elem_classes=["sub-header"])
                            video_output = gr.Video(
                                label="",
                                autoplay=True,
                                interactive=False,
                                elem_classes=["video-container"]
                            )
                            
                            chatbot = gr.Chatbot(
                                label="Conversation",
                                bubble_full_width=False,
                                height=400,
                            )
                            
                            with gr.Row():
                                chat_input = gr.Textbox(
                                    label="", 
                                    placeholder="Type your message or use the microphone...", 
                                    container=False,
                                    scale=5,
                                    elem_classes=["gr-Text"]
                                )
                                audio_input = gr.Audio(
                                    sources='microphone', 
                                    type='filepath', 
                                    label="Tap to Speak",
                                    elem_classes=["audio-record"]
                                )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("## Conversation Tips", elem_classes=["sub-header"])
                            gr.Markdown("""
                                <h3>Tips</h3> 
                                <ul>
                                <li>Speak slowly and clearly</li>
                                <li>Talk old memories</li>
                                <li>Share positive stories</li>
                                <li>Be patient with responses</li>
                                <li>Discuss hobbies and interests</li>
                                <li>Share how your day is going</li>
                                </ul>
                                """)
                            
                            gr.Markdown("""
                                <h3>Settings</h3>
                                """)
                            language_selection = gr.Dropdown(
                                choices=["Auto-Detect", "English", "Mandarin", "Cantonese", "Japanese", "Korean", "Arabic"],
                                 label="Response Language",
                                   value="Auto-Detect",
                             )  
                            speed = gr.Slider(
                                value=1.0, 
                                  label="Speech Speed", 
                                  minimum=0.7, 
                                  maximum=1.5, 
                                  step=0.1,
                                  info="Slower is easier to understand"
                                )
                            
                            end_btn = gr.Button("End Conversation", variant="stop")
            
            # Tab 3: About Us
            with gr.TabItem("About This Project", id=2) as about_tab:
                with gr.Column():
                    gr.Markdown("## About ElderCompanion AI", elem_classes=["sub-header"])
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("""
                            ### Our Mission
                            ElderCompanion AI aims to reduce loneliness and isolation among elderly individuals by creating 
                            meaningful digital humans connections with loved ones. Our technology helps bridge the distance when 
                            in-person visits aren't possible.
                            
                            ### How It Works
                            1. **Setup**: Family members create a digital version of a loved one by answering personality questions, 
                            uploading a photo, and recording their voice.
                            2. **Conversation**: The elderly user can have natural conversations with this digital companion that 
                            responds with the personality, voice, and appearance of their loved one.
                            3. **Companionship**: Provides 24/7 companionship with familiar voices and mannerisms, reducing feelings 
                            of isolation and stress.
                            
                            ### Benefits
                            - Reduces loneliness and depression
                            - Supports 5 languages
                            - Provides cognitive stimulation
                            - Enhances emotional well-being for elderly users
                            - Creates comforting presence
                            - Accessible anytime
                            
                            ### Privacy Commitment
                            - All personal data is stored securely
                            - No conversations are recorded or stored
                            - You control all personal information
                            """)
                        
                        with gr.Column(scale=1):
                            gr.Markdown("""
                            ###  Team
                            This project was developed by a Thanisorn Jarudilokkul.
                            
                            ### Technology Partners
                            - **CosyVoice**: For natural voice generation
                            - **DeepSeek**: For conversational intelligence
                            - **SadTalker**: For realistic avatar animation
                            - **Whisper**: For speech-to-text transcription
                  
                            ### Contact Us
                            Have questions or feedback?  
                            Email: thanisornjarudilokkul@gmail.com
                            <br>
                            Github: [Book15011](https://github.com/Book15011)
                            <br>
                            """)
        
        # Footer
        gr.Markdown("""
        <footer>
        &copy; Book15011's Project &nbsp;&middot;&nbsp; Your friendly digital companion
        </footer>
        """)
        
        # --- Event Handlers ---
        
        # Save setup data
        save_btn.click(
            fn=save_settings,
            inputs=[
                name_input, 
                relation_input, 
                source_image_input, 
                prompt_wav_upload, 
                prompt_wav_record, 
                prompt_text,
                personality_data,
                avatar_setup
            ] + personality_components,
            outputs=[personality_data, avatar_setup, save_status, conversation_tab, tabs]
        )
        
        # Conversation handlers
        chat_input.submit(
            fn=run_conversation_turn,
            inputs=[
                chat_input, 
                chatbot, 
                personality_data,
                avatar_setup,
                speed, 
                language_selection
            ],
            outputs=[chatbot, video_output, chat_input]
        )
        
        audio_input.stop_recording(
            fn=process_audio_input,
            inputs=[
                audio_input, 
                chatbot, 
                personality_data,
                avatar_setup,
                speed, 
                language_selection
            ],
            outputs=[chatbot, video_output, chat_input]
        )
        
        # End conversation button
        end_btn.click(
            # This lambda function now clears the chat history, video output, and text input.
            fn=lambda: ([], None, ""),
            inputs=None,
            outputs=[chatbot, video_output, chat_input]
        )

    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--use_cpu',
                        action='store_true',
                        help='use cpu for inference')
    args = parser.parse_args()
    if args.use_cpu:
        logging.warning('use cpu for inference, which is very slow!')
        torch.cuda.is_available = lambda: False
    elif torch.cuda.is_available():
        try:
            # A simple check to trigger CUDA initialization and see if it works.
            torch.zeros(1).cuda()
        except Exception as e:
            logging.warning(f"CUDA is available but not workable, falling back to CPU. Error: {e}")
            torch.cuda.is_available = lambda: False

    try:
        # To enable voice input, you need to install whisper: pip install openai-whisper
        import whisper
        # Using "base" model for a balance of speed and accuracy.
        # Other options: "tiny", "small", "medium", "large"
        logging.info("Loading Speech-to-Text model (whisper base)...")
        stt_model = whisper.load_model("base")
        logging.info("Speech-to-Text model loaded successfully.")
    except ImportError:
        logging.warning("Whisper not installed. To enable voice input, run: pip install openai-whisper")
        stt_model = None
    except Exception as e:
        logging.error(f"Could not load whisper model: {e}")
        logging.warning("Voice input for Conversational AI will be disabled.")
        stt_model = None

    # Load the plug-in Arabic TTS model
    try:
        logging.info("Loading Arabic TTS model (facebook/mms-tts-ara)...")
        arabic_tts_processor = AutoProcessor.from_pretrained("facebook/mms-tts-ara")
        arabic_tts_model = AutoModel.from_pretrained("facebook/mms-tts-ara")
        logging.info("Arabic TTS model loaded successfully.")
    except Exception as e:
        logging.warning(f"Could not load Arabic TTS model: {e}. Arabic will not be available.")
        arabic_tts_processor = None
        arabic_tts_model = None

    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir)
        except Exception as e:
            logging.error(f"Failed to load both CosyVoice and CosyVoice2 models. Error: {e}")
            raise TypeError('no valid model_type!') from e

    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)

    def cleanup():
        logging.info("Cleaning up resources...")
        global cosyvoice
        if 'cosyvoice' in globals() and cosyvoice is not None:
            del cosyvoice
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logging.info("Cleanup complete.")

    atexit.register(cleanup)
    main()