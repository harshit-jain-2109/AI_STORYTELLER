# !pip install gradio
# %pip install --quiet --upgrade diffusers transformers accelerate mediapy peft
# !pip install gtts moviepy




import re
import numpy as np
import random
import sys
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import DiffusionPipeline, TCDScheduler
from huggingface_hub import hf_hub_download
from gtts import gTTS
from moviepy.editor import ImageSequenceClip, VideoFileClip, concatenate_videoclips, AudioFileClip
import gradio as gr

# Install necessary packages (if not already installed)
# Choose among 1, 2, 4 and 8:
num_inference_steps = 8

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
plural = "s" if num_inference_steps > 1 else ""
ckpt_name = f"Hyper-SDXL-{num_inference_steps}step{plural}-lora.safetensors"
device = "cuda"

pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora()
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

def generate_image(prompt, step_count=50, seed=None):
    if seed is None:
        seed = random.randint(0, sys.maxsize)
    generator = torch.Generator(device).manual_seed(seed)
    eta = 0.5
    images = pipe(
        prompt=prompt,
        num_inference_steps=step_count,
        guidance_scale=0.0,
        eta=eta,
        generator=generator,
    ).images
    return images[0]

def draw_text_on_image(image, text, font_path="arial.ttf", font_size=24):
    image_with_text = image.copy()
    draw = ImageDraw.Draw(image_with_text)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"Font {font_path} not found. Using default font.")
        font = ImageFont.load_default()

    # Split text into multiple lines to fit within the image
    lines = []
    max_width = image.width - 20  # Padding of 10 pixels on each side
    words = text.split()
    while words:
        line = ''
        while words and draw.textlength(line + words[0], font=font) <= max_width:
            line = f"{line} {words.pop(0)}" if line else words.pop(0)
        lines.append(line)

    # Calculate total text height
    text_height = sum(draw.textbbox((0, 0), line, font=font)[3] for line in lines)
    # Position text at the bottom of the image
    text_y = image.height - text_height - 20  # Padding of 10 pixels from the bottom

    for line in lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (image.width - text_width) // 2  # Centered horizontally

        # Draw background rectangle for text
        draw.rectangle([(text_x - 5, text_y - 5), (text_x + text_width + 5, text_y + text_height + 5)], fill="black")
        # Draw text on top of the rectangle
        draw.text((text_x, text_y), line, font=font, fill="white")
        text_y += text_height + 5  # Move to the next line with some padding

    return image_with_text

def process_story(story):
    # Use regular expressions to split the text into sentences
    sentences = re.split(r'(?<=[.!?]) +', story.strip())

    # Initialize lists for video clips and audio clips
    video_clips = []
    fps = 24  # Frames per second

    # Generate images, overlay text, and create audio
    for i, sentence in enumerate(sentences):
        print(f"Sentence {i+1}: {sentence}\n")
        seed = random.randint(0, sys.maxsize)
        image = generate_image(sentence, step_count=50, seed=seed)  # Increase step count for better quality images
        resized_image = image.resize((256, 256))
        image_with_text = draw_text_on_image(resized_image, sentence)

        # Save the image with text
        image_path = f"sentence_{i+1}.png"
        image_with_text.save(image_path)

        frame = np.array(image_with_text)  # Convert to NumPy array

        # Generate audio for the sentence
        tts = gTTS(sentence, lang='en')
        audio_path = f"sentence_{i+1}.mp3"
        tts.save(audio_path)
        audio_clip = AudioFileClip(audio_path)

        # Create a video clip from the image and set the duration to the audio duration
        video_clip = ImageSequenceClip([frame], fps=fps)
        video_clip = video_clip.set_duration(audio_clip.duration)
        video_clip = video_clip.set_audio(audio_clip)

        # Save the individual video clip
        clip_path = f"sentence_{i+1}.mp4"
        video_clip.write_videofile(clip_path, codec="libx264", audio_codec="aac")

        video_clips.append(video_clip)

        # Clear GPU memory
        del resized_image, image_with_text
        torch.cuda.empty_cache()

    # Concatenate all video clips into a final video
    final_video = concatenate_videoclips(video_clips)
    final_video_path = "story_video.mp4"
    final_video.write_videofile(final_video_path, codec="libx264", audio_codec="aac")

    return final_video_path

def generate_story_video(story):
    final_video_path = process_story(story)
    return final_video_path

iface = gr.Interface(
    fn=generate_story_video,
    inputs="text",
    outputs="video",
    title="Story to Video Generator",
    description="Enter a story and get a video with images and narrated text.",
)

if __name__ == "__main__":
    iface.launch()
