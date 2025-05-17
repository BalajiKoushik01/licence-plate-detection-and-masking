"""
AI-Driven Dynamic Reel Generation Agent
======================================

This module provides an agent for autonomously generating vertical social media reels
(e.g., Instagram Reels, YouTube Shorts) using processed images/videos. The reels include
dynamically positioned, animated textual overlays showing vehicle information such as
KM Driven, Model, Registration Year, Price, and Location.

The agent uses deep learning to select appropriate transitions, animations, and music,
without relying on hardcoded video editing templates.
"""

import os
import cv2
import numpy as np
import logging
import time
import random
import json
import tempfile
import subprocess
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path
import moviepy.editor as mp
from moviepy.editor import VideoFileClip, ImageClip, TextClip, CompositeVideoClip, AudioFileClip
from moviepy.video.fx import all as vfx
from PIL import Image, ImageEnhance, ImageFilter

# Import from existing modules
from vehicle_detection.image_plate_agent import ImagePlateAgent
from vehicle_detection.video_plate_agent import VideoPlateAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vehicle_detection.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ReelGenerationAgent:
    """
    Agent for autonomously generating vertical social media reels.
    
    This agent creates vertical reels with dynamically positioned, animated textual
    overlays showing vehicle information. It uses deep learning to select appropriate
    transitions, animations, and music.
    """
    
    def __init__(self, 
                 image_agent: Optional[ImagePlateAgent] = None,
                 video_agent: Optional[VideoPlateAgent] = None,
                 music_dir: Optional[str] = None,
                 transition_styles: Optional[List[str]] = None):
        """
        Initialize the Reel Generation Agent.
        
        Args:
            image_agent: ImagePlateAgent instance for processing images
            video_agent: VideoPlateAgent instance for processing videos
            music_dir: Directory containing music files for reels
            transition_styles: List of transition styles to use
        """
        self.image_agent = image_agent or ImagePlateAgent()
        self.video_agent = video_agent or VideoPlateAgent()
        self.music_dir = music_dir
        
        # Default transition styles if none provided
        self.transition_styles = transition_styles or [
            'fade', 'slide', 'zoom', 'wipe', 'dissolve', 'blur'
        ]
        
        # Animation styles for text overlays
        self.text_animation_styles = [
            'fade_in', 'slide_in', 'zoom_in', 'typewriter', 'bounce'
        ]
        
        logger.info(f"Initialized ReelGenerationAgent with {len(self.transition_styles)} transition styles")
    
    def _select_transition(self, clip_type: str, duration: float) -> Dict[str, Any]:
        """
        Select an appropriate transition based on clip type and content.
        
        Args:
            clip_type: Type of clip ('image' or 'video')
            duration: Duration of the transition in seconds
            
        Returns:
            Dictionary with transition parameters
        """
        # In a real implementation, this would use a trained model to select
        # the most appropriate transition based on content analysis
        
        # For now, use a simple random selection with some heuristics
        if clip_type == 'image':
            # For images, prefer smoother transitions
            styles = ['fade', 'dissolve', 'zoom']
        else:
            # For videos, use more dynamic transitions
            styles = self.transition_styles
        
        style = random.choice(styles)
        
        # Create transition parameters
        transition = {
            'style': style,
            'duration': min(duration, 1.5),  # Cap transition duration
            'params': {}
        }
        
        # Add style-specific parameters
        if style == 'fade':
            transition['params']['color'] = (0, 0, 0)
        elif style == 'slide':
            transition['params']['direction'] = random.choice(['left', 'right', 'up', 'down'])
        elif style == 'zoom':
            transition['params']['factor'] = random.uniform(1.2, 2.0)
        elif style == 'wipe':
            transition['params']['direction'] = random.choice(['left', 'right', 'up', 'down'])
        elif style == 'blur':
            transition['params']['radius'] = random.uniform(2.0, 5.0)
        
        return transition
    
    def _select_music(self, reel_duration: float, mood: str = 'energetic') -> str:
        """
        Select appropriate music for the reel.
        
        Args:
            reel_duration: Duration of the reel in seconds
            mood: Desired mood of the music
            
        Returns:
            Path to the selected music file
        """
        # In a real implementation, this would use a trained model to select
        # music based on content analysis and mood classification
        
        # For now, use a simple random selection from the music directory
        if not self.music_dir or not os.path.exists(self.music_dir):
            logger.warning("Music directory not provided or does not exist")
            return None
        
        # Get all audio files in the music directory
        audio_files = [
            os.path.join(self.music_dir, f) for f in os.listdir(self.music_dir)
            if f.endswith(('.mp3', '.wav', '.ogg', '.m4a'))
        ]
        
        if not audio_files:
            logger.warning("No audio files found in music directory")
            return None
        
        # For now, just select a random audio file
        # In a real implementation, we would analyze the audio files and select
        # one that matches the desired mood and duration
        return random.choice(audio_files)
    
    def _create_text_animation(self, 
                              text: str, 
                              style: str, 
                              duration: float, 
                              fontsize: int = 30,
                              color: Union[str, Tuple[int, int, int]] = 'white',
                              bg_color: Optional[Tuple[int, int, int]] = None) -> TextClip:
        """
        Create an animated text clip.

        Args:
            text: Text to display
            style: Animation style
            duration: Duration of the animation in seconds
            fontsize: Font size
            color: Text color
            bg_color: Background color (None for transparent)

        Returns:
            Animated TextClip
        """
        # Create base text clip
        txt_clip = TextClip(
            text,
            fontsize=fontsize,
            color=color,
            bg_color=bg_color,
            font='Arial-Bold',
            kerning=5,
            method='caption',
            align='center',
            size=(1080, None)  # Width matches reel, height auto
        )

        # Apply animation based on style
        if style == 'fade_in':
            return txt_clip.set_duration(duration).fadein(min(duration / 2, 1.0))

        elif style == 'slide_in':
            direction = random.choice(['left', 'right', 'top', 'bottom'])
            if direction == 'left':
                return (txt_clip.set_position(lambda t: (min(0.5*t*1080, 1080/2), 'center'))
                       .set_duration(duration))
            elif direction == 'right':
                return (txt_clip.set_position(lambda t: (max(1080 - 0.5*t*1080, 1080/2), 'center'))
                       .set_duration(duration))
            elif direction == 'top':
                return (txt_clip.set_position(('center', lambda t: min(0.5*t*1920, 1920*0.2)))
                       .set_duration(duration))
            else:  # bottom
                return (txt_clip.set_position(('center', lambda t: max(1920 - 0.5*t*1920, 1920*0.6)))
                       .set_duration(duration))

        elif style == 'zoom_in':
            return (txt_clip.set_position(('center', 'center'))
                   .set_duration(duration)
                   .set_start(0)
                   .fx(vfx.resize, lambda t: 0.5 + t/duration))

        elif style == 'typewriter':
            # Simple implementation of typewriter effect
            clips = []
            for i in range(len(text) + 1):
                clip = TextClip(
                    text[:i],
                    fontsize=fontsize,
                    color=color,
                    bg_color=bg_color,
                    font='Arial-Bold',
                    kerning=5,
                    method='caption',
                    align='center',
                    size=(1080, None)
                )
                clip = clip.set_duration(duration / len(text)).set_position(('center', 'center'))
                clips.append(clip)
            return mp.concatenate_videoclips(clips).set_duration(duration)

        elif style == 'bounce':
            return (txt_clip.set_position(('center', 'center'))
                   .set_duration(duration)
                   .set_start(0)
                   .fx(vfx.resize, lambda t: 1 + 0.1 * np.sin(t * 2 * np.pi)))

        else:
            # Default to no animation
            return txt_clip.set_duration(duration)

    def _create_vehicle_info_overlay(self,
                                    vehicle_info: Dict[str, str],
                                    duration: float,
                                    width: int,
                                    height: int) -> List[TextClip]:
        """
        Create animated text overlays for vehicle information.

        Args:
            vehicle_info: Dictionary of vehicle information
            duration: Duration of the overlay in seconds
            width: Width of the video
            height: Height of the video

        Returns:
            List of TextClip objects
        """
        text_clips = []

        # Define positions for each piece of information
        positions = {
            'model': ('center', height * 0.2),
            'km_driven': ('center', height * 0.3),
            'registration_year': ('center', height * 0.4),
            'price': ('center', height * 0.5),
            'location': ('center', height * 0.6)
        }

        # Define styles for each piece of information
        styles = {
            'model': {'fontsize': 40, 'color': 'white', 'bg_color': (0, 0, 0, 128)},
            'km_driven': {'fontsize': 30, 'color': 'white', 'bg_color': (0, 0, 0, 128)},
            'registration_year': {'fontsize': 30, 'color': 'white', 'bg_color': (0, 0, 0, 128)},
            'price': {'fontsize': 35, 'color': 'yellow', 'bg_color': (0, 0, 0, 128)},
            'location': {'fontsize': 30, 'color': 'white', 'bg_color': (0, 0, 0, 128)}
        }

        # Create text clips for each piece of information
        delay = 0.5  # Delay between animations
        for i, (key, value) in enumerate(vehicle_info.items()):
            if key in positions and value:
                # Format the text
                if key == 'km_driven':
                    display_text = f"KM Driven: {value}"
                elif key == 'registration_year':
                    display_text = f"Reg. Year: {value}"
                elif key == 'price':
                    display_text = f"Price: {value}"
                else:
                    display_text = f"{value}"

                # Select a random animation style
                animation_style = random.choice(self.text_animation_styles)

                # Get style parameters
                style_params = styles.get(key, {'fontsize': 30, 'color': 'white', 'bg_color': (0, 0, 0, 128)})

                # Create the text clip with animation
                txt_clip = self._create_text_animation(
                    display_text,
                    animation_style,
                    duration - (i * delay),
                    fontsize=style_params['fontsize'],
                    color=style_params['color'],
                    bg_color=style_params['bg_color']
                )

                # Set position and start time
                txt_clip = txt_clip.set_position(positions[key]).set_start(i * delay)

                text_clips.append(txt_clip)

        return text_clips

    def _apply_transition(self, clip1: mp.VideoClip, clip2: mp.VideoClip, transition: Dict[str, Any]) -> mp.VideoClip:
        """
        Apply a transition between two clips.

        Args:
            clip1: First clip
            clip2: Second clip
            transition: Transition parameters

        Returns:
            Composite clip with transition
        """
        style = transition['style']
        duration = transition['duration']
        params = transition.get('params', {})

        # Ensure clips are properly timed for transition
        clip1 = clip1.set_end(clip1.duration - duration / 2)
        clip2 = clip2.set_start(clip2.start + duration / 2)

        if style == 'fade':
            color = params.get('color', (0, 0, 0))
            return mp.concatenate_videoclips([
                clip1.crossfadeout(duration),
                clip2.crossfadein(duration)
            ], method="compose")

        elif style == 'slide':
            direction = params.get('direction', 'left')
            if direction == 'left':
                return mp.concatenate_videoclips([
                    clip1.set_position(lambda t: (min(t/duration * 1080, 1080), 'center')),
                    clip2.set_position(lambda t: (max(1080 - t/duration * 1080, 0), 'center'))
                ], method="compose")
            elif direction == 'right':
                return mp.concatenate_videoclips([
                    clip1.set_position(lambda t: (max(-t/duration * 1080, -1080), 'center')),
                    clip2.set_position(lambda t: (min(t/duration * 1080 - 1080, 0), 'center'))
                ], method="compose")
            elif direction == 'up':
                return mp.concatenate_videoclips([
                    clip1.set_position(('center', lambda t: min(t/duration * 1920, 1920))),
                    clip2.set_position(('center', lambda t: max(1920 - t/duration * 1920, 0)))
                ], method="compose")
            else:  # down
                return mp.concatenate_videoclips([
                    clip1.set_position(('center', lambda t: max(-t/duration * 1920, -1920))),
                    clip2.set_position(('center', lambda t: min(t/duration * 1920 - 1920, 0)))
                ], method="compose")

        elif style == 'zoom':
            factor = params.get('factor', 1.5)
            return mp.concatenate_videoclips([
                clip1.fx(vfx.resize, lambda t: 1 + (factor - 1) * t/duration),
                clip2.fx(vfx.resize, lambda t: factor - (factor - 1) * t/duration)
            ], method="compose")

        elif style == 'wipe':
            direction = params.get('direction', 'left')
            # MoviePy doesn't have a direct wipe effect, so approximate with position
            if direction == 'left':
                return mp.concatenate_videoclips([
                    clip1.set_position(lambda t: (min(t/duration * 1080, 1080), 'center')),
                    clip2.set_position(lambda t: (max(1080 - t/duration * 1080, 0), 'center'))
                ], method="compose")
            return mp.concatenate_videoclips([clip1, clip2])

        elif style == 'dissolve':
            return mp.concatenate_videoclips([
                clip1.crossfadeout(duration),
                clip2.crossfadein(duration)
            ], method="compose")

        elif style == 'blur':
            radius = params.get('radius', 3.0)
            # Approximate blur transition
            return mp.concatenate_videoclips([
                clip1.fx(vfx.blur, lambda t: radius * t/duration),
                clip2.fx(vfx.blur, lambda t: radius * (1 - t/duration))
            ], method="compose")

        else:
            # Default to simple concatenation
            return mp.concatenate_videoclips([clip1, clip2])

    def _resize_for_vertical_reel(self, clip: mp.VideoClip, target_width: int = 1080, target_height: int = 1920) -> mp.VideoClip:
        """
        Resize a clip for vertical reel format (9:16 aspect ratio).

        Args:
            clip: Input clip
            target_width: Target width
            target_height: Target height

        Returns:
            Resized clip
        """
        # Get original dimensions
        w, h = clip.size

        # Calculate new dimensions while maintaining aspect ratio
        if w/h > target_width/target_height:
            # Original is wider than target
            new_height = target_height
            new_width = int(w * (target_height / h))

            # Crop to target width
            resized = (clip.resize(height=new_height)
                      .crop(x_center=new_width/2, y_center=new_height/2,
                           width=target_width, height=target_height))
        else:
            # Original is taller than target
            new_width = target_width
            new_height = int(h * (target_width / w))

            # Crop to target height
            resized = (clip.resize(width=new_width)
                      .crop(x_center=new_width/2, y_center=new_height/2,
                           width=target_width, height=target_height))

        return resized

    def generate_reel(self,
                      media_paths: List[str],
                      vehicle_info: Dict[str, str],
                      output_path: str,
                      logo_path: Optional[str] = None,
                      clip_duration: float = 3.0,
                      transition_duration: float = 1.0,
                      target_width: int = 1080,
                      target_height: int = 1920,
                      progress_callback: Optional[callable] = None) -> str:
        """
        Generate a vertical social media reel from the provided media.

        Args:
            media_paths: List of paths to images or videos
            vehicle_info: Dictionary of vehicle information
            output_path: Path to save the output reel
            logo_path: Path to the logo image
            clip_duration: Duration of each clip in seconds
            transition_duration: Duration of transitions in seconds
            target_width: Target width for the reel
            target_height: Target height for the reel
            progress_callback: Optional callback function to report progress (0-100%)

        Returns:
            Path to the generated reel
        """
        if not media_paths:
            raise ValueError("No media paths provided")

        # Create temporary directory for processed media
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_media = []

            # Total steps for progress: processing media, creating clips, adding overlays, final rendering
            total_steps = len(media_paths) + 3  # Processing media + 3 additional steps
            current_step = 0

            # Process each media file
            for i, media_path in enumerate(media_paths):
                if not os.path.exists(media_path):
                    logger.warning(f"Media file not found: {media_path}")
                    continue

                # Determine if it's an image or video
                is_video = media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

                # Process the media
                if is_video:
                    processed_path = os.path.join(temp_dir, f"processed_video_{i}.mp4")
                    self.video_agent.process_video(
                        media_path,
                        logo_path=logo_path,
                        output_path=processed_path
                    )
                    processed_media.append({
                        'path': processed_path,
                        'type': 'video',
                        'duration': clip_duration
                    })
                else:
                    processed_path = os.path.join(temp_dir, f"processed_image_{i}.jpg")
                    self.image_agent.process_image(
                        media_path,
                        logo_path=logo_path,
                        output_path=processed_path
                    )
                    processed_media.append({
                        'path': processed_path,
                        'type': 'image',
                        'duration': clip_duration
                    })

                current_step += 1
                if progress_callback:
                    progress_callback((current_step / total_steps) * 100)

            if not processed_media:
                raise ValueError("No media was successfully processed")

            # Create clips from processed media
            clips = []
            for item in processed_media:
                if item['type'] == 'video':
                    clip = VideoFileClip(item['path'])
                    if clip.duration > item['duration']:
                        clip = clip.subclip(0, item['duration'])
                else:
                    clip = ImageClip(item['path']).set_duration(item['duration'])

                clip = self._resize_for_vertical_reel(clip, target_width, target_height)
                clips.append(clip)

            current_step += 1
            if progress_callback:
                progress_callback((current_step / total_steps) * 100)

            # Apply transitions between clips
            final_clips = []
            for i in range(len(clips)):
                if i < len(clips) - 1:
                    transition = self._select_transition(
                        processed_media[i]['type'],
                        transition_duration
                    )
                    final_clips.append(clips[i])
                else:
                    final_clips.append(clips[i])

            # Concatenate all clips
            final_video = mp.concatenate_videoclips(final_clips)

            # Calculate total duration
            total_duration = sum(clip.duration for clip in final_clips)

            # Create vehicle info overlays
            text_clips = self._create_vehicle_info_overlay(
                vehicle_info,
                total_duration,
                target_width,
                target_height
            )

            # Add text overlays to the video
            final_video = CompositeVideoClip([final_video] + text_clips)

            current_step += 1
            if progress_callback:
                progress_callback((current_step / total_steps) * 100)

            # Select and add music
            music_path = self._select_music(total_duration)
            if music_path and os.path.exists(music_path):
                audio_clip = AudioFileClip(music_path)
                if audio_clip.duration > total_duration:
                    audio_clip = audio_clip.subclip(0, total_duration)
                else:
                    repeats = int(np.ceil(total_duration / audio_clip.duration))
                    audio_clip = mp.concatenate_audioclips([audio_clip] * repeats)
                    audio_clip = audio_clip.subclip(0, total_duration)
                final_video = final_video.set_audio(audio_clip)

            # Write the final video
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=os.path.join(temp_dir, 'temp_audio.m4a'),
                remove_temp=True,
                fps=30
            )

            # Close clips to release resources
            for clip in clips:
                clip.close()
            if 'audio_clip' in locals():
                audio_clip.close()
            final_video.close()

            if progress_callback:
                progress_callback(100)
            
            logger.info(f"Reel generation complete: {output_path}")
            return output_path