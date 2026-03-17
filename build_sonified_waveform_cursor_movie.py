#!/usr/bin/env python3
import argparse
import os
import numpy as np

# Use a non-interactive backend for headless runs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import soundfile as sf
except Exception as e:
    raise SystemExit("This script requires the 'soundfile' package. Install with: pip install soundfile") from e

try:
    from moviepy.editor import VideoClip, AudioFileClip
    from moviepy.video.io.bindings import mplfig_to_npimage
except Exception as e:
    raise SystemExit("This script requires 'moviepy'. Install with: pip install moviepy") from e


def make_waveform_video_with_cursor(
    audio_path: str,
    out_path: str,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    line_color: str = "#E64646",
    background: str = "white",
    waveform_color: str = "#1f77b4",
    decimate_points: int = 6000,
):
    """
    Create an MP4 video that shows the waveform and a vertical cursor moving in sync with the audio.

    Parameters
    ----------
    audio_path : str
        Path to the input WAV/MP3/… file (anything ffmpeg can decode).
    out_path : str
        Path to the output .mp4 file.
    fps : int
        Frames per second for the video.
    width, height : int
        Pixel dimensions of the output video.
    line_color, background, waveform_color : str
        Colors for the cursor, figure background, and waveform line.
    decimate_points : int
        If the audio has many samples, plot at most this many points for the background waveform line
        (to keep rendering fast). The audio itself is NOT decimated; only the plotted background.
    """

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio to get samples (for waveform) and sample rate
    y, sr = sf.read(audio_path, dtype="float32", always_2d=False)

    # If stereo, convert to mono for plotting (average channels)
    if y.ndim == 2:
        y = y.mean(axis=1)

    n = len(y)
    if n == 0:
        raise ValueError("Loaded audio is empty.")

    duration_wave = n / float(sr)

    # Build time axis in seconds
    t = np.arange(n, dtype=np.float64) / float(sr)

    # Prepare a decimated version for the static background waveform (speeds up rendering)
    if n > decimate_points:
        step = int(np.ceil(n / decimate_points))
        t_plot = t[::step]
        y_plot = y[::step]
    else:
        t_plot = t
        y_plot = y

    # Basic plot setup
    dpi = 200
    fig_w = width / dpi
    fig_h = height / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    # Plot the (decimated) waveform once as static background
    max_abs = float(np.max(np.abs(y))) if np.any(y) else 1.0
    ax.plot(t_plot, y_plot, lw=0.7, color=waveform_color)
    ax.set_xlim(t_plot[0], t_plot[-1])
    ax.set_ylim(-1.05 * max_abs, 1.05 * max_abs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(False)

    # Add a cursor line (vertical)
    cursor = ax.axvline(t_plot[0], lw=1.5, color=line_color)

    # Prepare audio clip via MoviePy (lets ffmpeg handle formats and aac encoding)
    audio_clip = AudioFileClip(audio_path)
    duration = min(duration_wave, audio_clip.duration)  # protect against tiny mismatches

    # Frame-making function for MoviePy
    def make_frame(current_t: float):
        cursor.set_xdata([current_t, current_t])
        # Convert the Matplotlib figure to an RGB image (numpy array)
        return mplfig_to_npimage(fig)

    # Build the video clip
    video_clip = VideoClip(make_frame, duration=duration).set_audio(audio_clip.subclip(0, duration)).set_fps(fps)

    # Write the file
    # libx264 + yuv420p for broad compatibility; aac for audio
    video_clip.write_videofile(
        out_path,
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        preset="medium",
        threads=4,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Make a waveform video with a moving cursor synced to the audio.")
    parser.add_argument("--audio", required=True, help="Path to input audio (wav/mp3/etc.).")
    parser.add_argument("--out", required=True, help="Path to output .mp4 video.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the video.")
    parser.add_argument("--width", type=int, default=1280, help="Output video width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Output video height in pixels.")
    parser.add_argument("--cursor_color", default="#E64646", help="Vertical cursor color.")
    parser.add_argument("--wave_color", default="#1f77b4", help="Waveform line color.")
    parser.add_argument("--bg", default="white", help="Background color.")
    args = parser.parse_args()

    make_waveform_video_with_cursor(
        audio_path=args.audio,
        out_path=args.out,
        fps=args.fps,
        width=args.width,
        height=args.height,
        line_color=args.cursor_color,
        background=args.bg,
        waveform_color=args.wave_color,
    )


if __name__ == "__main__":
    main()
