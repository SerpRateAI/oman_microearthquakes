#!/usr/bin/env python3
# Create an MP4 movie with:
#   (top) waveform from the HDF5 slice (matching your preprocessing) with a vertical cursor synced to audio
#   (bottom) spectrum (multitaper) like in your plot
# All helper functions are defined at the beginning; imports are explicit. The waveform is read
# using the same utilities you use in your sonification script (load_waveform_slice, ROOTDIR_GEO, etc.).
# The soundtrack is generated directly from the waveform (resampled to an audio-compatible rate).

# -----------------------------
# Imports
# -----------------------------
from argparse import ArgumentParser
from os.path import join, splitext

# Headless-friendly Matplotlib backend
from matplotlib import use as mpl_use
mpl_use("Agg")

from matplotlib.pyplot import subplots, close
from matplotlib.gridspec import GridSpec
from matplotlib.dates import date2num, DateFormatter

from numpy import arange, float64, abs as np_abs, max as np_max, any as np_any, ceil as np_ceil, float32, frombuffer, uint8

from moviepy.editor import VideoClip, AudioFileClip

from functools import partial
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Project utilities — same as your script
from scipy.signal.windows import dpss
from scipy.signal import resample_poly
from math import gcd
from soundfile import write as sf_write
import librosa

from utils_mt import mt_autospec
from utils_basic import (
    ROOTDIR_GEO as dirpath_in,
    AUDIO_DIR as dirpath_out,
    SAMPLING_RATE as sampling_rate_in,
    SAMPLING_RATE_AUDIO as sampling_rate_out,
    get_freq_limits_string,
    power2db,
)
from utils_cont_waveform import load_waveform_slice

from utils_plot import component2label

# Optional project-specific axis formatters
try:
    from utils_plot import format_datetime_xlabels  # type: ignore
except Exception:
    format_datetime_xlabels = None  # type: ignore

try:
    from utils_plot import format_freq_xlabels  # type: ignore
except Exception:
    format_freq_xlabels = None  # type: ignore

from pandas import Timestamp
import tempfile


# -----------------------------
# Constants
# -----------------------------
SEC_PER_DAY = 86400.0


# -----------------------------
# Helper functions (defined first)
# -----------------------------
def decimate_for_plot(array, max_points: int):
    n = len(array)
    if n <= max_points:
        return array
    step = int(np_ceil(n / float(max_points)))
    return array[::step]


def compute_datetime_axis_from_start(starttime: "Timestamp", n_samples: int, sample_rate: int):
    t_sec = arange(n_samples, dtype=float64) / float(sample_rate)
    start_num = date2num(starttime.to_pydatetime())
    tnum = start_num + t_sec / SEC_PER_DAY
    return start_num, tnum


def setup_figure_with_spectrum(width: int, height: int, dpi: int, bg_color: str):
    fig_w = width / float(dpi)
    fig_h = height / float(dpi)
    fig, (ax_wave, ax_spec) = subplots(2, 1, figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(bg_color)
    ax_wave.set_facecolor(bg_color)
    ax_spec.set_facecolor(bg_color)
    ax_wave = ax_wave
    ax_spec = ax_spec
    return fig, ax_wave, ax_spec


def format_time_axis(ax, major_tick_spacing: str, axis_label_size: float, tick_label_size: float):
    if format_datetime_xlabels is not None:
        format_datetime_xlabels(ax, major_tick_spacing=major_tick_spacing,
                                axis_label_size=axis_label_size,
                                tick_label_size=tick_label_size)
    else:
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
        ax.tick_params(axis="x", labelsize=tick_label_size)
        ax.set_xlabel("Time", fontsize=axis_label_size)


def draw_static_waveform(ax, tnum_plot, y_plot_dec, wave_color: str, axis_label_size: float, tick_label_size: float):
    if np_any(y_plot_dec):
        max_abs = float(np_max(np_abs(y_plot_dec)))
    else:
        max_abs = 1.0
    ax.plot(tnum_plot, y_plot_dec, lw=0.7, color=wave_color)
    ax.set_xlim(tnum_plot[0], tnum_plot[-1])
    ax.set_ylim(-1.05 * max_abs, 1.05 * max_abs)
    ax.set_ylabel("Normalized amplitude")
    ax.yaxis.set_tick_params(labelsize=tick_label_size)
    return max_abs


def draw_static_spectrum(ax, y_mono, sr: int, nw: int, fmin: float, fmax: float,
                         axis_label_size: float, tick_label_size: float, line_color: str):
    num_pts = len(y_mono)
    num_taper = 2 * nw - 1
    taper_mat, ratio_vec = dpss(num_pts, nw, num_taper, return_ratios=True)
    param = mt_autospec(y_mono, taper_mat, ratio_vec, sr, normalize=True)
    spec_db = power2db(param.aspec)
    freq_axis = param.freqax

    ax.plot(freq_axis, spec_db, color=line_color, linewidth=1.0)
    ax.set_xlim(fmin, fmax)
    ax.set_ylabel("Normalized power (dB)")
    ax.yaxis.set_tick_params(labelsize=tick_label_size)

    if format_freq_xlabels is not None:
        major = max((fmax - fmin) / 5.0, 1.0)
        format_freq_xlabels(ax, major_tick_spacing=major,
                            axis_label_size=axis_label_size,
                            tick_label_size=tick_label_size)
    else:
        ax.set_xlabel("Frequency (Hz)")
        ax.tick_params(axis="x", labelsize=tick_label_size)
    ax.set_ylim(-10, 40)


def create_cursor(ax, x0: float, cursor_color: str):
    return ax.axvline(x0, lw=1.8, color=cursor_color)


def fig_to_rgb_frame(fig):
    """Matplotlib→RGB numpy array, compatible with Matplotlib ≥3.8 (no tostring_rgb)."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    # Convert RGBA memoryview → (H,W,3) uint8
    frame = frombuffer(buf, dtype=uint8).reshape(h, w, 4)[..., :3]
    return frame


def render_frame(current_t: float, fig, cursor, start_num: float):
    x = start_num + current_t / SEC_PER_DAY
    cursor.set_xdata([x, x])
    return fig_to_rgb_frame(fig)


# -----------------------------
# Primary function
# -----------------------------
def make_movie_from_hdf5(
    station: str,
    starttime: "Timestamp",
    duration: float,
    component: str,
    outpath: str,
    min_freq_filter: float,
    max_freq_filter: float,
    nw: int,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    dpi: int = 300,
    cursor_color: str = "crimson",
    wave_color: str = "black",
    bg_color: str = "white",
    axis_label_size: float = 12.0,
    tick_label_size: float = 10.0,
    major_tick_spacing: str = "15s",
    plot_max_points: int = 12000,
) -> None:

    print(f"Width: {width}, Height: {height}")

    # ---- Load waveform slice ----

    freq_str = get_freq_limits_string(min_freq_filter, max_freq_filter)
    num_pts = int(duration * sampling_rate_in)
    filename = f"preprocessed_data_{freq_str}.h5"
    filepath_in = join(dirpath_in, filename)

    waveform_dict, time_axis = load_waveform_slice(
        filepath_in, station, starttime, num_pts=num_pts
    )
    y = waveform_dict[component]

    # Normalize for audio safety
    y = y / np_max(np_abs(y)) * 0.98

    # Resample to audio-compatible rate first
    g = gcd(sampling_rate_out, int(sampling_rate_in))
    up = sampling_rate_out // g
    down = sampling_rate_in // g
    y_audio = resample_poly(y, up, down).astype(float32)

    # Pitch shift WITHOUT changing duration (phase-vocoder)
    if 'freq_scale_global' in globals() and abs(freq_scale_global - 1.0) > 1e-9:
        import math
        semitones = 12 * math.log2(freq_scale_global)
        y_audio = librosa.effects.pitch_shift(y_audio.astype(float32), sr=sampling_rate_out, n_steps=semitones).astype(float32)

    # Safety normalization after processing
    peak = float(np_max(np_abs(y_audio))) if np_any(y_audio) else 0.0
    if peak > 0:
        y_audio = (0.98 / peak) * y_audio


    # Save temporary WAV file
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf_write(tmp_wav.name, y_audio, sampling_rate_out, subtype="PCM_16")

    # Build datetime axis for plotting
    n = len(y)
    start_num, tnum = compute_datetime_axis_from_start(starttime, n, sampling_rate_in)
    tnum_plot = decimate_for_plot(tnum, plot_max_points)
    y_plot_dec = decimate_for_plot(y, plot_max_points)

    # ---- Figure ----
    fig, ax_wave, ax_spec = setup_figure_with_spectrum(width=width, height=height, dpi=dpi, bg_color=bg_color)
    fig.subplots_adjust(hspace=0.3)
    ax_wave.set_title(f"{station}, {component2label(component)}", fontsize=14, fontweight="bold")

    draw_static_waveform(ax_wave, tnum_plot, y_plot_dec, wave_color, axis_label_size, tick_label_size)
    format_time_axis(ax_wave, major_tick_spacing, axis_label_size, tick_label_size)
    draw_static_spectrum(ax_spec, y, sampling_rate_in, nw, min_freq_filter, max_freq_filter,
                         axis_label_size, tick_label_size, wave_color)

    cursor = create_cursor(ax_wave, tnum_plot[0], cursor_color)

    # verify right before building the clip
    frame0 = render_frame(0.0, fig=fig, cursor=cursor, start_num=start_num)
    print("FIRST FRAME SHAPE (WxHxC):", frame0.shape[1], frame0.shape[0], frame0.shape[2])
    
    # ---- Audio ----
    audio_clip = AudioFileClip(tmp_wav.name)
    duration_clip = audio_clip.duration

    frame_fn = partial(render_frame, fig=fig, cursor=cursor, start_num=start_num)

    clip = VideoClip(frame_fn, duration=duration_clip).set_audio(audio_clip.subclip(0, duration_clip)).set_fps(fps)

    try:
        clip.write_videofile(
            outpath,
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            preset="medium",
            threads=4,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
    finally:
        # Ensure resources are released even if ffmpeg errors, and remove temp file
        try:
            audio_clip.close()
        except Exception:
            pass
        try:
            clip.close()
        except Exception:
            pass
        try:
            import os
            os.unlink(tmp_wav.name)
        except Exception:
            pass
        close(fig)


# -----------------------------
# CLI entry point
# -----------------------------
def main() -> None:
    parser = ArgumentParser(description="Make a waveform+cursor movie with spectrum, reading waveform from HDF5 and generating soundtrack.")
    parser.add_argument("--station", required=True, type=str)
    parser.add_argument("--starttime", required=True, type=Timestamp)
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--component", default="Z", type=str)
    parser.add_argument("--min_freq_filter", default=20.0, type=float)
    parser.add_argument("--max_freq_filter", default=200.0, type=float)
    parser.add_argument("--nw", default=3, type=int)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--cursor_color", default="crimson")
    parser.add_argument("--wave_color", default="black")
    parser.add_argument("--bg_color", default="white")
    parser.add_argument("--axis_label_size", type=float, default=12.0)
    parser.add_argument("--tick_label_size", type=float, default=10.0)
    parser.add_argument("--major_tick_spacing", default="15s")
    parser.add_argument("--plot_max_points", type=int, default=12000)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--freq_scale", type=float, default=1.0,
                        help="Multiply audio frequencies by this factor (e.g., 2.0 = one octave up, 0.5 = one octave down). Duration stays the same.")

    args = parser.parse_args()

    station = args.station
    starttime = args.starttime
    duration = args.duration
    component = args.component
    min_freq_filter = args.min_freq_filter
    max_freq_filter = args.max_freq_filter
    nw = args.nw
    fps = args.fps
    width = args.width
    height = args.height
    dpi = args.dpi
    cursor_color = args.cursor_color
    wave_color = args.wave_color
    bg_color = args.bg_color
    axis_label_size = args.axis_label_size
    tick_label_size = args.tick_label_size
    major_tick_spacing = args.major_tick_spacing
    plot_max_points = args.plot_max_points
    freq_scale = args.freq_scale

    if width % 2:  
        width  -= 1
        print(f"Width is odd, decrementing to {width}")
    if height % 2: 
        height -= 1
        print(f"Height is odd, decrementing to {height}")

    # Expose pitch setting to the inner function via closure
    global freq_scale_global
    freq_scale_global = freq_scale

    filename_out = f"tremor_waveform_movie_{station}_{component}_{starttime.strftime('%Y%m%d%H%M%S')}_{duration:.0f}s_freq_scale{freq_scale:.0f}.mp4"
    outpath = join(dirpath_out, filename_out)

    make_movie_from_hdf5(
        station=station,
        starttime=starttime,
        duration=duration,
        component=component,
        min_freq_filter=min_freq_filter,
        max_freq_filter=max_freq_filter,
        nw=nw,
        fps=fps,
        width=width,
        height=height,
        dpi=dpi,
        outpath=outpath,
        cursor_color=cursor_color,
        wave_color=wave_color,
        bg_color=bg_color,
        axis_label_size=axis_label_size,
        tick_label_size=tick_label_size,
        major_tick_spacing=major_tick_spacing,
        plot_max_points=plot_max_points,
    )



if __name__ == "__main__":
    main()
