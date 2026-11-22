"""Makes the pitch curve a note chart is cut from: run as its own process.

    python -m pikaraoke.lib.chart_maker ORIGINAL_AUDIO OUTPUT_JSON

The original recording's lead vocals are separated from the music with a
MelBand RoFormer model, then pitch-tracked with CREPE. Both are neural
networks run with PyTorch, on the graphics card when there is one (NVIDIA
through CUDA, AMD through ROCm), else the CPU.

This needs the chart add-on (pip install "pikaraoke[charts]"); the server
never imports it, but starts it as a subprocess, so the models' memory is all
handed back when a chart is done and a crash cannot take the server down.
The output is the pitch curve only; chart.py turns it into notes.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

SEPARATION_MODEL = "vocals_mel_band_roformer.ckpt"
SAMPLERATE = 16000
HOP_S = 0.01
# A singing voice: a low bass to a high soprano.
MIN_HZ = 65.0
MAX_HZ = 1100.0
# Vocals quieter than this under the stem's loud parts are the separator's
# leftovers from the music, not singing.
_QUIET_DB = -35.0

# Frames pitch-tracked at once on the graphics card.
_PITCH_BATCH = 512

# Exit codes the server tells apart.
EXIT_NO_ADDON = 3


def available() -> bool:
    """Whether the add-on's libraries are installed, without importing them."""
    from importlib.util import find_spec

    return all(find_spec(m) for m in ("torch", "torchcrepe", "audio_separator"))


def describe_device() -> str:
    """What charts are made on, for the settings page."""
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def _device():
    import torch

    if not torch.cuda.is_available():
        return torch.device("cpu")
    # For testing that a chart fits a smaller card than this one.
    cap = os.environ.get("PIKARAOKE_CHART_MAX_VRAM_GB")
    if cap:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap) * 2**30 / total))
    return torch.device("cuda")


def separate_vocals(audio_path: str, work_dir: str, model_dir: str) -> str:
    """The lead vocals of a recording, as a WAV file in work_dir."""
    from audio_separator.separator import Separator

    separator = Separator(
        log_level=logging.WARNING,
        model_file_dir=model_dir,
        output_dir=work_dir,
        output_format="WAV",
        output_single_stem="Vocals",
        # Half precision: 6x faster on a graphics card, and the stem is as good.
        use_autocast=True,
    )
    separator.load_model(model_filename=SEPARATION_MODEL)
    outputs = separator.separate(audio_path)
    path = outputs[0]
    return path if os.path.isabs(path) else os.path.join(work_dir, path)


def load_mono(path: str, samplerate: int = SAMPLERATE) -> np.ndarray:
    """Any audio file as mono float samples, decoded by ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", str(samplerate)]
        + ["-f", "f32le", "-"],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32)


def track_pitch(vocals: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """(MIDI note or NaN, confidence 0-1) every HOP_S of the vocals."""
    import torch
    import torchcrepe

    hop = int(SAMPLERATE * HOP_S)
    # Every batch the same size: AMD cards tune their kernels for each input
    # shape they meet (minutes, once), so a last batch sized by the song's
    # length would be tuned afresh for every song.
    frames = 1 + len(vocals) // hop
    whole_batches = -(-frames // _PITCH_BATCH) * _PITCH_BATCH
    padded_audio = np.pad(vocals, (0, (whole_batches - 1) * hop - len(vocals) + 1))
    audio = torch.from_numpy(padded_audio).unsqueeze(0)
    frequency, periodicity = torchcrepe.predict(
        audio,
        SAMPLERATE,
        hop_length=hop,
        fmin=MIN_HZ,
        fmax=MAX_HZ,
        model="full",
        # Viterbi smoothing runs on the CPU and takes minutes; chart.py smooths instead.
        decoder=torchcrepe.decode.weighted_argmax,
        return_periodicity=True,
        batch_size=_PITCH_BATCH,
        device=device,
        pad=True,
    )
    periodicity = torchcrepe.filter.median(periodicity, 3)
    frequency = frequency.squeeze(0).cpu().numpy()[:frames]
    confidence = periodicity.squeeze(0).cpu().numpy()[:frames]

    # Loudness per frame: separation leaves faint traces of the music behind.
    padded = np.pad(vocals, (0, max(0, frames * hop - len(vocals))))[: frames * hop]
    rms = np.sqrt(np.mean(padded.reshape(frames, hop) ** 2, axis=1) + 1e-12)
    loud = 20 * np.log10(rms / (np.percentile(rms, 95) + 1e-12))
    confidence = np.where(loud > _QUIET_DB, confidence, 0.0)

    midi = 69 + 12 * np.log2(np.maximum(frequency, 1e-6) / 440)
    return np.where(confidence > 0, midi, np.nan), confidence


def make(audio_path: str, out_path: str, model_dir: str, keep_vocals: str | None = None) -> None:
    device = _device()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="pikaraoke-chart-") as work:
        # The separator reads WAV reliably; YouTube audio comes as Opus or AAC.
        wav = os.path.join(work, "original.wav")
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                audio_path,
                "-ac",
                "2",
                "-ar",
                "44100",
                wav,
            ],
            check=True,
        )
        vocals_path = separate_vocals(wav, work, model_dir)
        separated = time.monotonic()
        vocals = load_mono(vocals_path)
        if keep_vocals:
            os.replace(vocals_path, keep_vocals)
    _free(device)
    midi, confidence = track_pitch(vocals, device)
    logging.info(
        f"Chart: separated in {separated - started:.0f}s, "
        f"pitch-tracked in {time.monotonic() - separated:.0f}s on {device}"
        + (f", peak {_peak_gb(device):.1f} GB" if device.type == "cuda" else "")
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "hop": HOP_S,
                "midi": [None if np.isnan(m) else round(float(m), 2) for m in midi],
                "confidence": [round(float(c), 2) for c in confidence],
            },
            f,
        )


def _peak_gb(device) -> float:
    import torch

    return torch.cuda.max_memory_reserved(device) / 2**30


def _free(device) -> None:
    """Hand the separator's graphics memory back before the pitch tracker loads."""
    import gc

    import torch

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio")
    parser.add_argument("output")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--keep-vocals", help="also save the separated vocals here")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    if not available():
        logging.error('The chart add-on is not installed: pip install "pikaraoke[charts]"')
        return EXIT_NO_ADDON
    make(args.audio, args.output, args.model_dir, args.keep_vocals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
