"""
Synthesizes the iconic Dragon Ball Z Instant Transmission ("shwip" / "bzz-wip") sound effect
as a stereo WAV file in assets/instant_transmission.wav if not already present.
You can also replace assets/instant_transmission.wav with any custom MP3/WAV file!
"""

import pathlib
import wave
import numpy as np


def ensure_sound_file(output_path: pathlib.Path) -> pathlib.Path:
    if output_path.exists():
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 44100
    duration = 0.42  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # DBZ Instant Transmission sound has two layered components:
    # 1. Rapidly modulated metallic pitch sweep (high-to-low-to-high warble)
    # 2. Crisp high-frequency transient "shwip" snap
    lfo = np.sin(2 * np.pi * 55 * t) * 450
    base_freq = 1900 - 1200 * np.sin(np.pi * t / duration) + lfo
    phase = 2 * np.pi * np.cumsum(base_freq) / sample_rate

    # Harmonic rich synth wave
    tone = (
        0.55 * np.sin(phase)
        + 0.30 * np.sin(2.0 * phase)
        + 0.15 * np.sign(np.sin(3.0 * phase))
    )

    # Add a sharp frequency-chirp snap in the middle (at t ~ 0.12s)
    chirp_env = np.exp(-((t - 0.10) ** 2) / (2 * 0.035**2))
    chirp_freq = np.linspace(800, 4800, len(t))
    chirp = np.sin(2 * np.pi * np.cumsum(chirp_freq) / sample_rate) * chirp_env

    # Subtle white noise burst for air/energy displacement
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.25, len(t)) * np.exp(-t / 0.09)

    # Amplitude envelope: sharp attack, double pulse, fast decay
    envelope = np.exp(-t / 0.14) * (0.75 + 0.25 * np.cos(2 * np.pi * 18 * t))
    envelope[: int(0.008 * sample_rate)] *= np.linspace(0, 1, int(0.008 * sample_rate))

    signal = (0.65 * tone * envelope) + (0.45 * chirp) + (0.20 * noise)
    signal = np.clip(signal / (np.max(np.abs(signal)) + 1e-6), -1.0, 1.0)

    # Convert to 16-bit stereo PCM
    pcm = (signal * 30000).astype(np.int16)
    stereo = np.column_stack((pcm, pcm)).ravel()

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(stereo.tobytes())

    return output_path


if __name__ == "__main__":
    path = ensure_sound_file(pathlib.Path("assets/instant_transmission.wav"))
    print(f"Generated sound effect at: {path}")
