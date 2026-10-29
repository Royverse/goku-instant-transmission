# Goku Instant Transmission for Microsoft Teams & Zoom

Real-time **Dragon Ball Z Instant Transmission (Option A: Vanish In-Place)** virtual webcam filter powered by **OpenCV**, **MediaPipe Tasks**, **`pyvirtualcam`**, and **Pygame**.

When you place **two fingers (Index + Middle) on your forehead**, your character triggers the classic DBZ horizontal scanline/speed-line distortion, flashes with anime energy, plays the *"shwip"* sound effect, and vanishes from your chair—leaving behind your empty room on Microsoft Teams or Zoom until you perform the pose again to teleport back in!

## Quick Start

Double-click **[run.bat](file:///C:/Users/User/.gemini/antigravity/scratch/goku-instant-transmission/run.bat)** or run in PowerShell:

```powershell
cd C:\Users\User\.gemini\antigravity\scratch\goku-instant-transmission
uv run python main.py
```

## Controls & Usage

1. **Capture Your Empty Chair Background (Recommended for Seamless Vanishing):**
   - Press **`B`** on the preview window to start a **3-second countdown**, lean out of the frame for a moment so it snaps your empty chair, and sit back down.
   - *(Even if you skip this step, the AI Selfie Segmenter continuously learns your background around your silhouette and inpaints any gaps automatically!)*
2. **Teleport Out / Teleport Back In:**
   - **Gesture:** Hold your **Index + Middle fingers** together and touch your **forehead** (where the yellow target dots appear on your local preview).
   - **Keyboard Fallback:** Press **`SPACEBAR`** anytime to trigger Instant Transmission manually.
3. **Other Keys:**
   - **`N`**: Instantly snap the current frame as the empty background.
   - **`H`**: Toggle the local debug HUD (Note: the HUD is **never** sent to Teams/Zoom—Teams/Zoom only sees your clean video feed!).
   - **`Q` or `ESC`**: Quit.

## Connecting to Microsoft Teams & Zoom

1. Make sure **OBS Studio** is installed on your machine (already detected at `C:\Program Files\obs-studio`).
2. Start this script (`uv run python main.py`) **before** selecting your camera in Teams/Zoom so it claims your physical webcam and outputs to **`OBS Virtual Camera`**.
3. In **Microsoft Teams** or **Zoom**, go to **Settings → Video → Camera** and select **`OBS Virtual Camera`**.
4. *(Optional Custom Sound)*: Place any custom MP3 named `instant_transmission.mp3` in the `assets/` folder to override the built-in synthesized DBZ sound effect.
