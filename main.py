"""
Goku Instant Transmission (Option A: Vanish In-Place) for Microsoft Teams & Zoom
---------------------------------------------------------------------------------
Tech Stack:
- OpenCV (webcam capture + DBZ horizontal speed-line/scanline displacement VFX)
- MediaPipe Tasks (FaceLandmarker for forehead tracking, HandLandmarker for 2-finger pose,
  ImageSegmenter for cleanly slicing only your body during teleportation & auto-capturing background)
- pyvirtualcam (routes processed frames directly to OBS Virtual Camera for Teams & Zoom)
- Pygame (plays the iconic Instant Transmission sound effect)
"""

import argparse
import math
import pathlib
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

from generate_sfx import ensure_sound_file

# Optional imports with graceful fallback
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import pyvirtualcam
    VIRTUALCAM_AVAILABLE = True
except ImportError:
    VIRTUALCAM_AVAILABLE = False


BASE_DIR = pathlib.Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"


class InstantTransmissionApp:
    # State constants
    STATE_PRESENT = "PRESENT"
    STATE_TELEPORTING_OUT = "TELEPORTING_OUT"
    STATE_VANISHED = "VANISHED"
    STATE_TELEPORTING_IN = "TELEPORTING_IN"

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720, enable_vcam: bool = True):
        self.camera_index = camera_index
        self.req_width = width
        self.req_height = height
        self.enable_vcam = enable_vcam

        self.state = self.STATE_PRESENT
        self.anim_start_time = 0.0
        self.anim_duration = 0.42  # Duration of the DBZ scanline/flash animation in seconds
        self.cooldown_until = 0.0  # Prevents accidental double-triggers
        self.pose_hold_frames = 0
        self.required_hold_frames = 4  # Require ~4 consecutive frames (~130ms) of the pose to trigger reliably

        self.show_hud = True  # Show debug landmarks & instructions on local preview ONLY (never on Teams/Zoom!)
        self.clean_bg: np.ndarray | None = None
        self.rolling_bg: np.ndarray | None = None

        self._init_audio()
        self._init_mediapipe()

    def _init_audio(self) -> None:
        self.sfx = None
        if not PYGAME_AVAILABLE:
            return
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            # Check for custom user MP3/WAV first, otherwise synthesize default WAV
            custom_mp3 = ASSETS_DIR / "instant_transmission.mp3"
            wav_path = ASSETS_DIR / "instant_transmission.wav"
            sound_path = custom_mp3 if custom_mp3.exists() else ensure_sound_file(wav_path)
            self.sfx = pygame.mixer.Sound(str(sound_path))
        except Exception as exc:
            print(f"[Audio Warning] Could not initialize sound effect: {exc}")

    def play_sfx(self) -> None:
        if self.sfx is not None:
            try:
                self.sfx.play()
            except Exception:
                pass

    def _ensure_models(self) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_urls = {
            "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            "selfie_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
        }
        import urllib.request
        for filename, url in model_urls.items():
            target = MODELS_DIR / filename
            if not target.exists():
                print(f"[Setup] Downloading MediaPipe model '{filename}'...")
                urllib.request.urlretrieve(url, target)
        return (
            MODELS_DIR / "hand_landmarker.task",
            MODELS_DIR / "face_landmarker.task",
            MODELS_DIR / "selfie_segmenter.tflite",
        )

    def _init_mediapipe(self) -> None:
        hand_model, face_model, seg_model = self._ensure_models()

        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(hand_model)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.55,
                min_hand_presence_confidence=0.55,
                min_tracking_confidence=0.55,
            )
        )

        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(face_model)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        self.segmenter = mp_vision.ImageSegmenter.create_from_options(
            mp_vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(seg_model)),
                running_mode=mp_vision.RunningMode.VIDEO,
                output_category_mask=False,
                output_confidence_masks=True,
            )
        )

    @staticmethod
    def _dist_2d(p1, p2) -> float:
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def detect_goku_pose(
        self, frame_rgb: np.ndarray, timestamp_ms: int
    ) -> tuple[bool, dict]:
        """
        Detects Goku's Instant Transmission pose:
        1. Locates forehead landmarks from FaceLandmarker (glabella #9, mid-forehead #151, upper forehead #10).
        2. Checks if Index (#8) and Middle (#12) fingers are extended and close together.
        3. Checks if Index or Middle fingertip is touching / near the forehead region.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        seg_result = self.segmenter.segment_for_video(mp_image, timestamp_ms)

        person_mask = None
        if seg_result.confidence_masks:
            person_mask = np.squeeze(seg_result.confidence_masks[0].numpy_view()).copy()

        debug_info = {
            "forehead_pts": [],
            "fingertip_pt": None,
            "two_fingers_valid": False,
            "distance": None,
            "threshold": 0.12,
            "person_mask": person_mask,
        }

        if not face_result.face_landmarks:
            return False, debug_info

        face_lms = face_result.face_landmarks[0]
        # Landmarks 9 (glabella), 151 (center forehead), 10 (top forehead), 67 & 297 (left/right forehead temples)
        forehead_indices = [9, 151, 10, 67, 297]
        forehead_pts = [face_lms[idx] for idx in forehead_indices]
        debug_info["forehead_pts"] = [(pt.x, pt.y) for pt in forehead_pts]

        # Scale threshold dynamically by face width (temple to temple distance) so it works at any distance from camera
        left_cheek = face_lms[234]
        right_cheek = face_lms[454]
        face_width = max(0.08, self._dist_2d(left_cheek, right_cheek))
        touch_threshold = face_width * 0.65
        debug_info["threshold"] = touch_threshold

        if not hand_result.hand_landmarks:
            return False, debug_info

        for hand_lms in hand_result.hand_landmarks:
            wrist = hand_lms[0]
            idx_mcp, idx_pip, idx_tip = hand_lms[5], hand_lms[6], hand_lms[8]
            mid_mcp, mid_pip, mid_tip = hand_lms[9], hand_lms[10], hand_lms[12]
            ring_pip, ring_tip = hand_lms[14], hand_lms[16]
            pinky_pip, pinky_tip = hand_lms[18], hand_lms[20]

            # Check that index and middle fingers are extended (tip further from wrist than PIP joint)
            idx_extended = self._dist_2d(wrist, idx_tip) > self._dist_2d(wrist, idx_pip) * 1.08
            mid_extended = self._dist_2d(wrist, mid_tip) > self._dist_2d(wrist, mid_pip) * 1.08

            # Index and middle fingertips should be held together (like Goku's two-finger salute)
            fingers_together = self._dist_2d(idx_tip, mid_tip) < face_width * 0.48

            # Ring and pinky should be more curled than index/middle
            ring_curled = self._dist_2d(wrist, ring_tip) < self._dist_2d(wrist, idx_tip) * 0.92
            pinky_curled = self._dist_2d(wrist, pinky_tip) < self._dist_2d(wrist, mid_tip) * 0.92

            if idx_extended and mid_extended and fingers_together and (ring_curled or pinky_curled):
                debug_info["two_fingers_valid"] = True
                # Midpoint between index and middle fingertips
                tip_x = (idx_tip.x + mid_tip.x) * 0.5
                tip_y = (idx_tip.y + mid_tip.y) * 0.5
                debug_info["fingertip_pt"] = (tip_x, tip_y)

                # Minimum distance to any forehead anchor point
                min_dist = min(
                    math.hypot(tip_x - fpt.x, tip_y - fpt.y)
                    for fpt in forehead_pts
                )
                debug_info["distance"] = min_dist

                if min_dist <= touch_threshold:
                    return True, debug_info

        return False, debug_info

    def update_background_estimate(self, frame_bgr: np.ndarray, person_mask: np.ndarray | None) -> None:
        """
        Continuously learns the background pixels wherever your body ISN'T, so even if you didn't step
        out of frame to press 'B', it gradually reconstructs a clean background plate!
        """
        if self.rolling_bg is None:
            self.rolling_bg = frame_bgr.astype(np.float32)
            return

        if person_mask is not None:
            # Update only pixels where person confidence is very low (< 0.15)
            bg_region = (person_mask < 0.15)[..., np.newaxis]
            self.rolling_bg = np.where(
                bg_region,
                0.92 * self.rolling_bg + 0.08 * frame_bgr.astype(np.float32),
                self.rolling_bg,
            )

    def get_background_frame(self, fallback_frame: np.ndarray, person_mask: np.ndarray | None) -> np.ndarray:
        """Returns the captured empty background frame (or inpainted fallback if none was captured yet)."""
        if self.clean_bg is not None:
            return self.clean_bg.copy()
        if self.rolling_bg is not None:
            bg = self.rolling_bg.astype(np.uint8)
            # If the user never stepped out of frame, inpaint any remaining silhouette artifacts
            if person_mask is not None:
                fg_mask = (person_mask > 0.35).astype(np.uint8) * 255
                fg_mask = cv2.dilate(fg_mask, np.ones((15, 15), np.uint8), iterations=1)
                bg = cv2.inpaint(bg, fg_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
            return bg
        return fallback_frame.copy()

    def render_dbz_teleport_vfx(
        self,
        live_frame: np.ndarray,
        bg_frame: np.ndarray,
        person_mask: np.ndarray | None,
        progress: float,
        appearing: bool = False,
    ) -> np.ndarray:
        """
        Renders the authentic Dragon Ball Z Instant Transmission animation:
        - Phase 1: High-contrast aura flash + horizontal scanline slicing on the character silhouette
        - Phase 2: Rapid horizontal speed-line displacement ("shwip" comb effect)
        - Phase 3: Clean cut to empty background (or character if appearing)
        """
        # If appearing, reverse the visual progression
        p = 1.0 - progress if appearing else progress
        h, w, _ = live_frame.shape
        out = bg_frame.copy()

        if person_mask is None:
            mask_3c = np.ones((h, w, 3), dtype=np.float32)
        else:
            mask_resized = cv2.resize(person_mask, (w, h))
            mask_3c = np.clip(mask_resized[..., np.newaxis], 0.0, 1.0)

        # Create horizontal scanline comb (classic DBZ black/white teleport lines)
        stripe_height = max(2, int(6 - 3 * p))
        y_indices = np.arange(h)[:, np.newaxis]
        # Keep fewer horizontal slices as p approaches 1.0
        keep_prob = max(0.05, 1.0 - p * 1.15)
        scanline_pattern = (((y_indices // stripe_height) % 3) == 0).astype(np.float32)

        # Horizontal slice jitter/displacement on the character
        displaced_person = live_frame.copy()
        num_slices = 18
        slice_h = h // num_slices
        rng = np.random.default_rng(int(progress * 1000))
        for i in range(num_slices):
            y1 = i * slice_h
            y2 = h if i == num_slices - 1 else (i + 1) * slice_h
            shift = int(rng.integers(-65, 65) * p)
            displaced_person[y1:y2] = np.roll(displaced_person[y1:y2], shift, axis=1)

        # Brighten character silhouette toward white/cyan anime energy during mid-teleport
        energy_tint = np.full_like(displaced_person, (255, 250, 220), dtype=np.uint8)
        glow_strength = math.sin(p * math.pi) * 0.75
        energized_person = cv2.addWeighted(
            displaced_person, 1.0 - glow_strength, energy_tint, glow_strength, 0
        )

        # Combine scanline comb with person mask
        effective_alpha = mask_3c * scanline_pattern[..., np.newaxis] * keep_prob
        out = (energized_person * effective_alpha + out * (1.0 - effective_alpha)).astype(np.uint8)

        # Draw crisp black & white horizontal DBZ speed lines across the character bounding region
        if person_mask is not None and 0.1 < p < 0.9:
            ys, xs = np.where(person_mask > 0.5)
            if len(xs) > 0 and len(ys) > 0:
                y_min, y_max = int(np.min(ys)), int(np.max(ys))
                x_min, x_max = int(np.min(xs)), int(np.max(xs))
                cx = (x_min + x_max) // 2
                num_lines = int(28 * (1.0 - abs(p - 0.5)))
                for _ in range(num_lines):
                    ly = int(rng.integers(y_min, max(y_min + 1, y_max)))
                    half_len = int(rng.integers(30, max(35, (x_max - x_min) // 2 + 40)) * (1.0 - p * 0.5))
                    color = (255, 255, 255) if rng.random() > 0.4 else (15, 15, 15)
                    thickness = 2 if rng.random() > 0.5 else 1
                    cv2.line(out, (max(0, cx - half_len), ly), (min(w - 1, cx + half_len), ly), color, thickness)

        # Subtle initial impact frame flash (first 18% of animation)
        if progress < 0.18:
            flash_alpha = (1.0 - (progress / 0.18)) * 0.55
            white_frame = np.full_like(out, 255)
            out = cv2.addWeighted(out, 1.0 - flash_alpha, white_frame, flash_alpha, 0)

        return out

    def trigger_teleport(self, current_frame: np.ndarray, person_mask: np.ndarray | None) -> None:
        now = time.perf_counter()
        if now < self.cooldown_until:
            return

        if self.state == self.STATE_PRESENT:
            # Snapshot or build the background frame before vanishing
            if self.clean_bg is None:
                self.clean_bg = self.get_background_frame(current_frame, person_mask)
            self.state = self.STATE_TELEPORTING_OUT
            self.anim_start_time = now
            self.cooldown_until = now + 1.1
            self.play_sfx()
            print("[Instant Transmission] SHWIP! Vanishing from chair...")

        elif self.state == self.STATE_VANISHED:
            self.state = self.STATE_TELEPORTING_IN
            self.anim_start_time = now
            self.cooldown_until = now + 1.1
            self.play_sfx()
            print("[Instant Transmission] SHWIP! Teleported back into chair!")

    def draw_local_preview_hud(
        self, preview_frame: np.ndarray, debug_info: dict, vcam_device: str | None
    ) -> np.ndarray:
        """Draws helpful status & forehead target on the local monitor window (NOT sent to Teams/Zoom)."""
        if not self.show_hud:
            return preview_frame

        hud = preview_frame.copy()
        h, w, _ = hud.shape

        # Draw forehead target points
        for fx, fy in debug_info.get("forehead_pts", []):
            cv2.circle(hud, (int(fx * w), int(fy * h)), 4, (0, 230, 255), -1)

        # Draw fingertip tracker if two-finger pose is active
        fingertip = debug_info.get("fingertip_pt")
        if fingertip is not None:
            tx, ty = int(fingertip[0] * w), int(fingertip[1] * h)
            color = (0, 255, 0) if debug_info.get("distance", 1.0) <= debug_info.get("threshold", 0.1) else (0, 165, 255)
            cv2.circle(hud, (tx, ty), 8, color, 2)

        # Top status banner
        overlay = hud.copy()
        cv2.rectangle(overlay, (12, 12), (560, 128), (18, 18, 24), -1)
        cv2.addWeighted(overlay, 0.72, hud, 0.28, 0, hud)

        state_color = (80, 240, 120) if self.state == self.STATE_PRESENT else (0, 215, 255)
        cv2.putText(
            hud,
            f"MODE: {self.state}",
            (26, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            state_color,
            2,
            cv2.LINE_AA,
        )

        bg_status = "CAPTURED (Ready)" if self.clean_bg is not None else "AUTO-LEARNING (Press 'B' when out of frame for best quality)"
        cv2.putText(
            hud,
            f"Empty Chair BG: {bg_status}",
            (26, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        vcam_label = f"Streaming -> {vcam_device}" if vcam_device else "Local Preview Only (Start OBS Virtual Cam if needed)"
        cv2.putText(
            hud,
            f"Teams/Zoom Output: {vcam_label}",
            (26, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            hud,
            "[2 Fingers on Forehead] or [SPACE]: Teleport | [B]: 3s Empty BG Timer | [H]: Hide HUD | [Q]: Quit",
            (26, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )

        return hud

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {self.camera_index}.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        vcam = None
        vcam_device = None
        if self.enable_vcam and VIRTUALCAM_AVAILABLE:
            try:
                vcam = pyvirtualcam.Camera(width=actual_w, height=actual_h, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
                vcam_device = vcam.device
                print(f"[Virtual Camera] Active on '{vcam_device}' ({actual_w}x{actual_h} @ 30fps)")
                print(" -> Select 'OBS Virtual Camera' in Microsoft Teams or Zoom video settings!")
            except Exception as exc:
                print(f"[Virtual Camera Note] Could not attach to virtual camera yet ({exc}).")
                print(" -> Running in local preview window. Install/initialize OBS Virtual Camera if needed.")

        bg_countdown_end = 0.0
        start_clock = time.perf_counter()

        print("\n=== GOKU INSTANT TRANSMISSION READY ===")
        print("1. Press 'B' to start a 3-second countdown, lean out of frame so it snaps your empty chair.")
        print("   (Or just start right away—AI body segmentation will auto-estimate your background!)")
        print("2. Put two fingers (Index + Middle) to your forehead to vanish / reappear!")
        print("3. Or press SPACEBAR anytime to trigger manually.\n")

        try:
            while True:
                ok, raw_frame = cap.read()
                if not ok:
                    break

                # Mirror horizontally so moving your hand feels natural
                live_frame = cv2.flip(raw_frame, 1)
                now = time.perf_counter()
                timestamp_ms = int((now - start_clock) * 1000)

                # Run MediaPipe pose + segmentation detection on the live camera feed
                frame_rgb = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
                pose_detected, debug_info = self.detect_goku_pose(frame_rgb, timestamp_ms)
                person_mask = debug_info.get("person_mask")

                # Update rolling background while present
                if self.state == self.STATE_PRESENT and bg_countdown_end == 0.0:
                    self.update_background_estimate(live_frame, person_mask)

                # Handle 3-second empty chair background capture countdown ('B' key)
                if bg_countdown_end > 0.0:
                    remaining = bg_countdown_end - now
                    if remaining <= 0.0:
                        self.clean_bg = live_frame.copy()
                        bg_countdown_end = 0.0
                        self.play_sfx()
                        print("[Setup] Clean empty background captured!")

                # Debounce gesture detection
                if pose_detected and bg_countdown_end == 0.0:
                    self.pose_hold_frames += 1
                    if self.pose_hold_frames >= self.required_hold_frames:
                        self.trigger_teleport(live_frame, person_mask)
                        self.pose_hold_frames = 0
                else:
                    self.pose_hold_frames = max(0, self.pose_hold_frames - 1)

                # State machine rendering for the clean output frame (what Teams/Zoom sees)
                bg_plate = self.get_background_frame(live_frame, person_mask)

                if self.state == self.STATE_PRESENT:
                    output_frame = live_frame.copy()

                elif self.state == self.STATE_TELEPORTING_OUT:
                    elapsed = now - self.anim_start_time
                    progress = min(1.0, elapsed / self.anim_duration)
                    output_frame = self.render_dbz_teleport_vfx(
                        live_frame, bg_plate, person_mask, progress, appearing=False
                    )
                    if progress >= 1.0:
                        self.state = self.STATE_VANISHED

                elif self.state == self.STATE_VANISHED:
                    # Show the empty room/chair!
                    output_frame = bg_plate.copy()

                elif self.state == self.STATE_TELEPORTING_IN:
                    elapsed = now - self.anim_start_time
                    progress = min(1.0, elapsed / self.anim_duration)
                    output_frame = self.render_dbz_teleport_vfx(
                        live_frame, bg_plate, person_mask, progress, appearing=True
                    )
                    if progress >= 1.0:
                        self.state = self.STATE_PRESENT

                # Send clean output_frame (NO HUD text!) to OBS Virtual Camera for Teams/Zoom
                if vcam is not None:
                    vcam.send(output_frame)

                # Build local preview window (shows HUD + countdown if active)
                preview_frame = self.draw_local_preview_hud(output_frame, debug_info, vcam_device)
                if bg_countdown_end > 0.0:
                    rem = max(0.0, bg_countdown_end - now)
                    cv2.putText(
                        preview_frame,
                        f"STEP OUT OF FRAME! Capturing empty chair in {rem:.1f}s...",
                        (40, actual_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 80, 255),
                        3,
                        cv2.LINE_AA,
                    )

                cv2.imshow("Goku Instant Transmission (Local Control Preview)", preview_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or ESC
                    break
                elif key == ord(" "):  # Manual trigger via Spacebar
                    self.trigger_teleport(live_frame, person_mask)
                elif key == ord("b"):  # 3-second countdown to snap clean empty background
                    bg_countdown_end = time.perf_counter() + 3.0
                    print("[Setup] Step out of frame! Capturing empty background in 3 seconds...")
                elif key == ord("n"):  # Instant snap of empty background
                    self.clean_bg = live_frame.copy()
                    print("[Setup] Instant background frame captured!")
                elif key == ord("h"):  # Toggle HUD on local preview
                    self.show_hud = not self.show_hud

        finally:
            cap.release()
            if vcam is not None:
                vcam.close()
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Goku Instant Transmission Virtual Camera for Teams & Zoom")
    parser.add_argument("--camera", type=int, default=0, help="Physical webcam index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Capture height (default: 720)")
    parser.add_argument("--no-vcam", action="store_true", help="Disable virtual camera output")
    args = parser.parse_args()

    app = InstantTransmissionApp(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        enable_vcam=not args.no_vcam,
    )
    app.run()


if __name__ == "__main__":
    main()
