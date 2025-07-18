# scripts/utils.py

import os
import random
from pydub import AudioSegment

def ensure_dir(path):
    """Создаёт папку, если её ещё нет."""
    os.makedirs(path, exist_ok=True)

def load_random_clip(folder: str, duration_s: float) -> AudioSegment:
    """
    Берёт случайный непустой .wav из папки и возвращает AudioSegment
    ровно duration_s секунд (лупит или обрезает).
    """
    candidates = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(".wav"):
                full = os.path.join(root, fn)
                if os.path.getsize(full) > 1024:
                    candidates.append(full)
    if not candidates:
        raise FileNotFoundError(f"No WAV files in {folder}")

    # Находим первый непустой файл
    while candidates:
        choice = random.choice(candidates)
        clip = AudioSegment.from_file(choice)
        if len(clip) > 0:
            break
        candidates.remove(choice)
    if len(clip) == 0:
        raise ValueError(f"All WAVs in {folder} are empty")

    # Лупим или обрезаем до нужной длины
    req_ms = int(duration_s * 1000)
    if len(clip) < req_ms:
        times = req_ms // len(clip) + 1
        clip = clip * times
    return clip[:req_ms]

def duck_overlay(speech: AudioSegment, music: AudioSegment, duck_db: float) -> AudioSegment:
    """Накладывает музыку под речь, понижая её уровень на duck_db дБ."""
    return speech.overlay(music - duck_db)

def fade_transition(a1: AudioSegment, a2: AudioSegment, fade_ms: int) -> AudioSegment:
    """Плавный переход: fade-out первой дорожки и fade-in второй."""
    return a1.fade_out(fade_ms) + a2.fade_in(fade_ms)

def overlay_noise(audio: AudioSegment, noise_folder: str, reduction_db_range=(15, 30)) -> AudioSegment:
    """
    Накладывает на весь audio фоновый шум из noise_folder:
      – шум лупится до длины audio,
      – понижается на random.uniform(*reduction_db_range) дБ,
      – накладывается сверху.
    """
    dur = audio.duration_seconds
    noise_clip = load_random_clip(noise_folder, dur)
    reduction = random.uniform(*reduction_db_range)
    return audio.overlay(noise_clip - reduction)
