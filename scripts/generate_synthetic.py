#!/usr/bin/env python3
# scripts/generate_synthetic.py

import os
import random
import argparse
import yaml
import json
from pydub import AudioSegment

# Конфигурация и утилиты вынесены в отдельные модули:
from config import load_config
from utils  import (
    ensure_dir,
    load_random_clip,
    duck_overlay,
    fade_transition,
    overlay_noise,
)

# Загружаем все пути из config.yaml
cfg = load_config()
PREP_SPEECH_DIR = cfg["paths"]["prepared_speech"]
PREP_MUSIC_DIR  = cfg["paths"]["prepared_music"]
PREP_NOISE_DIR  = cfg["paths"]["prepared_noise"]
SCENARIOS_DIR   = cfg["paths"]["scenarios"]
OUTPUT_DIR      = cfg["paths"]["output"]

def load_scenarios(dir_path):
    """Считывает все .yaml/.yml-файлы сценариев из папки."""
    scenarios = []
    for fn in os.listdir(dir_path):
        if fn.lower().endswith((".yaml", ".yml")):
            with open(os.path.join(dir_path, fn), encoding="utf-8") as f:
                scenarios.append(yaml.safe_load(f))
    return scenarios

def make_example(scenario, noise_prob):
    """
    Собирает один пример:
      – склеивает сегменты по сценарию,
      – опционально накладывает фоновый шум,
      – возвращает готовый AudioSegment и аннотацию.
    """
    out = AudioSegment.silent(0)
    timeline = []
    t0 = 0.0

    for step in scenario["sequence"]:
        tp      = step["type"]
        dur     = step["duration"]
        gain_db = step.get("gain_db", 0.0)

        if tp == "speech":
            clip = load_random_clip(PREP_SPEECH_DIR, dur) + gain_db
            segs = [("speech", t0, t0+dur)]

        elif tp == "background_music":
            duck_db = step.get("duck_db", random.uniform(7,18))
            sp = load_random_clip(PREP_SPEECH_DIR, dur) + gain_db
            mu = load_random_clip(PREP_MUSIC_DIR,  dur)
            clip = duck_overlay(sp, mu, duck_db)
            segs = [("speech", t0, t0+dur),
                    ("music",  t0, t0+dur)]

        elif tp == "music":
            clip = load_random_clip(PREP_MUSIC_DIR, dur) + gain_db
            segs = [("music", t0, t0+dur)]

        elif tp == "fade_to_music":
            fade_ms = int(step.get("fade_ms", 500))
            sp = load_random_clip(PREP_SPEECH_DIR, dur) + gain_db
            mu = load_random_clip(PREP_MUSIC_DIR,  dur)
            clip = fade_transition(sp, mu, fade_ms)
            segs = [("speech", t0, t0+dur),
                    ("music",  t0, t0+dur)]

        elif tp == "noise":
            clip = load_random_clip(PREP_NOISE_DIR, dur)
            segs = [("noise", t0, t0+dur)]

        else:
            continue

        out += clip
        for label, start, end in segs:
            timeline.append({
                "label": label,
                "start": round(start, 3),
                "end":   round(end,   3),
            })
        t0 += dur

    # фоновой шум
    if random.random() < noise_prob:
        out = overlay_noise(out, PREP_NOISE_DIR)

    return out, timeline

def main(n, noise_prob):
    ensure_dir(OUTPUT_DIR)
    scenarios = load_scenarios(SCENARIOS_DIR)

    for i in range(n):
        sc = random.choice(scenarios)
        audio, ann = make_example(sc, noise_prob)
        base = f"example_{i:05}"
        wav_path  = os.path.join(OUTPUT_DIR, base + ".wav")
        json_path = os.path.join(OUTPUT_DIR, base + ".json")

        audio.export(wav_path, format="wav")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"segments": ann}, f, ensure_ascii=False, indent=2)

        if i % 100 == 0:
            print(f"{i} examples generated")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate synthetic audio dataset"
    )
    p.add_argument("--n",          type=int,   default=10000,
                   help="Number of examples to generate")
    p.add_argument("--noise_prob", type=float, default=0.3,
                   help="Probability to overlay background noise")
    args = p.parse_args()
    main(args.n, args.noise_prob)
