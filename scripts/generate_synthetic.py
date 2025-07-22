#!/usr/bin/env python3
# scripts/generate_synthetic.py

import os
import random
import argparse
import yaml
from pydub import AudioSegment

from config import load_config
from utils import (
    ensure_dir,
    load_random_clip,
    duck_overlay,
    fade_transition,
    overlay_noise,
)

# Загрузка путей из config.yaml
cfg = load_config()
PREP_SPEECH_DIR = cfg["paths"]["prepared_speech"]
PREP_MUSIC_DIR  = cfg["paths"]["prepared_music"]
PREP_NOISE_DIR  = cfg["paths"]["prepared_noise"]
SCENARIOS_DIR   = cfg["paths"]["scenarios"]
OUTPUT_DIR      = cfg["paths"]["output"]

def load_scenarios(dir_path):
    """Читает все YAML-сценарии из папки."""
    scenarios = []
    for fn in os.listdir(dir_path):
        if fn.lower().endswith((".yaml", ".yml")):
            path = os.path.join(dir_path, fn)
            with open(path, encoding="utf-8") as f:
                scenarios.append(yaml.safe_load(f))
    return scenarios

def parse_val(raw, default):
    """
    Если raw — список [min, max], возвращает random.uniform(min, max).
    Если raw задан как число, возвращает float(raw).
    Если raw is None, возвращает default.
    """
    if raw is None:
        return default
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return random.uniform(raw[0], raw[1])
    return float(raw)

def make_example(scenario, noise_prob):
    out = AudioSegment.silent(0)
    timeline = []
    t0 = 0.0

    for step in scenario["sequence"]:
        tp      = step["type"]
        dur     = parse_val(step.get("duration"), 0.0)
        gain_db = parse_val(step.get("gain_db"), 0.0)

        if tp == "speech":
            clip = load_random_clip(PREP_SPEECH_DIR, dur) + gain_db
            segs = [("speech", t0, t0 + dur)]

        elif tp == "background_music":
            duck_db = parse_val(step.get("duck_db"), random.uniform(7,18))
            sp = load_random_clip(PREP_SPEECH_DIR, dur) + gain_db
            mu = load_random_clip(PREP_MUSIC_DIR,  dur)
            clip = duck_overlay(sp, mu, duck_db)
            segs = [("speech", t0, t0 + dur), ("music", t0, t0 + dur)]

        elif tp == "music":
            clip = load_random_clip(PREP_MUSIC_DIR, dur) + gain_db
            segs = [("music", t0, t0 + dur)]

        elif tp == "fade_to_music":
            fade_ms = int(parse_val(step.get("fade_ms"), 500))
            duck_db = parse_val(step.get("duck_db"), random.uniform(7,18))
            sp = load_random_clip(PREP_SPEECH_DIR, dur) + gain_db
            mu = load_random_clip(PREP_MUSIC_DIR,  dur)
            clip = fade_transition(sp, mu, fade_ms)
            segs = [("speech", t0, t0 + dur), ("music", t0, t0 + dur)]

        elif tp == "noise":
            clip = load_random_clip(PREP_NOISE_DIR, dur)
            segs = [("noise", t0, t0 + dur)]

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

    # опциональный фоновый шум
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
        yaml_path = os.path.join(OUTPUT_DIR, base + ".yaml")

        # Сохраняем аудио
        audio.export(wav_path, format="wav")

        # Преобразуем аннотации в список словарей и сохраняем в YAML
        segments = [
            {"label": s["label"], "start": s["start"], "end": s["end"]}
            for s in ann
        ]
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump({"segments": segments},
                      f,
                      sort_keys=False,
                      allow_unicode=True)

        if i % 100 == 0:
            print(f"{i} examples generated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic audio dataset with YAML annotations"
    )
    parser.add_argument(
        "--n", type=int, default=10000,
        help="Number of examples to generate"
    )
    parser.add_argument(
        "--noise_prob", type=float, default=0.3,
        help="Probability to overlay background noise"
    )
    args = parser.parse_args()
    main(args.n, args.noise_prob)
