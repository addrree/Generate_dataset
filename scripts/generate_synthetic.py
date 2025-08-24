import os
import random
import argparse
import yaml
from pydub import AudioSegment
from tqdm import tqdm

from .config import load_config
from .utils import (
    ensure_dir,
    load_random_clip,
    duck_overlay,
    fade_transition,
    overlay_noise,
)

# Загрузка путей из config.yaml
cfg = load_config()
PREP_SPEECH_DIR = cfg["paths"]["prepared_speech"]
PREP_MUSIC_DIR = cfg["paths"]["prepared_music"]
PREP_NOISE_DIR = cfg["paths"]["prepared_noise"]
SCENARIOS_DIR = cfg["paths"]["scenarios"]
OUTPUT_DIR = cfg["paths"]["output"]


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
        tp = step["type"]
        dur = parse_val(step.get("duration"), 0.0)
        gain_db = parse_val(step.get("gain_db"), 0.0)
        xfade_ms = int(parse_val(step.get("crossfade_ms", step.get("fade_ms", 0)), 0))
        
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
            sp = load_random_clip(PREP_SPEECH_DIR, dur) + gain_db
            mu = load_random_clip(PREP_MUSIC_DIR,  dur)

          
            clip = fade_transition(sp, mu, xfade_ms)

            # Геометрия кроссфейда
            sp_ms = len(sp)
            mu_ms = len(mu)
            overlap_start_ms = sp_ms - xfade_ms            
            step_len_ms = sp_ms + mu_ms - xfade_ms
            step_end = t0 + step_len_ms / 1000.0

            
            sp_tail = sp[-xfade_ms:].fade_out(xfade_ms)      
            mu_head = mu[:xfade_ms].fade_in(xfade_ms)        

           
            def first_crossing_offset_ms(speech_seg, music_seg, win_ms=20, hop_ms=10):
                n = len(speech_seg)
                if n <= 0:
                    return 0
             
                win = max(1, min(win_ms, n))
                hop = max(1, hop_ms)
                pos = 0
                while pos + win <= n:
                    sp_win = speech_seg[pos:pos+win].rms
                    mu_win = music_seg[pos:pos+win].rms
                  
                    if mu_win >= sp_win:
                        return pos
                    pos += hop
                return n  

            offset_ms = first_crossing_offset_ms(sp_tail, mu_head, win_ms=20, hop_ms=10)

           
            boundary = t0 + (overlap_start_ms + offset_ms) / 1000.0  

            
            speech_end = min(boundary, step_end)
            music_start = max(boundary, t0)

            segs = [
                ("speech", t0,         speech_end),
                ("music",  music_start, step_end),
            ]
            out += clip
            t0 = len(out) / 1000.0
            for lab, st, en in segs:
                if en > st:
                    timeline.append({"label": lab, "start": round(st, 3), "end": round(en, 3)})
            continue 

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
            
        t0 = len(out) / 1000.0

    # опциональный фоновый шум
    if random.random() < noise_prob:
        out = overlay_noise(out, PREP_NOISE_DIR)

    return out, timeline


def generate_synthetic(n, noise_prob):
    ensure_dir(OUTPUT_DIR)
    scenarios = load_scenarios(SCENARIOS_DIR)

    for i in tqdm(range(n), desc="Generating dataset"):
        sc = random.choice(scenarios)
        audio, ann = make_example(sc, noise_prob)
        base = f"example_{i:05}"
        wav_path = os.path.join(OUTPUT_DIR, base + ".wav")
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

    return OUTPUT_DIR
