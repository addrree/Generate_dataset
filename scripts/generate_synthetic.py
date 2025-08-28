import os
import random
import argparse
import yaml
from pydub import AudioSegment
from tqdm import tqdm
import math
from typing import List, Dict, Any

from .config import load_config
from utils import (
    ensure_dir,
    load_random_clip,
    duck_overlay,
    fade_transition,
    overlay_noise,
    load_random_music_clip,
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



def allocate_durations_ms(sequence: List[Dict[str, Any]],
                          total_dur_s: float,
                          weight_sigma: float,
                          dmin_s: float,
                          dmax_s: float) -> List[int]:
    """
    Возвращает список длительностей в МИЛЛИСЕКУНДАХ для каждого шага sequence так,
    чтобы сумма == target_ms.
    """
    n = len(sequence)
    out_ms = [0] * n

    target_ms = int(round(max(0.0, total_dur_s) * 1000))
    out_ms = [0] * len(sequence)
    flex_idx = list(range(len(sequence)))
    remain_ms = target_ms

    if remain_ms == 0 or not flex_idx:
        return out_ms

    weights = []
    for i in flex_idx:
        step = sequence[i]
        mult = float(step.get("dur_mult", 1.0))
        base = max(1e-4, mult)
        mu = math.log(base)
        sigma = max(1e-6, float(weight_sigma))
        w = random.lognormvariate(mu, sigma)
        weights.append(w)
    sw = sum(weights) or 1.0

    alloc_ms = []
    acc = 0
    for w in weights:
        ms = int(remain_ms * (w / sw))
        alloc_ms.append(ms); acc += ms
    leftover = remain_ms - acc
    j = 0
    while leftover > 0:
        alloc_ms[j % len(alloc_ms)] += 1
        leftover -= 1
        j += 1

    # 5) применим min/max на каждый гибкий шаг
    dmin_ms = int(max(0.0, dmin_s) * 1000)
    dmax_ms = int(max(0.0, dmax_s) * 1000) if dmax_s is not None else None

    for k, i in enumerate(flex_idx):
        ms = alloc_ms[k]
        if ms < dmin_ms:
            ms = dmin_ms
        if dmax_ms is not None and ms > dmax_ms:
            ms = dmax_ms
        out_ms[i] = ms

    # 6) нормализация суммы (строго к target_ms): подгоняем последний гибкий шаг
    used_ms = sum(out_ms[i] for i in flex_idx)
    delta = target_ms - used_ms
    if delta != 0 and flex_idx:
        last = flex_idx[-1]
        adj = out_ms[last] + delta
        out_ms[last] = max(0, adj)

    return out_ms
def make_example(scenario, noise_prob,
                 total_dur: float | None,
                 weight_sigma: float,
                 dur_min: float,
                 dur_max: float):
    """
    Если total_dur задан — заранее распределяем длительности шагов в МИЛЛИСЕКУНДАХ так,
    чтобы итоговая сумма равнялась total_dur
    """
    out = AudioSegment.silent(0)
    timeline = []
    t0 = 0.0

    seq = scenario["sequence"]

    pre_ms = None 
    if total_dur is not None:
        pre_ms = allocate_durations_ms(seq, total_dur,
                                       weight_sigma=weight_sigma,
                                       dmin_s=dur_min, dmax_s=dur_max)

    for idx, step in enumerate(seq):
        tp       = step["type"]
        gain_db  = parse_val(step.get("gain_db"), 0.0)
        xfade_ms = int(parse_val(step.get("crossfade_ms", step.get("fade_ms", 0)), 0))

        if pre_ms is not None:
            dur_ms = int(pre_ms[idx])
            dur_s  = dur_ms / 1000.0
        else:
            dur_s = parse_val(step.get("duration"), 0.0)
            dur_ms = int(round(dur_s * 1000))

        if tp == "speech":
            clip = load_random_clip(PREP_SPEECH_DIR, dur_s) + gain_db
            segs = [("speech", t0, t0 + len(clip)/1000.0)]

        elif tp == "music":
            clip = load_random_music_clip(PREP_MUSIC_DIR, dur_s) + gain_db
            segs = [("music", t0, t0 + len(clip)/1000.0)]

        elif tp == "background_music":
            duck_db = parse_val(step.get("duck_db"), random.uniform(7, 18))
            sp = load_random_clip(PREP_SPEECH_DIR, dur_s) + gain_db
            mu = load_random_music_clip(PREP_MUSIC_DIR,  dur_s)
            clip = duck_overlay(sp, mu, duck_db)
            L = len(clip)/1000.0
            segs = [("speech", t0, t0 + L), ("music", t0, t0 + L)]

        elif tp in ("fade_music", "fade_to_music"):
            # duration из YAML трактуем как ОБЩУЮ длину шага (в ms при total_dur)
            if pre_ms is not None:
                # уже задано как общая длина в ms
                total_ms = max(0, int(pre_ms[idx]))
            else:
                total_ms = max(0, int(round(parse_val(step.get("duration"), 0.0) * 1000)))

            # делим строго в ms, чтобы len(sp)+len(mu) == total_ms
            sp_ms = total_ms // 2
            mu_ms = total_ms - sp_ms

            sp = load_random_clip(PREP_SPEECH_DIR, sp_ms / 1000.0) + gain_db
            mu = load_random_music_clip(PREP_MUSIC_DIR,  mu_ms / 1000.0)

            clip = fade_transition(sp, mu, xfade_ms)  # длина = len(sp)+len(mu) == total_ms

            step_end   = t0 + len(clip)/1000.0
            speech_end = t0 + len(sp)/1000.0

            # «слышимый старт» музыки по голове fade-in
            head_ms = min(xfade_ms, len(mu))
            mu_head = mu[:head_ms].fade_in(xfade_ms)

            def first_audible_offset_ms(seg: AudioSegment, win_ms=30, hop_ms=5, frac=0.2):
                n = len(seg)
                if n <= 0: return 0
                win = max(1, min(win_ms, n))
                hop = max(1, hop_ms)
                # максимум RMS
                vals, pos = [], 0
                while pos + win <= n:
                    vals.append(seg[pos:pos+win].rms)
                    pos += hop
                vmax = max(vals) or 1
                thr = frac * vmax
                # первая точка >= thr (центр окна)
                pos = 0
                while pos + win <= n:
                    if seg[pos:pos+win].rms >= thr:
                        return pos + win // 2
                    pos += hop
                return n

            offset_ms = first_audible_offset_ms(mu_head, win_ms=30, hop_ms=5, frac=0.18)
            music_start = speech_end + offset_ms/1000.0
            if music_start > step_end:
                music_start = step_end

            segs = [
                ("speech", t0,          speech_end),
                ("music",  music_start,  step_end),
            ]

            out += clip
            t0 = len(out)/1000.0
            for lab, st, en in segs:
                if en > st:
                    timeline.append({"label": lab, "start": round(st, 3), "end": round(en, 3)})
            continue

        elif tp == "noise":
            clip = load_random_clip(PREP_NOISE_DIR, dur_s)
            segs = [("noise", t0, t0 + len(clip)/1000.0)]

        else:
            continue

        out += clip
        for label, start, end in segs:
            timeline.append({"label": label, "start": round(start, 3), "end": round(end, 3)})
        t0 = len(out)/1000.0

    if random.random() < noise_prob:
        out = overlay_noise(out, PREP_NOISE_DIR)

    total_len = round(len(out)/1000.0, 3)

    norm = []
    for s in timeline:
        st = max(0.0, round(s["start"], 3))
        en = min(total_len, round(s["end"],   3))
        if en > st:
            norm.append({"label": s["label"], "start": st, "end": en})
    norm.sort(key=lambda x: (x["start"], x["end"], x["label"]))

    merged = []
    for s in norm:
        if merged and merged[-1]["label"] == s["label"] and s["start"] <= merged[-1]["end"] + 1e-3:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append(s)

    if merged:
        merged[-1]["end"] = total_len

    return out, merged

def main(n, noise_prob, total_dur, weight_sigma, dur_min, dur_max):
    ensure_dir(OUTPUT_DIR)
    scenarios = load_scenarios(SCENARIOS_DIR)

    for i in range(n):
        sc = random.choice(scenarios)
        audio, ann = make_example(sc, noise_prob, total_dur, weight_sigma, dur_min, dur_max)
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
