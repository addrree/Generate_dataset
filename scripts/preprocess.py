import os
import subprocess

from tqdm import tqdm

from config import load_config
from utils import ensure_dir

ERROR_LOG = "preprocess_errors.log"


def process_one(inp_path: str, out_path: str):
    """
    Конвертация одним FFmpeg-вызовом:
      - mono 16 kHz PCM
      - LUFS-нормализация (I=-23 LUFS)
      - обрезка тишины спереди/сзади
    """
    filters = (
        "loudnorm=I=-23:TP=-1:LRA=7,"
        "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB,"
        "areverse,"
        "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB,"
        "areverse"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-y", "-i", inp_path,
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        "-af", filters,
        "-c:a", "pcm_s16le",
        out_path
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        with open(ERROR_LOG, "a", encoding="utf-8") as logf:
            logf.write(f"{inp_path}\n")
        print(f"[ERROR] failed to process: {inp_path}")


def main():
    cfg = load_config()
    #ffmpeg = cfg["ffmpeg"]

    mapping = [
        (cfg["paths"]["raw_music"],    cfg["paths"]["prepared_music"]),
        (cfg["paths"]["raw_noise"],    cfg["paths"]["prepared_noise"]),
        (cfg["paths"]["raw_speech"],    cfg["paths"]["prepared_speech"]),
    ]
    exts = (".mp3", ".flac", ".wav", ".ogg", ".m4a", ".opus")

    # Создаём папки
    for _, dst in mapping:
        ensure_dir(dst)

    # Обходим все raw → конвертим в prepared
    for src, dst in mapping:
        for root, _, files in os.walk(src):
            for fn in tqdm(files):
                if not fn.lower().endswith(exts):
                    continue
                inp = os.path.join(root, fn)
                base = os.path.splitext(fn)[0]
                out = os.path.join(dst, base + ".wav")
                if os.path.exists(out):
                    continue
                #print(f"Processing {inp} → {out}")
                process_one(inp, out)

    print("Done! Ошибки, если были, в", ERROR_LOG)


if __name__ == "__main__":
    main()
