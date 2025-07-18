import os
import yaml

def load_config():
    """
    Читает config.yaml из корня проекта и возвращает словарь:
      - cfg["paths"] — пути к папкам (raw и prepared)
      - cfg["ffmpeg"] — путь до ffmpeg
    """
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = os.path.join(base, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # делаем все пути из config.yaml абсолютными
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = os.path.join(base, v)
    # ffmpeg уже должен быть абсолютным путём в config.yaml
    return cfg

