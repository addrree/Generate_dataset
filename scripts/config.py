import os
import yaml


def load_config():
    """
    Читает config.yaml из корня проекта и возвращает словарь:
      - cfg["paths"] — пути к папкам (raw и prepared)
    """
    base = os.getcwd()
    cfg_path = os.path.join(base, "generation_config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # делаем все пути из config.yaml абсолютными
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = os.path.join(base, v)

    return cfg

