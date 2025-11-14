# Генератор синтетического аудио-датасета  
Речь • Музыка • Шумы • Сценарии

Проект генерирует аудиотреки (`.wav`) и разметку (`.yaml`), комбинируя речь, музыку, шумы по YAML-сценариям.

---

# 1. Структура проекта
```
generate_dataset/
├─ dataset/ # куда падают итоговые .wav + .yaml
├─ scenarios/ # сценарии генерации (*.yaml)
├─ scripts/
│ ├─ config.py
│ ├─ generate_synthetic.py
│ ├─ preprocess.py
│ └─ utils.py
└─ config.yaml # конфиг путей
```
# 2. Что нужно для запуска

## Обязательные компоненты
- Python **3.9+**
- Установленный **FFmpeg** и доступный в `PATH`  
  Проверить:
```bash
  ffmpeg -version
```
  - Установленные Python-библиотеки: 
```bash
   pip install -r requirements.txt
```
  
## Необходимые папки и файлы
- `config.yaml` — конфигурация путей
- Папки, указанные в конфиге:
- `raw_music`, `raw_noise`,`raw_speech` — сырые данные
- `prepared_music`, `prepared_noise`,`prepared_speech` — результат препроцессинга
- `scenarios` — YAML-сценарии
- `output` — финальный датасет

---

# 3. Конфигурация: `config.yaml`
```
paths:
raw_music:      data/raw/music
prepared_music: data/prepared/music

raw_noise:      data/raw/noise
prepared_noise: data/prepared/noise

raw_speech:      data/raw/speech
prepared_speech: data/prepared/speech

scenarios: scenarios
output:   dataset
```

# 4. Как работает генерация

# 4.1 Препроцессинг аудио
Выполняется скриптом 
```bash
scripts/preprocess.py:
```
читает файлы из raw_music и raw_noise и raw_speech
приводит к формату:mono 16 kHz PCM WAV
сохраняет в prepared_music и prepared_noise и prepared_speech

# 4.2 Сценарии (scenarios/*.yaml)
Каждый сценарий — список шагов:

scenario: call_from_cafe
sequence:
  - type: speech
    dur_mult: 2
  - type: background_music
    duck_db: [8, 15]
    dur_mult: 3
  - type: speech
    gain_db: -2
    dur_mult: 2


type: speech, music, silence, background_music, fade_to_music, fade_music

dur_mult: вес шага при распределении общей длительности

gain_db: усиление/ослабление

duck_db: приглушение музыки, когда идёт речь

# 4.3. Генерация

выбирается случайный файл нужного типа (речь/музыка/шум)

шаги сценария формируют трек

создаются два файла:

example_00000.wav    (аудио)
example_00000.yaml   (разметка)

# 5. Препроцессинг (запуск)
```bash
cd scripts
python preprocess.py
```
# 6. Генерация датасета

Пример:
```bash
python -m scripts.generate_synthetic --n 200 --noise-prob 0.4 --seconds 30
```
Где:

--n — количество треков

--noise-prob — вероятность добавления шумов

--seconds — длительность каждого аудиофайла

Результат появится в paths.output:

dataset/
  example_00000.wav
  example_00000.yaml
  example_00001.wav
  example_00001.yaml

# 7. Формат разметки (*.yaml рядом с аудио)
audio_path: example_00000.wav
sample_rate: 16000
segments:
  - label: speech
    start: 0.00
    end: 3.25
  - label: music
    start: 3.25
    end: 10.40
  - label: silence
    start: 10.40
    end: 12.00

# 8. Итог

 Python 3.9+

 FFmpeg установлен

 pip install -r requirements.txt

 Папки существуют

 В scenarios/ есть сценарии

 Препроцессинг выполнен

 Генерация запускается без ошибок
