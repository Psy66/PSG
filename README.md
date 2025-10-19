# 🏥 Sleep Analysis Pipeline

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![MNE](https://img.shields.io/badge/MNE-Python-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![GitHub](https://img.shields.io/badge/Platform-GitHub-lightgrey)

**Automated Polysomnography (PSG) Data Analysis System**

A comprehensive pipeline for processing sleep study data from EDF files, extracting key sleep metrics, and generating database-ready SQL queries.

## ✨ Features

- 🎯 **Automatic EDF File Processing** - Read and parse polysomnography data
- 🔍 **Artifact Detection** - Smart identification of signal artifacts
- 📊 **Sleep Stage Analysis** - Comprehensive sleep architecture scoring
- ❤️ **ECG Analysis** - Heart rate variability and arrhythmia detection
- 🌬️ **Respiratory Analysis** - Breathing patterns and event detection  
- 💓 **SpO2 Monitoring** - Oxygen saturation statistics
- 🗄️ **SQL Export** - Automated database updates
- ⚡ **Parallel Processing** - Fast multi-file analysis

## 🚀 Quick Start

### Installation

```bash
pip install mne numpy scipy
```

### Basic Usage

```python
from edf34 import SQLGenerator

# Process all EDF files in a folder
generator = SQLGenerator()
generator.generate_sql_files('edf_files/', 'sql_output/')
```

### Individual Component Usage

#### Analyze Sleep Stages
```python
from edf34 import SleepAnalyzer

analyzer = SleepAnalyzer()
analyzer.load_edf('sleep_study.edf')
stages = analyzer.calculate_stages()
print(f"Sleep Efficiency: {stages['sleep_efficiency']}%")
```

#### ECG Analysis
```python
from edf34 import SignalAnalyzer

analyzer = SignalAnalyzer()
ecg_results = analyzer.analyze_ecg(raw_data)
print(f"Average HR: {ecg_results['avg_heart_rate']} bpm")
```

## 📈 Output Metrics

### Sleep Architecture
- Total Sleep Time & Efficiency
- N1, N2, N3, REM percentages
- Sleep Latency & WASO
- REM Cycles & Density

### Respiratory Analysis
- Apnea-Hypopnea Index (AHI)
- Obstructive/Central/Mixed events
- Oxygen Desaturation Index (ODI)
- Snoring Analysis

### Cardiac Metrics
- Heart Rate & Variability
- Tachycardia/Bradycardia events
- Comprehensive HRV analysis

### Quality Scores
- Overall Sleep Quality (0-100)
- REM Sleep Quality
- Fragmentation Indices

## 🗃️ Database Integration

The system generates optimized SQL queries for MariaDB/MySQL:

```sql
UPDATE sleep_statistics SET 
    total_sleep_time = 429,
    sleep_efficiency = 85.39,
    ahi = 13.83,
    overall_sleep_quality = 56
WHERE edf_uuid = '1b16b146-91d4-49be-a313-98fea1907037';
```

## 🛠️ Configuration

Easy customization through config dictionary:

```python
CONFIG = {
    'ecg': {'hr_min': 40, 'hr_max': 150},
    'respiration': {'min_rate': 8, 'max_rate': 25},
    'spo2': {'threshold_90': 90, 'threshold_85': 85}
}
```

## 📁 Project Structure

```
edf34.py              # Main analysis pipeline
PSG.sql               # Database schema
sleep_stats_*.sql     # Generated SQL files
```

## 🎯 Use Cases

- **Sleep Clinics** - Automated PSG report generation
- **Research** - Bulk analysis of sleep studies
- **Telemedicine** - Remote sleep monitoring
- **Clinical Trials** - Standardized sleep metrics

## 🤝 Contributing

We welcome contributions! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review the example SQL outputs

---

**Transform your sleep study analysis with automated, reliable, and comprehensive processing!** 🎉

---

# Техническая документация для разработчиков

# Sleep Analysis Pipeline - Техническая документация

## Обзор архитектуры

Проект представляет собой модульную систему анализа полисомнографических данных с четким разделением ответственности между компонентами.

## Детальное описание компонентов

### 1. ArtifactProcessor - Обработка артефактов

#### Метод `get_artifact_mask(raw, artifact_marker='Артефакт(blockArtefact)')`
**Назначение:** Создание бинарной маски валидных данных
**Входные параметры:**
- `raw`: MNE Raw объект
- `artifact_marker`: строка-идентификатор артефакта в аннотациях

**Возвращает:**
- `valid_mask`: numpy array булевых значений (True - валидные данные)
- `artifact_regions`: список словарей с метаданными артефактов

**Пример использования:**
```python
processor = ArtifactProcessor()
valid_mask, artifacts = processor.get_artifact_mask(raw_data)
clean_data = raw_data[valid_mask]  # Применение маски
```

#### Метод `get_heartbeat_gaps(raw, marker='pointIlluminationSensorValue', max_gap=5.0, min_duration=10.0)`
**Назначение:** Обнаружение продолжительных пропусков в сигнале пульса
**Параметры:**
- `max_gap`: максимальный допустимый интервал между心跳 (сек)
- `min_duration`: минимальная длительность gap для регистрации

### 2. SignalAnalyzer - Анализ сигналов

#### ЭКГ анализ: `analyze_ecg(raw)`

**Алгоритм детекции R-пиков:**
1. **Предобработка:**
   ```python
   ecg_clean = ecg_signal - np.median(ecg_signal)  # Удаление DC смещения
   b, a = butter(3, [5/(sfreq/2), 35/(sfreq/2)], btype='band')  # БПФ 5-35 Гц
   ecg_filtered = filtfilt(b, a, ecg_clean)
   ```

2. **Детекция пиков:**
   ```python
   ecg_squared = np.square(ecg_filtered)  # Усиление R-пиков
   ecg_smoothed = medfilt(ecg_squared, kernel_size=window_size)  # Медианный фильтр
   threshold = np.percentile(ecg_smoothed, 85)  # Адаптивный порог
   peaks, _ = find_peaks(ecg_smoothed, height=threshold, distance=int(0.3*sfreq))
   ```

3. **Валидация RR-интервалов:**
   ```python
   rr_intervals = np.diff(r_peaks) / sfreq
   valid_rr = rr_intervals[(rr_intervals > rr_min) & (rr_intervals < rr_max)]
   hr = 60.0 / valid_rr  # Преобразование в ЧСС
   ```

**Возвращаемые метрики:**
- `avg_heart_rate`: медиана валидной ЧСС
- `heart_rate_variability`: SDNN (стандартное отклонение NN интервалов)
- `tachycardia_events/bradycardia_events`: счетчики событий из аннотаций

#### Анализ дыхания: `analyze_respiration(raw)`

**Поиск дыхательных каналов:**
```python
resp_keywords = ['resp', 'breath', 'дыхание', 'thorax', 'chest', 'abdomen']
resp_channels = [ch for ch in raw.ch_names if any(p in ch.lower() for p in resp_keywords)]
```

**Алгоритм детекции дыхательных циклов:**
```python
def analyze_breathing(resp_signal, sfreq):
    normalized = (resp_signal - np.mean(resp_signal)) / (np.std(resp_signal) + 1e-8)
    peaks, properties = find_peaks(
        normalized,
        distance=int(0.6 * sfreq),  # Минимальный интервал 0.6 сек
        prominence=0.05,            # Минимальная prominance
        height=0.02,               # Минимальная высота
        width=int(0.2 * sfreq),    # Минимальная ширина
        wlen=int(2 * sfreq)        # Размер окна для prominance
    )
```

#### Анализ SpO2: `analyze_spo2(raw)`

**Метрики:**
- `avg_spo2`: медиана сатурации
- `min_spo2`: 1-й процентиль
- `spo2_baseline`: 90-й процентиль (базовая линия)
- `time_below_spo2_90/85`: время ниже порогов в минутах

### 3. SleepAnalyzer - Анализ сна

#### Парсинг гипнограммы: `calculate_stages()`

**Маппинг аннотаций:**
```python
mapping = {
    'Sleep stage W(eventUnknown)': 'Wake',
    'Sleep stage 1(eventUnknown)': 'N1', 
    'Sleep stage 2(eventUnknown)': 'N2',
    'Sleep stage 3(eventUnknown)': 'N3',
    'Sleep stage R(eventUnknown)': 'REM'
}
```

**Расчет архитектуры сна:**
```python
def calculate_architecture(self):
    total_sleep = sum(self.stages[s]['minutes'] for s in ['N1', 'N2', 'N3', 'REM'])
    return {
        'n1_percentage': (self.stages['N1']['minutes'] / total_sleep) * 100,
        'n2_percentage': (self.stages['N2']['minutes'] / total_sleep) * 100,
        # ... аналогично для других стадий
    }
```

#### Расчет респираторных событий: `calculate_respiratory_events()`

**Маппинг событий:**
```python
mapping = {
    'Обструктивное апноэ(pointPolySomnographyObstructiveApnea)': 'obstructive_apneas',
    'Центральное апноэ(pointPolySomnographyCentralApnea)': 'central_apneas',
    'Обструктивное гипопноэ(pointPolySomnographyHypopnea)': 'obstructive_hypopneas',
    # ... и т.д.
}
```

#### Расчет качества сна: `calculate_sleep_quality()`

**Система оценки (0-100 баллов):**

1. **Эффективность сна (до 25 баллов):**
   - ≥85%: 25 баллов
   - ≥70%: 20 баллов  
   - ≥50%: 10 баллов

2. **Глубокий сон (15 баллов):**
   - N3 ≥ 15%: 15 баллов

3. **REM сон (15 баллов):**
   - REM ≥ 20%: 15 баллов

4. **Индекс апноэ (до 30 баллов):**
   - AHI < 5: 30 баллов
   - AHI < 15: 20 баллов
   - AHI < 30: 10 баллов

5. **Индекс активаций (до 15 баллов):**
   - AI < 10: 15 баллов
   - AI < 20: 10 баллов

6. **REM качество (до 15 баллов):**
   - Оценка REM качества (0-100) × 0.15

7. **Коррекция по аритмиям:**
   - Тахи/брадикардия > 10 событий: -15 баллов
   - Тахи/брадикардия > 5 событий: -10 баллов
   - Любые события: -5 баллов

8. **REM циклы:**
   - ≥4 цикла: +10 баллов
   - ≥3 цикла: +5 баллов

### 4. SQLGenerator - Генерация БД запросов

#### Параллельная обработка:
```python
def generate_sql_files(self, input_dir, output_dir, max_workers=10):
    edf_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.edf')]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(self.process_file, edf_files))
```

## Примеры использования отдельных компонентов

### Только анализ ЭКГ
```python
from edf34 import SignalAnalyzer, CONFIG
import mne

# Загрузка данных
raw = mne.io.read_raw_edf('recording.edf', preload=True)

# Создание анализатора
analyzer = SignalAnalyzer(CONFIG)

# Анализ ЭКГ
ecg_results = analyzer.analyze_ecg(raw)
print(f"Средняя ЧСС: {ecg_results['avg_heart_rate']}")
print(f"ВСР: {ecg_results['heart_rate_variability']} мс")
```

### Анализ дыхания с кастомными параметрами
```python
custom_config = {
    'respiration': {'min_rate': 6, 'max_rate': 40, 'filter_low': 0.05, 'filter_high': 0.8}
}
analyzer = SignalAnalyzer(custom_config)
resp_results = analyzer.analyze_respiration(raw)
```

### Расчет стадий сна
```python
from edf34 import SleepAnalyzer

analyzer = SleepAnalyzer()
analyzer.load_edf('sleep_study.edf')
stages = analyzer.calculate_stages()
efficiency = analyzer.calculate_efficiency()

print(f"Эффективность сна: {efficiency['sleep_efficiency']:.1f}%")
print(f"REM сон: {stages['REM']['minutes']} минут")
```

### Экспорт гипнограммы
```python
hypnogram = analyzer.export_hypnogram()
# Формат: {'e': количество эпох, 'd': длительность эпохи, 's': последовательность стадий}
```

## Расширение функциональности

### Добавление нового типа анализа
```python
class CustomAnalyzer(SignalAnalyzer):
    def analyze_custom_signal(self, raw, channel_name):
        # Реализация custom логики
        pass
```

### Кастомная конфигурация
```python
CUSTOM_CONFIG = {
    **CONFIG,
    'custom_analysis': {
        'param1': value1,
        'param2': value2
    }
}
```

---
