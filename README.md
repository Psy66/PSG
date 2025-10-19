# 🏥 Sleep Analysis Pipeline

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![MNE](https://img.shields.io/badge/MNE-Python-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![GitHub](https://img.shields.io/badge/Platform-GitHub-lightgrey)

**Enterprise-Grade Polysomnography Data Analysis System**

A comprehensive, production-ready pipeline for automated processing and analysis of sleep study data from EDF files, featuring advanced signal processing and database integration.

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Advanced Usage](#-advanced-usage)
- [Output Metrics](#-output-metrics)
- [Database Integration](#-database-integration)
- [Configuration](#-configuration)
- [Technical Documentation](#-technical-documentation)
- [Contributing](#-contributing)
- [License](#-license)
- [Medical Disclaimer](#-medical-disclaimer)

## ✨ Features

### Core Analysis Capabilities
- **Automated EDF Processing** - Robust parsing of polysomnography data with artifact handling
- **Advanced Signal Processing** - Multi-channel ECG, respiration, and SpO2 analysis
- **Sleep Architecture Analysis** - Comprehensive sleep stage scoring and architecture assessment
- **Respiratory Event Detection** - Apnea, hypopnea, and desaturation analysis
- **Cardiac Monitoring** - Heart rate variability and arrhythmia detection
- **Quality Metrics** - Automated sleep quality scoring and data quality assessment

### Enterprise Features
- **Parallel Processing** - High-throughput multi-file analysis with configurable workers
- **Database Integration** - Automated SQL generation for MariaDB/MySQL
- **Artifact Management** - Intelligent artifact detection and data validation
- **Modular Architecture** - Extensible component-based design
- **Production Ready** - Comprehensive error handling and logging

## 🏗 System Architecture

The system employs a modular architecture with four core components:

### 1. ArtifactProcessor
- **Artifact Mask Generation** - Binary masking of valid data regions
- **Heartbeat Gap Detection** - Identification of signal discontinuities
- **Data Quality Assessment** - Automated quality scoring

### 2. SignalAnalyzer
- **ECG Analysis** - R-peak detection, HRV calculation, arrhythmia monitoring
- **Respiratory Analysis** - Breathing pattern analysis and event detection
- **SpO2 Monitoring** - Oxygen saturation statistics and desaturation events

### 3. SleepAnalyzer
- **Sleep Stage Processing** - Hypnogram parsing and architecture calculation
- **Respiratory Event Scoring** - AHI, ODI, and respiratory indices
- **Sleep Quality Assessment** - Multi-factor quality scoring (0-100)

### 4. SQLGenerator
- **Parallel Processing** - Concurrent EDF file processing
- **Database Integration** - Automated SQL query generation
- **Batch Operations** - Bulk analysis and export capabilities

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- MariaDB/MySQL database (for SQL export)

### Installation Methods

#### Method 1: PIP Installation
```bash
pip install mne>=1.0.0 numpy>=1.21.0 scipy>=1.7.0
```

#### Method 2: From Source
```bash
git clone https://github.com/Psy66/sleep-analysis-pipeline.git
cd sleep-analysis-pipeline
pip install -r requirements.txt
```

## 📖 Quick Start

### Basic Pipeline Usage

```python
from psg_edf import SQLGenerator

# Process all EDF files in directory
generator = SQLGenerator()
generator.generate_sql_files('edf_files/', 'sql_output/')
```

### Individual Component Usage

#### Sleep Stage Analysis
```python
from psg_edf import SleepAnalyzer

analyzer = SleepAnalyzer()
analyzer.load_edf('sleep_study.edf')
stages = analyzer.calculate_stages()
efficiency = analyzer.calculate_efficiency()

print(f"Sleep Efficiency: {efficiency['sleep_efficiency']:.1f}%")
print(f"REM Sleep: {stages['REM']['minutes']} minutes")
```

#### Signal Analysis
```python
from psg_edf import SignalAnalyzer

analyzer = SignalAnalyzer()
ecg_results = analyzer.analyze_ecg(raw_data)
resp_results = analyzer.analyze_respiration(raw_data)

print(f"Average HR: {ecg_results['avg_heart_rate']} bpm")
print(f"Respiratory Rate: {resp_results['avg_resp_rate']} breaths/min")
```

#### Annotation Analysis
```python
from anno_uniq import extract_annotations_simple

# Extract all unique annotations from EDF files
annotations = extract_annotations_simple("EDF")
print(f"Found {len(annotations)} unique annotation types")
```

## 🔧 Advanced Usage

### Custom Configuration

```python
CONFIG = {
    # Конфигурация анализа ЭКГ (электрокардиографии)
    'ecg': {
        'rr_min': 0.3,        # Минимальный допустимый RR-интервал в секундах
                              # (соответствует ЧСС ~200 уд/мин)
        
        'rr_max': 2.0,        # Максимальный допустимый RR-интервал в секундах  
                              # (соответствует ЧСС ~30 уд/мин)
        
        'hr_min': 40,         # Минимальная допустимая частота сердечных сокращений (ЧСС)
                              # в ударах в минуту (bpm)
        
        'hr_max': 150         # Максимальная допустимая частота сердечных сокращений (ЧСС)
                              # в ударах в минуту (bpm)
    },
    
    # Конфигурация анализа дыхания
    'respiration': {
        'min_rate': 8,        # Минимальная физиологически допустимая частота дыхания
                              # в дыхательных движениях в минуту
        
        'max_rate': 25,       # Максимальная физиологически допустимая частота дыхания
                              # в дыхательных движениях в минуту
        
        'filter_low': 0.1,    # Нижняя граница полосового фильтра для дыхательного сигнала
                              # в герцах (соответствует ~6 дыханий в минуту)
        
        'filter_high': 1.0    # Верхняя граница полосового фильтра для дыхательного сигнала
                              # в герцах (соответствует ~60 дыханий в минуту)
    },
    
    # Конфигурация анализа насыщения крови кислородом (SpO2)
    'spo2': {
        'min_valid': 75,      # Минимальное допустимое значение SpO2 в процентах
                              # (значения ниже считаются артефактами)
        
        'max_valid': 100,     # Максимальное допустимое значение SpO2 в процентах
                              # (значения выше считаются артефактами)
        
        'threshold_90': 90,   # Пороговое значение SpO2 для расчета времени десатурации
                              # (время, проведенное ниже 90%)
        
        'threshold_85': 85    # Пороговое значение SpO2 для расчета времени тяжелой десатурации
                              # (время, проведенное ниже 85%)
    },
    
    # Конфигурация расчета качества сна
    'sleep_quality': {
        # Весовые коэффициенты для оценки эффективности сна
        'efficiency_weights': {
            85: 25,   # 25 баллов за эффективность сна ≥85%
            70: 20,   # 20 баллов за эффективность сна ≥70%
            50: 10    # 10 баллов за эффективность сна ≥50%
        },
        
        'n3_threshold': 15,   # Минимальный процент глубокого сна (N3) для получения баллов
                              # (15% от общего времени сна)
        
        'rem_threshold': 20,  # Минимальный процент REM-сна для получения баллов
                              # (20% от общего времени сна)
        
        # Весовые коэффициенты для индекса апноэ-гипопноэ (AHI)
        'ahi_weights': {
            5: 30,    # 30 баллов за AHI < 5 (норма)
            15: 20,   # 20 баллов за AHI < 15 (легкая степень)
            30: 10    # 10 баллов за AHI < 30 (средняя степень)
        },
        
        # Весовые коэффициенты для индекса активаций (arousal index)
        'arousal_weights': {
            10: 15,   # 15 баллов за индекс активаций < 10
            20: 10    # 10 баллов за индекс активаций < 20
        }
    }
}
```

### Parallel Processing
```python
# Process files with 8 parallel workers
generator.generate_sql_files('input_edf/', 'output_sql/', max_workers=8)
```

### SQL File Combination
```python
# Combine individual SQL files into single execution file
generator.combine_sql_files('sql_output/', 'combined_updates.sql')
```

## 📊 Output Metrics

### Sleep Architecture Metrics
- **Total Sleep Time** (minutes) and **Sleep Efficiency** (%)
- **Sleep Stage Distribution**: N1, N2, N3, REM percentages
- **Sleep Latency** and **Wake After Sleep Onset** (WASO)
- **REM Cycles** count and **REM Latency**

### Respiratory Analysis
- **Apnea-Hypopnea Index** (AHI) - Total, Obstructive, Central, Mixed
- **Oxygen Desaturation Index** (ODI)
- **Event Counts**: Apneas, Hypopneas, Desaturations, Snoring
- **SpO2 Statistics**: Average, Minimum, Baseline, Time below thresholds

### Cardiac Metrics
- **Heart Rate**: Average, Minimum, Maximum
- **Heart Rate Variability** (SDNN)
- **Arrhythmia Events**: Tachycardia and Bradycardia counts

### Quality and Movement Metrics
- **Sleep Quality Score** (0-100) with status classification
- **Arousal Index** and **Sleep Fragmentation Index**
- **Limb Movements**: Total, Periodic (PLMI)
- **Bruxism Events** count

### Data Quality
- **Artifact Analysis**: Count and total duration
- **Signal Quality** assessment
- **Processing Metadata**

## 🗃️ Database Integration

### Generated SQL Structure
The system produces optimized SQL updates for MariaDB/MySQL:

```sql
-- SQL запрос для обновления статистики сна
-- UUID исследования: 1b16b146-91d4-49be-a313-98fea1907037
-- Файл: sleep_study.edf
-- Сгенерировано: 2024-01-15 14:30:25

UPDATE `sleep_statistics` ss
JOIN `psg_studies` ps ON ss.study_id = ps.study_id
SET `total_sleep_time` = 429,
    `sleep_efficiency` = 85.39,
    `ahi` = 13.83,
    `overall_sleep_quality` = 56,
    -- ... additional metrics
WHERE ps.edf_uuid = '1b16b146-91d4-49be-a313-98fea1907037';

UPDATE `psg_studies` 
SET `artifact_count` = 12, 
    `artifact_duration_minutes` = 45.5
WHERE `edf_uuid` = '1b16b146-91d4-49be-a313-98fea1907037';
```

### Database Schema
Refer to `PSG.sql` for complete database schema including:
- `psg_studies` - Study metadata and artifact information
- `sleep_statistics` - Comprehensive sleep metrics
- Additional tables for raw data and event logging


## 📚 Technical Documentation

### Algorithm Details

#### ECG Analysis Pipeline
1. **Signal Preprocessing** - DC removal and bandpass filtering (5-35 Hz)
2. **R-Peak Detection** - Adaptive thresholding and peak validation
3. **HRV Calculation** - RR interval analysis and outlier removal
4. **Arrhythmia Detection** - Annotation-based event counting

#### Respiratory Analysis
1. **Channel Identification** - Automatic detection of respiratory signals
2. **Breathing Cycle Detection** - Peak detection with prominence validation
3. **Rate Calculation** - Statistical analysis with outlier removal
4. **Event Integration** - Combination with annotation-based events

#### Sleep Stage Processing
1. **Hypnogram Parsing** - Annotation to stage mapping
2. **Architecture Calculation** - Percentage and duration calculations
3. **Quality Scoring** - Multi-factor assessment algorithm
4. **REM Analysis** - Cycle detection and quality assessment

### Performance Characteristics
- **Processing Speed**: ~2-5 minutes per EDF file (depending on duration and channels)
- **Memory Usage**: Optimized for typical PSG file sizes (100-500 MB)
- **Parallel Scaling**: Linear scaling with core count up to 10-12 workers

## 🤝 Contributing

We welcome contributions from the research community. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Submit a pull request

### Code Standards
- Follow PEP8 guidelines
- Include type hints for new functions
- Add tests for new functionality
- Update documentation accordingly

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer

**Important**: This software is intended for research purposes only. It is not a medical device and should not be used for diagnostic or therapeutic purposes. Always consult qualified healthcare professionals for medical advice and diagnosis.

See [DISCLAIMER.md](DISCLAIMER.md) for complete medical disclaimer.

## 👨‍💻 Author

**Tim Liner**  
- GitHub: [@Psy66](https://github.com/Psy66)  
- Project Repository: [sleep-analysis-pipeline](https://github.com/Psy66/PSG)

---

*Last updated: October 2025*
