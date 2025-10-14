import mne
from scipy import signal
import numpy as np
import json
import os
import re
from collections import Counter
from datetime import datetime

def calculate_sleep_stages(raw):
    """Расчет стадий сна по 30-секундным эпохам"""
    if not raw or not hasattr(raw, 'annotations'):
        return None

    annotations = raw.annotations

    stage_mapping = {
        'Sleep stage W(eventUnknown)': 'Wake',
        'Sleep stage 1(eventUnknown)': 'N1',
        'Sleep stage 2(eventUnknown)': 'N2',
        'Sleep stage 3(eventUnknown)': 'N3',
        'Sleep stage R(eventUnknown)': 'REM'
    }

    stages = {stage: {'count': 0, 'minutes': 0} for stage in ['Wake', 'N1', 'N2', 'N3', 'REM']}

    for desc, duration in zip(annotations.description, annotations.duration):
        desc_str = str(desc)
        if desc_str in stage_mapping and abs(duration - 30) < 1:
            stage = stage_mapping[desc_str]
            stages[stage]['count'] += 1
            stages[stage]['minutes'] += 0.5

    return stages

def calculate_rem_quality(raw):
    """Расчет качества REM-сна"""
    if not raw or not hasattr(raw, 'annotations'):
        return None

    annotations = raw.annotations

    rem_epochs = sum(1 for desc, duration in zip(annotations.description, annotations.duration)
                     if str(desc) == 'Sleep stage R(eventUnknown)' and abs(duration - 30) < 1)

    rem_events = sum(1 for desc in annotations.description
                     if str(desc) == 'БДГ(pointPolySomnographyREM)')

    rem_minutes = rem_epochs * 0.5
    rem_density = rem_events / rem_minutes if rem_minutes > 0 else 0

    time_score = 40 if rem_minutes >= 15 else 20 if rem_minutes >= 5 else 0
    density_score = 60 if rem_density >= 1.5 else 30 if rem_density >= 0.5 else 0
    quality_score = min(time_score + density_score, 100)

    status = ("отлично" if quality_score >= 80 else
              "хорошо" if quality_score >= 60 else
              "удовлетворительно" if quality_score >= 40 else "низкое")

    return {
        'rem_quality_score': int(quality_score),
        'rem_minutes': rem_minutes,
        'rem_events': rem_events,
        'rem_density': rem_density,
        'status': status
    }

def calculate_sleep_efficiency(raw, stages):
    """Расчет эффективности сна"""
    if not stages:
        return None

    total_sleep_time = sum(stages[stage]['minutes'] for stage in ['N1', 'N2', 'N3', 'REM'])
    total_bed_time = sum(stage['minutes'] for stage in stages.values())

    sleep_efficiency = (total_sleep_time / total_bed_time * 100) if total_bed_time > 0 else 0

    return {
        'sleep_efficiency': sleep_efficiency,
        'total_sleep_time': total_sleep_time,
        'total_bed_time': total_bed_time,
        'wake_after_sleep_onset': stages['Wake']['minutes']
    }

def calculate_sleep_latencies(raw, stages):
    """Расчет различных латентностей"""
    annotations = raw.annotations

    sleep_onset_latency = None
    rem_latency = None

    first_sleep_epoch = None
    first_rem_epoch = None

    current_time = 0
    for desc, duration in zip(annotations.description, annotations.duration):
        desc_str = str(desc)

        if desc_str in ['Sleep stage 1(eventUnknown)', 'Sleep stage 2(eventUnknown)',
                        'Sleep stage 3(eventUnknown)', 'Sleep stage R(eventUnknown)']:
            if first_sleep_epoch is None and abs(duration - 30) < 1:
                first_sleep_epoch = current_time

        if desc_str == 'Sleep stage R(eventUnknown)' and first_rem_epoch is None and abs(duration - 30) < 1:
            first_rem_epoch = current_time

        current_time += duration

    return {
        'sleep_onset_latency': first_sleep_epoch / 60 if first_sleep_epoch else None,
        'rem_latency': (first_rem_epoch - first_sleep_epoch) / 60 if first_sleep_epoch and first_rem_epoch else None
    }

def calculate_sleep_architecture(stages):
    """Анализ архитектуры сна"""
    if not stages:
        return None

    total_sleep_time = sum(stages[stage]['minutes'] for stage in ['N1', 'N2', 'N3', 'REM'])

    if total_sleep_time == 0:
        return None

    architecture = {
        'n1_percentage': (stages['N1']['minutes'] / total_sleep_time) * 100,
        'n2_percentage': (stages['N2']['minutes'] / total_sleep_time) * 100,
        'n3_percentage': (stages['N3']['minutes'] / total_sleep_time) * 100,
        'rem_percentage': (stages['REM']['minutes'] / total_sleep_time) * 100,
        'deep_sleep_ratio': stages['N3']['minutes'] / total_sleep_time,
        'rem_nrem_ratio': stages['REM']['minutes'] / (
                stages['N1']['minutes'] + stages['N2']['minutes'] + stages['N3']['minutes'])
    }

    return architecture

def calculate_sleep_fragmentation(raw, stages):
    """Анализ фрагментации сна с разделением типов движений"""
    annotations = raw.annotations

    activations = sum(1 for desc in annotations.description
                      if str(desc) == 'Активация(pointPolySomnographyActivation)')

    # РАЗДЕЛЯЕМ движения на обычные и периодические
    limb_movements = sum(1 for desc in annotations.description
                         if str(desc) == 'Движение конечностей(pointPolySomnographyLegsMovements)')

    periodic_limb_movements = sum(1 for desc in annotations.description
                                  if
                                  str(desc) == 'Периодические движения конечностей(pointPolySomnographyPeriodicalLegsMovements)')

    total_sleep_time = sum(stages[stage]['minutes'] for stage in ['N1', 'N2', 'N3', 'REM'])

    total_movements = limb_movements + periodic_limb_movements
    fragmentation_index = (activations + total_movements) / (total_sleep_time / 60) if total_sleep_time > 0 else 0

    return {
        'fragmentation_index': fragmentation_index,
        'activations': activations,
        'limb_movements': limb_movements,  # Обычные движения
        'periodic_limb_movements': periodic_limb_movements,  # Периодические движения
        'total_limb_movements': total_movements,  # Все движения
        'arousal_index': activations / (total_sleep_time / 60) if total_sleep_time > 0 else 0
    }

def calculate_sleep_indices(raw, stages):
    """Расчет различных индексов с защитой от None"""
    if not raw or not stages:
        return {}

    respiratory_events = calculate_respiratory_events(raw) or {}
    total_sleep_time = sum(stages[stage]['minutes'] for stage in ['N1', 'N2', 'N3', 'REM'])

    if total_sleep_time == 0:
        return {}

    ahi = (respiratory_events.get('apneas', 0) + respiratory_events.get('hypopneas', 0)) / (total_sleep_time / 60)
    odi = respiratory_events.get('desaturations', 0) / (total_sleep_time / 60)
    snoring_index = respiratory_events.get('snoring', 0) / (total_sleep_time / 60)

    ahi_severity = ("норма" if ahi < 5 else
                    "легкая" if ahi < 15 else
                    "средняя" if ahi < 30 else "тяжелая")

    return {
        'ahi': ahi,
        'ahi_severity': ahi_severity,
        'odi': odi,
        'snoring_index': snoring_index,
        'total_apneas': respiratory_events.get('apneas', 0),
        'total_hypopneas': respiratory_events.get('hypopneas', 0),
        'total_desaturations': respiratory_events.get('desaturations', 0),
        'total_snores': respiratory_events.get('snoring', 0)
    }

def calculate_overall_sleep_quality(raw, stages):
    """Комплексная оценка качества сна с защитой от None"""
    if not raw or not stages:
        return {}

    efficiency_data = calculate_sleep_efficiency(raw, stages) or {}
    architecture_data = calculate_sleep_architecture(stages) or {}
    sleep_indices = calculate_sleep_indices(raw, stages) or {}
    fragmentation_data = calculate_sleep_fragmentation(raw, stages) or {}
    rem_quality = calculate_rem_quality(raw) or {}

    # Анализ сердечных эпизодов и REM-циклов
    hr_stats = analyze_heart_rate_comprehensive(raw) or {}
    rem_cycles = calculate_rem_cycles(raw)

    # Упрощенная балльная система
    score = 0

    # Эффективность сна
    efficiency = efficiency_data.get('sleep_efficiency', 0)
    if efficiency >= 85:
        score += 25
    elif efficiency >= 70:
        score += 20
    elif efficiency >= 50:
        score += 10

    # Архитектура сна
    n3_percentage = architecture_data.get('n3_percentage', 0)
    rem_percentage = architecture_data.get('rem_percentage', 0)

    if n3_percentage >= 15:
        score += 15
    if rem_percentage >= 20:
        score += 15

    # Дыхательные нарушения
    ahi = sleep_indices.get('ahi', 0)
    if ahi < 5:
        score += 30
    elif ahi < 15:
        score += 20
    elif ahi < 30:
        score += 10

    # Фрагментация
    arousal_index = fragmentation_data.get('arousal_index', 0)
    if arousal_index < 10:
        score += 15
    elif arousal_index < 20:
        score += 10

    # REM качество
    rem_score = rem_quality.get('rem_quality_score', 0)
    score += rem_score * 0.15

    # Штраф за тахикардию (вставляется после существующих расчетов)
    tachycardia_events = hr_stats.get('tachycardia_events')
    if tachycardia_events > 10:
        score -= 15
    elif tachycardia_events > 5:
        score -= 10
    elif tachycardia_events > 0:
        score -= 5

    # Бонус за хорошую архитектуру REM-циклов
    if rem_cycles >= 4:
        score += 10
    elif rem_cycles >= 3:
        score += 5

    overall_score = min(score, 100)

    if overall_score >= 85:
        status = "отличное"
    elif overall_score >= 70:
        status = "хорошее"
    elif overall_score >= 50:
        status = "удовлетворительное"
    else:
        status = "плохое"

    return {
        'overall_score': int(overall_score),
        'status': status
    }

def generate_hypnogram(raw):
    """Генерация гипнограммы - распределения стадий сна по времени"""
    if not raw or not hasattr(raw, 'annotations'):
        return None

    annotations = raw.annotations

    # Маппинг аннотаций на стадии сна
    stage_mapping = {
        'Sleep stage W(eventUnknown)': 'Wake',
        'Sleep stage 1(eventUnknown)': 'N1',
        'Sleep stage 2(eventUnknown)': 'N2',
        'Sleep stage 3(eventUnknown)': 'N3',
        'Sleep stage R(eventUnknown)': 'REM'
    }

    # Собираем все эпохи сна в хронологическом порядке
    hypnogram = []
    current_time = 0

    for desc, duration in zip(annotations.description, annotations.duration):
        desc_str = str(desc)

        # Проверяем, что это стадия сна и эпоха длится ~30 секунд
        if desc_str in stage_mapping and abs(duration - 30) < 1:
            stage = stage_mapping[desc_str]
            start_time = current_time
            end_time = current_time + duration

            hypnogram.append({
                'stage': stage,
                'start_time_minutes': start_time / 60,
                'end_time_minutes': end_time / 60,
                'start_time_formatted': f"{int(start_time // 60):02d}:{int(start_time % 60):02d}",
                'end_time_formatted': f"{int(end_time // 60):02d}:{int(end_time % 60):02d}",
                'duration_minutes': duration / 60
            })

        current_time += duration

    return hypnogram

def get_hypnogram_statistics(hypnogram):
    """Расчет статистики гипнограммы"""
    if not hypnogram:
        return None

    stats = {
        'total_epochs': len(hypnogram),
        'total_duration_minutes': sum(epoch['duration_minutes'] for epoch in hypnogram),
        'stage_changes': 0,
        'stage_durations': {},
        'longest_stage_periods': {}
    }

    # Подсчет смен стадий и длительности каждой стадии
    previous_stage = None
    current_stage_duration = 0
    current_stage_start = 0

    for epoch in hypnogram:
        current_stage = epoch['stage']

        # Подсчет смен стадий
        if previous_stage is not None and current_stage != previous_stage:
            stats['stage_changes'] += 1

        # Накопление длительности для каждой стадии
        stats['stage_durations'][current_stage] = stats['stage_durations'].get(current_stage, 0) + epoch[
            'duration_minutes']

        previous_stage = current_stage

    # Поиск самых длинных непрерывных периодов для каждой стадии
    for stage in ['Wake', 'N1', 'N2', 'N3', 'REM']:
        stage_epochs = [epoch for epoch in hypnogram if epoch['stage'] == stage]

        if stage_epochs:
            # Группируем последовательные эпохи одной стадии
            continuous_periods = []
            current_period = []

            for epoch in hypnogram:
                if epoch['stage'] == stage:
                    current_period.append(epoch)
                else:
                    if current_period:
                        continuous_periods.append(current_period)
                        current_period = []

            if current_period:
                continuous_periods.append(current_period)

            # Находим самый длинный период
            if continuous_periods:
                longest_period = max(continuous_periods, key=len)
                stats['longest_stage_periods'][stage] = {
                    'duration_minutes': sum(epoch['duration_minutes'] for epoch in longest_period),
                    'epochs_count': len(longest_period),
                    'start_time': longest_period[0]['start_time_formatted'],
                    'end_time': longest_period[-1]['end_time_formatted']
                }

    return stats

def calculate_stage_transitions(epochs_sequence):
    """Расчет переходов между стадиями"""
    if not epochs_sequence:
        return {}

    transitions = {
        'total_transitions': 0,
        'transition_matrix': {
            'Wake': {'to_Wake': 0, 'to_N1': 0, 'to_N2': 0, 'to_N3': 0, 'to_REM': 0},
            'N1': {'to_Wake': 0, 'to_N1': 0, 'to_N2': 0, 'to_N3': 0, 'to_REM': 0},
            'N2': {'to_Wake': 0, 'to_N1': 0, 'to_N2': 0, 'to_N3': 0, 'to_REM': 0},
            'N3': {'to_Wake': 0, 'to_N1': 0, 'to_N2': 0, 'to_N3': 0, 'to_REM': 0},
            'REM': {'to_Wake': 0, 'to_N1': 0, 'to_N2': 0, 'to_N3': 0, 'to_REM': 0}
        }
    }

    previous_stage = None

    for epoch in epochs_sequence:
        current_stage = epoch['stage']

        if previous_stage is not None and previous_stage != current_stage:
            transitions['total_transitions'] += 1
            transitions['transition_matrix'][previous_stage][f'to_{current_stage}'] += 1

        previous_stage = current_stage

    return transitions

def calculate_rem_cycles(raw):
    """Расчет количества REM-циклов за ночь"""
    if not raw or not hasattr(raw, 'annotations'):
        return 0

    annotations = raw.annotations

    stage_mapping = {
        'Sleep stage W(eventUnknown)': 'W',
        'Sleep stage 1(eventUnknown)': 'N1',
        'Sleep stage 2(eventUnknown)': 'N2',
        'Sleep stage 3(eventUnknown)': 'N3',
        'Sleep stage R(eventUnknown)': 'R'
    }

    # Собираем последовательность стадий сна
    stages_sequence = []
    current_time = 0

    for desc, duration in zip(annotations.description, annotations.duration):
        desc_str = str(desc)
        if desc_str in stage_mapping and abs(duration - 30) < 1:
            stages_sequence.append(stage_mapping[desc_str])

    if not stages_sequence:
        return 0

    # Алгоритм определения REM-циклов
    rem_cycles = 0
    in_rem_cycle = False
    rem_cycle_started = False

    # REM-цикл определяется как последовательность: NREM -> REM -> NREM
    for i in range(1, len(stages_sequence) - 1):
        current_stage = stages_sequence[i]
        prev_stage = stages_sequence[i - 1]
        next_stage = stages_sequence[i + 1]

        # Начало REM-цикла: переход из NREM в REM
        if current_stage == 'R' and prev_stage in ['N1', 'N2', 'N3'] and not rem_cycle_started:
            rem_cycle_started = True
            in_rem_cycle = True

        # Конец REM-цикла: переход из REM в NREM
        elif current_stage == 'R' and next_stage in ['N1', 'N2', 'N3'] and in_rem_cycle:
            rem_cycles += 1
            in_rem_cycle = False
            rem_cycle_started = False

        # Прерывание REM-цикла пробуждением
        elif current_stage == 'W' and in_rem_cycle:
            in_rem_cycle = False
            rem_cycle_started = False

    return rem_cycles

def export_minimal_hypnogram(raw, output_file="hypnogram_minimal.json"):
    """Экспорт гипнограммы в минимальном формате"""
    if not raw or not hasattr(raw, 'annotations'):
        return None

    annotations = raw.annotations

    stage_mapping = {
        'Sleep stage W(eventUnknown)': 'W',
        'Sleep stage 1(eventUnknown)': '1',  # Однозначные цифры
        'Sleep stage 2(eventUnknown)': '2',
        'Sleep stage 3(eventUnknown)': '3',
        'Sleep stage R(eventUnknown)': 'R'
    }

    # Собираем только последовательность стадий
    stages_sequence = []

    for desc, duration in zip(annotations.description, annotations.duration):
        desc_str = str(desc)
        if desc_str in stage_mapping and abs(duration - 30) < 1:
            stages_sequence.append(stage_mapping[desc_str])

    # Минимальная структура данных
    minimal_data = {
        'e': len(stages_sequence),  # epochs
        'd': 30,  # duration per epoch
        's': stages_sequence  # stages
    }

    # Сохранение без пробелов
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(minimal_data, f, separators=(',', ':'), ensure_ascii=False)

    original_size = len(json.dumps({'stages_sequence': stages_sequence}))
    compressed_size = len(json.dumps(minimal_data))

    print(f"✅ Минимальная гипнограмма сохранена: {output_file}")
    print(f"📦 Сжатие: {original_size} → {compressed_size} байт "
          f"({(1 - compressed_size / original_size) * 100:.1f}% меньше)")

    return minimal_data

def export_hypnogram_to_json(raw, output_file=None):
    """Экспорт гипнограммы в JSON формат"""
    if not raw or not hasattr(raw, 'annotations'):
        return None

    annotations = raw.annotations

    # Маппинг аннотаций на стадии сна
    stage_mapping = {
        'Sleep stage W(eventUnknown)': 'Wake',
        'Sleep stage 1(eventUnknown)': 'N1',
        'Sleep stage 2(eventUnknown)': 'N2',
        'Sleep stage 3(eventUnknown)': 'N3',
        'Sleep stage R(eventUnknown)': 'REM'
    }

    # Собираем все эпохи сна в хронологическом порядке
    hypnogram_data = {
        'metadata': {
            'total_epochs': 0,
            'total_duration_minutes': 0,
            'epoch_duration_seconds': 30,
            'export_timestamp': datetime.now().isoformat(),
            'recording_start_time': None
        },
        'stages_summary': {
            'Wake': {'epochs': 0, 'minutes': 0},
            'N1': {'epochs': 0, 'minutes': 0},
            'N2': {'epochs': 0, 'minutes': 0},
            'N3': {'epochs': 0, 'minutes': 0},
            'REM': {'epochs': 0, 'minutes': 0}
        },
        'epochs_sequence': []
    }

    current_time = 0
    recording_start = None

    for desc, duration in zip(annotations.description, annotations.duration):
        desc_str = str(desc)

        # Проверяем, что это стадия сна и эпоха длится ~30 секунд
        if desc_str in stage_mapping and abs(duration - 30) < 1:
            stage = stage_mapping[desc_str]

            # Определяем время начала записи по первой эпохе
            if recording_start is None:
                recording_start = current_time

            epoch_data = {
                'epoch_index': len(hypnogram_data['epochs_sequence']),
                'stage': stage,
                'start_time_seconds': current_time,
                'end_time_seconds': current_time + duration,
                'start_time_minutes': current_time / 60,
                'end_time_minutes': (current_time + duration) / 60,
                'duration_seconds': duration,
                'start_time_formatted': format_time(current_time),
                'end_time_formatted': format_time(current_time + duration)
            }

            hypnogram_data['epochs_sequence'].append(epoch_data)

            # Обновляем статистику
            hypnogram_data['stages_summary'][stage]['epochs'] += 1
            hypnogram_data['stages_summary'][stage]['minutes'] += duration / 60

        current_time += duration

    # Обновляем метаданные
    hypnogram_data['metadata']['total_epochs'] = len(hypnogram_data['epochs_sequence'])
    hypnogram_data['metadata']['total_duration_minutes'] = sum(
        stage['minutes'] for stage in hypnogram_data['stages_summary'].values()
    )

    if recording_start is not None:
        hypnogram_data['metadata']['recording_start_time'] = format_time(recording_start)

    # Добавляем статистику переходов между стадиями
    hypnogram_data['transitions'] = calculate_stage_transitions(hypnogram_data['epochs_sequence'])

    # Сохраняем в файл если указан output_file
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(hypnogram_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Гипнограмма сохранена в: {output_file}")
        except Exception as e:
            print(f"❌ Ошибка сохранения JSON: {e}")

    return hypnogram_data

def print_hypnogram_statistics(stats):
    """Вывод статистики гипнограммы"""
    if not stats:
        return

    print("\n📊 СТАТИСТИКА ГИПНОГРАММЫ")
    print("=" * 50)
    print(f"Всего эпох: {stats['total_epochs']}")
    print(f"Общая продолжительность: {stats['total_duration_minutes']:.1f} мин")
    print(f"Количество смен стадий: {stats['stage_changes']}")

    print(f"\nОбщая длительность по стадиям:")
    for stage in ['Wake', 'N1', 'N2', 'N3', 'REM']:
        duration = stats['stage_durations'].get(stage, 0)
        percentage = (duration / stats['total_duration_minutes'] * 100) if stats['total_duration_minutes'] > 0 else 0
        print(f"  {stage}: {duration:.1f} мин ({percentage:.1f}%)")

    print(f"\nСамые длинные непрерывные периоды:")
    for stage in ['Wake', 'N1', 'N2', 'N3', 'REM']:
        period_info = stats['longest_stage_periods'].get(stage)
        if period_info:
            print(f"  {stage}: {period_info['duration_minutes']:.1f} мин "
                  f"({period_info['epochs_count']} эпох) "
                  f"[{period_info['start_time']}-{period_info['end_time']}]")

def print_hypnogram_report(hypnogram, max_epochs=50):
    """Вывод гипнограммы в текстовом формате"""
    if not hypnogram:
        print("❌ Не удалось сгенерировать гипнограмму")
        return

    print("\n📈 ГИПНОГРАММА СНА")
    print("=" * 70)
    print(f"{'Время':<12} | {'Стадия':<6} | {'Длит. (мин)':<10} | {'Период'}")
    print("-" * 70)

    # Выводим первые max_epochs эпох или все, если их меньше
    display_epochs = hypnogram[:max_epochs]

    for i, epoch in enumerate(display_epochs):
        time_range = f"{epoch['start_time_formatted']}-{epoch['end_time_formatted']}"
        stage_symbol = {
            'Wake': 'W',
            'N1': '1',
            'N2': '2',
            'N3': '3',
            'REM': 'R'
        }.get(epoch['stage'], epoch['stage'])

        print(f"{time_range:<12} | {stage_symbol:<6} | {epoch['duration_minutes']:<10.1f} | {epoch['stage']}")

    # Если эпох больше, чем выводим, показываем статистику
    if len(hypnogram) > max_epochs:
        print(f"... и еще {len(hypnogram) - max_epochs} эпох")

    print("-" * 70)

    # Статистика по гипнограмме
    total_epochs = len(hypnogram)
    total_duration = sum(epoch['duration_minutes'] for epoch in hypnogram)

    print(f"Всего эпох: {total_epochs}")
    print(f"Общая продолжительность: {total_duration:.1f} мин ({total_duration / 60:.1f} часов)")

    # Распределение по стадиям
    stage_counts = {}
    for epoch in hypnogram:
        stage = epoch['stage']
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    print("\nРаспределение эпох по стадиям:")
    for stage in ['Wake', 'N1', 'N2', 'N3', 'REM']:
        count = stage_counts.get(stage, 0)
        percentage = (count / total_epochs * 100) if total_epochs > 0 else 0
        print(f"  {stage}: {count} эпох ({percentage:.1f}%)")

def print_hypnogram_json_summary(hypnogram_data, max_epochs_display=10):
    """Краткий вывод гипнограммы в читаемом формате"""
    if not hypnogram_data:
        print("❌ Нет данных гипнограммы")
        return

    metadata = hypnogram_data['metadata']
    stages_summary = hypnogram_data['stages_summary']
    transitions = hypnogram_data['transitions']

    print("\n📊 JSON ГИПНОГРАММА - ОБЗОР")
    print("=" * 60)
    print(f"Всего эпох: {metadata['total_epochs']}")
    print(f"Общая продолжительность: {metadata['total_duration_minutes']:.1f} мин")
    print(f"Длительность эпохи: {metadata['epoch_duration_seconds']} сек")
    print(f"Время начала: {metadata['recording_start_time']}")

    print(f"\n📈 РАСПРЕДЕЛЕНИЕ СТАДИЙ:")
    for stage, data in stages_summary.items():
        percentage = (data['minutes'] / metadata['total_duration_minutes'] * 100) if metadata[
                                                                                         'total_duration_minutes'] > 0 else 0
        print(f"  {stage}: {data['epochs']} эпох, {data['minutes']:.1f} мин ({percentage:.1f}%)")

    print(f"\n🔄 ПЕРЕХОДЫ МЕЖДУ СТАДИЯМИ:")
    print(f"  Всего переходов: {transitions['total_transitions']}")

    print(f"\n⏱️  ПЕРВЫЕ {max_epochs_display} ЭПОХ:")
    for epoch in hypnogram_data['epochs_sequence'][:max_epochs_display]:
        print(
            f"  {epoch['epoch_index']:3d}. {epoch['start_time_formatted']} - {epoch['end_time_formatted']} | {epoch['stage']}")

    if len(hypnogram_data['epochs_sequence']) > max_epochs_display:
        print(f"  ... и еще {len(hypnogram_data['epochs_sequence']) - max_epochs_display} эпох")

def print_channel_info(raw):
    """Вывод подробной информации о каналах EDF файла"""
    if not raw:
        print("❌ Нет данных для анализа каналов")
        return

    print("\n📡 ИНФОРМАЦИЯ О КАНАЛАХ EDF")
    print("=" * 80)

    # Основная информация о записи
    print(f"📊 Общая информация:")
    print(f"  • Количество каналов: {len(raw.ch_names)}")
    print(f"  • Длительность записи: {raw.times[-1] / 60:.1f} мин ({raw.times[-1] / 3600:.1f} часов)")
    print(f"  • Частота дискретизации: {raw.info['sfreq']} Гц")
    print(f"  • Общее количество samples: {len(raw.times)}")

    # Получаем реальные типы каналов
    channel_types = raw.get_channel_types()

    # Группируем каналы по типам
    channel_groups = {
        'ЭЭГ (EEG)': [],
        'ЭКГ (ECG)': [],
        'ЭОГ (EOG)': [],
        'ЭМГ (EMG)': [],
        'Дыхание (Respiratory)': [],
        'Сатурация (SpO2)': [],
        'Пульс': [],
        'Положение тела': [],
        'Звук': [],
        'Другие': []
    }

    eeg_keywords = ['eeg', 'c3', 'c4', 'a1', 'a2', 'f3', 'f4', 'o1', 'o2', 'fz', 'cz', 'pz']
    ecg_keywords = ['ecg', 'ekg', 'electrocardiogram', 'экг', 'кардиограмма']
    eog_keywords = ['eog', 'electrooculogram', 'эог']
    emg_keywords = ['emg', 'electromyogram', 'эмг', 'подбородок', 'chin']
    resp_keywords = ['resp', 'breath', 'дыхание', 'thorax', 'chest', 'abdomen', 'flow', 'pressure']
    spo2_keywords = ['spo2', 'sao2', 'sat', 'saturation', 'сатурация']
    pulse_keywords = ['pulse', 'puls', 'пульс', 'chastota']
    position_keywords = ['position', 'pos', 'положение', 'body']
    sound_keywords = ['sound', 'snore', 'храп']

    for i, channel in enumerate(raw.ch_names):
        channel_lower = channel.lower()
        actual_type = channel_types[i]

        if any(keyword in channel_lower for keyword in eeg_keywords):
            channel_groups['ЭЭГ (EEG)'].append((channel, actual_type))
        elif any(keyword in channel_lower for keyword in ecg_keywords):
            channel_groups['ЭКГ (ECG)'].append((channel, actual_type))
        elif any(keyword in channel_lower for keyword in eog_keywords):
            channel_groups['ЭОГ (EOG)'].append((channel, actual_type))
        elif any(keyword in channel_lower for keyword in emg_keywords):
            channel_groups['ЭМГ (EMG)'].append((channel, actual_type))
        elif any(keyword in channel_lower for keyword in resp_keywords):
            channel_groups['Дыхание (Respiratory)'].append((channel, actual_type))
        elif any(keyword in channel_lower for keyword in spo2_keywords):
            channel_groups['Сатурация (SpO2)'].append((channel, actual_type))
        elif any(keyword in channel_lower for keyword in pulse_keywords):
            channel_groups['Пульс'].append((channel, actual_type))
        elif any(keyword in channel_lower for keyword in position_keywords):
            channel_groups['Положение тела'].append((channel, actual_type))
        elif any(keyword in channel_lower for keyword in sound_keywords):
            channel_groups['Звук'].append((channel, actual_type))
        else:
            channel_groups['Другие'].append((channel, actual_type))

    # Выводим каналы по группам
    for group_name, channels in channel_groups.items():
        if channels:
            print(f"\n{group_name} ({len(channels)}):")
            for i, (channel, chan_type) in enumerate(channels):
                sfreq = raw.info['sfreq']
                print(f"  {i + 1:2d}. {channel:<25} | {chan_type:<10} | {sfreq} Гц")

    # Статистика по типам каналов
    print(f"\n📈 СТАТИСТИКА КАНАЛОВ:")
    total_channels = len(raw.ch_names)
    for group_name, channels in channel_groups.items():
        if channels:
            percentage = (len(channels) / total_channels) * 100
            print(f"  • {group_name}: {len(channels)} каналов ({percentage:.1f}%)")

    # Анализ качества сигналов
    print(f"\n🔍 АНАЛИЗ КАЧЕСТВА СИГНАЛОВ:")
    analyze_signal_quality(raw, channel_groups)

def print_sleep_report(edf_path):
    """Итоговый отчет по анализу сна"""
    raw, annotation_counts = extract_all_annotations(edf_path)
    if not raw:
        print("❌ Ошибка загрузки файла или отсутствуют аннотации")
        return

    # В НАЧАЛЕ ОТЧЕТА добавить:
    print_channel_info(raw)

    # Остальной существующий код...
    artifact_mask, artifact_regions = get_artifact_masks(raw)
    if artifact_regions:
        print(f"🚫 Исключено регионов с артефактами: {len(artifact_regions)}")
        total_artifact_time = sum(region['duration'] for region in artifact_regions)
        print(f"⏱️  Общее время артефактов: {total_artifact_time / 60:.1f} мин")

    print("📊 СТАТИСТИКА АННОТАЦИЙ")
    print("=" * 50)
    print(f"Всего аннотаций: {len(raw.annotations)}")
    print(f"Уникальных типов: {len(annotation_counts)}")
    print("\nТипы аннотаций:")

    for desc, count in sorted(annotation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {count:>5} × {desc}")

    stages = calculate_sleep_stages(raw)
    if not stages:
        print("❌ Не удалось рассчитать стадии сна")
        return

    print("\n🛌 СТАДИИ СНА")
    print("-" * 40)
    print(f"{'Стадия':<6} | {'Эпох':<6} | {'Минут':<8} | {'%':<6}")
    print("-" * 40)

    total_minutes = sum(stage['minutes'] for stage in stages.values())

    for stage, data in stages.items():
        percentage = (data['minutes'] / total_minutes * 100) if total_minutes > 0 else 0
        print(f"{stage:<6} | {data['count']:<6} | {data['minutes']:<8.1f} | {percentage:<6.1f}")

    print("-" * 40)
    total_epochs = sum(stage['count'] for stage in stages.values())
    print(f"{'ИТОГО':<6} | {total_epochs:<6} | {total_minutes:<8.1f} | 100.0")

    print(f"\n💙 ФИЗИОЛОГИЧЕСКИЕ ПАРАМЕТРЫ")
    print("-" * 50)

    spo2_stats = analyze_spo2_channel_fast(raw)
    if spo2_stats.get('artifact_regions_excluded', 0) > 0:
        print(f"• Исключено артефактов: {spo2_stats['artifact_regions_excluded']}")

    if spo2_stats['avg_spo2']:
        print(f"🌡️  САТУРАЦИЯ (SpO2):")
        print(f"• Средняя: {spo2_stats['avg_spo2']}%")
        print(f"• Минимальная: {spo2_stats['min_spo2']}%")
        print(f"• Время <90%: {spo2_stats['time_below_spo2_90']} мин")
        print(f"• Время <85%: {spo2_stats['time_below_spo2_85']} мин")
        print(f"• Базовый уровень: {spo2_stats['spo2_baseline']}%")

    hr_stats = analyze_heart_rate_comprehensive(raw)
    if hr_stats.get('artifact_regions_excluded', 0) > 0:
        print(f"• Исключено артефактов: {hr_stats['artifact_regions_excluded']}")
    if hr_stats['avg_heart_rate']:
        print(f"\n❤️  СЕРДЕЧНЫЙ РИТМ:")
        print(f"• Средний: {hr_stats['avg_heart_rate']} уд/мин")
        print(f"• Минимальный: {hr_stats['min_heart_rate']} уд/мин")
        print(f"• Максимальный: {hr_stats['max_heart_rate']} уд/мин")
        print(f"• ВСР (вариабельность): {hr_stats['heart_rate_variability']} мс")
        print(f"• Эпизоды тахикардии: {hr_stats.get('tachycardia_events')}")
        print(f"• Эпизоды брадикардии: {hr_stats.get('bradycardia_events')}")


    resp_stats = analyze_respiratory_channels_improved(raw)
    if resp_stats['avg_resp_rate']:
        print(f"\n💨 ЧАСТОТА ДЫХАНИЯ:")
        print(f"• Средняя: {resp_stats['avg_resp_rate']} дых/мин")
        print(f"• Минимальная: {resp_stats['min_resp_rate']} дых/мин")
        print(f"• Максимальная: {resp_stats['max_resp_rate']} дых/мин")

    rem_quality = calculate_rem_quality(raw)
    rem_cycles = calculate_rem_cycles(raw)
    if rem_quality:
        print(f"\n🌟 КАЧЕСТВО REM-СНА:")
        print(f"• Оценка качества: {rem_quality['rem_quality_score']}/100 ({rem_quality['status']})")
        print(f"• REM-время: {rem_quality['rem_minutes']:.1f} мин")
        print(f"• REM-события: {rem_quality['rem_events']}")
        print(f"• REM-циклы: {rem_cycles}")
        print(f"• Плотность REM: {rem_quality['rem_density']:.2f} событий/мин")

    efficiency_data = calculate_sleep_efficiency(raw, stages)
    if efficiency_data:
        print(f"\n⚡ ЭФФЕКТИВНОСТЬ СНА:")
        print(f"• Эффективность: {efficiency_data['sleep_efficiency']:.1f}%")
        print(f"• Общее время сна: {efficiency_data['total_sleep_time']:.1f} мин")
        print(f"• Время в кровати: {efficiency_data['total_bed_time']:.1f} мин")
        print(f"• Бодрствование после засыпания: {efficiency_data['wake_after_sleep_onset']:.1f} мин")

    latency_data = calculate_sleep_latencies(raw, stages)
    if latency_data:
        print(f"\n⏰ ЛАТЕНТНОСТИ:")
        if latency_data['sleep_onset_latency']:
            print(f"• Латентность засыпания: {latency_data['sleep_onset_latency']:.1f} мин")
        if latency_data['rem_latency']:
            print(f"• Латентность REM-сна: {latency_data['rem_latency']:.1f} мин")

    architecture_data = calculate_sleep_architecture(stages)
    if architecture_data:
        print(f"\n🏛️  АРХИТЕКТУРА СНА:")
        print(f"• N1 (поверхностный): {architecture_data['n1_percentage']:.1f}%")
        print(f"• N2 (средней глубины): {architecture_data['n2_percentage']:.1f}%")
        print(f"• N3 (глубокий): {architecture_data['n3_percentage']:.1f}%")
        print(f"• REM: {architecture_data['rem_percentage']:.1f}%")
        print(f"• Соотношение REM/NREM: {architecture_data['rem_nrem_ratio']:.2f}")

    sleep_indices = calculate_sleep_indices(raw, stages)
    if sleep_indices:
        print(f"\n🌬️  ДЫХАТЕЛЬНЫЕ НАРУШЕНИЯ:")
        print(f"• Индекс AHI: {sleep_indices['ahi']:.1f} ({sleep_indices['ahi_severity']})")
        print(f"• Апноэ: {sleep_indices['total_apneas']} событий")
        print(f"• Гипопноэ: {sleep_indices['total_hypopneas']} событий")
        print(f"• Индекс десатураций: {sleep_indices['odi']:.1f}")
        print(f"• Индекс храпа: {sleep_indices['snoring_index']:.1f}")

    fragmentation_data = calculate_sleep_fragmentation(raw, stages)
    if fragmentation_data:
        print(f"\n🔍 ФРАГМЕНТАЦИЯ СНА:")
        print(f"• Индекс фрагментации: {fragmentation_data['fragmentation_index']:.1f}")
        print(f"• Активации: {fragmentation_data['activations']}")
        print(f"• Движения конечностей: {fragmentation_data['limb_movements']}")
        print(f"• Периодические движения: {fragmentation_data['periodic_limb_movements']}")
        print(f"• Всего движений: {fragmentation_data['total_limb_movements']}")
        print(f"• Индекс активаций: {fragmentation_data['arousal_index']:.1f}")

    # ДОБАВЛЕННЫЙ КОД ДЛЯ ГИПНОГРАММЫ
    print(f"\n📈 ГИПНОГРАММА И РАСПРЕДЕЛЕНИЕ СТАДИЙ")
    print("-" * 50)

    # ДОБАВЛЕННЫЙ КОД ДЛЯ JSON ЭКСПОРТА
    print(f"\n📊 JSON ЭКСПОРТ ГИПНОГРАММЫ")
    print("-" * 50)

    # Полный экспорт
    hypnogram_json = export_hypnogram_to_json(raw, "hypnogram_full.json")
    if hypnogram_json:
        print_hypnogram_json_summary(hypnogram_json, max_epochs_display=15)

    # Компактный экспорт
    compact_json = export_minimal_hypnogram(raw, "hypnogram_compact.json")

def generate_sql_insert(edf_path, study_id=None):
    """Генерация SQL запроса для вставки данных в БД"""

    # Загрузка и анализ данных
    raw, annotation_counts = extract_all_annotations(edf_path)
    if not raw:
        print("❌ Ошибка загрузки файла или отсутствуют аннотации")
        return None

    # Извлечение UUID
    patient_info = extract_patient_info_from_edf(edf_path)
    if not patient_info or not patient_info['uuid']:
        print("❌ Не найден UUID в файле EDF")
        return None

    uuid = patient_info['uuid']

    # Расчет статистики артефактов для psg_studies
    artifact_mask, artifact_regions = get_artifact_masks(raw)
    artifact_count = len(artifact_regions) if artifact_regions else 0
    artifact_duration_minutes = sum(region['duration'] for region in artifact_regions) / 60 if artifact_regions else 0

    # Расчет всех показателей с проверкой на None
    stages = calculate_sleep_stages(raw)
    spo2_stats = analyze_spo2_channel_fast(raw) or {}
    hr_stats = analyze_heart_rate_comprehensive(raw) or {}
    resp_stats = analyze_respiratory_channels_improved(raw) or {}
    efficiency_data = calculate_sleep_efficiency(raw, stages) or {}
    latency_data = calculate_sleep_latencies(raw, stages) or {}
    architecture_data = calculate_sleep_architecture(stages) or {}
    sleep_indices = calculate_sleep_indices(raw, stages) or {}
    fragmentation_data = calculate_sleep_fragmentation(raw, stages) or {}
    rem_quality = calculate_rem_quality(raw) or {}
    total_sleep_time = efficiency_data.get('total_sleep_time', 0)

    # Расчет общей оценки качества сна
    overall_quality = calculate_overall_sleep_quality(raw, stages) or {}

    # Получение дыхательных событий отдельно для безопасности
    respiratory_events = calculate_respiratory_events(raw) or {}

    # Генерация гипнограммы
    hypnogram_data = export_minimal_hypnogram(raw)

    # Расчет REM-циклов
    rem_cycles_count = calculate_rem_cycles(raw)

    # Подготовка данных для SQL UPDATE (только для sleep_statistics)
    sql_data = {
        # Общие параметры сна
        'total_sleep_time': int(efficiency_data.get('total_sleep_time', 0)),
        'total_bed_time': int(efficiency_data.get('total_bed_time', 0)),
        'sleep_efficiency': round(efficiency_data.get('sleep_efficiency', 0), 2),
        'sleep_latency': int(latency_data.get('sleep_onset_latency', 0)) if latency_data.get(
            'sleep_onset_latency') else 0,
        'wake_after_sleep_onset': int(efficiency_data.get('wake_after_sleep_onset', 0)),

        # Стадии сна (минуты)
        'n1_minutes': int(stages['N1']['minutes']) if stages else 0,
        'n2_minutes': int(stages['N2']['minutes']) if stages else 0,
        'n3_minutes': int(stages['N3']['minutes']) if stages else 0,
        'rem_minutes': int(stages['REM']['minutes']) if stages else 0,

        # Стадии сна (проценты)
        'n1_percentage': round(architecture_data.get('n1_percentage', 0), 2),
        'n2_percentage': round(architecture_data.get('n2_percentage', 0), 2),
        'n3_percentage': round(architecture_data.get('n3_percentage', 0), 2),
        'rem_percentage': round(architecture_data.get('rem_percentage', 0), 2),

        # REM-сон
        'rem_latency': int(latency_data.get('rem_latency')) if latency_data.get('rem_latency') else None,
        'rem_epochs': stages['REM']['count'] if stages else None,
        'rem_cycles': rem_cycles_count,
        'rem_events': rem_quality.get('rem_events'),
        'rem_density': round(rem_quality.get('rem_density', 0), 2) if rem_quality.get('rem_density') else None,
        'rem_quality_score': rem_quality.get('rem_quality_score'),

        # Дыхательные нарушения (события)
        'total_apneas': respiratory_events.get('apneas', 0),
        'obstructive_apneas': respiratory_events.get('apneas', 0),  # Упрощенно
        'central_apneas': 0,  # Требует дополнительного анализа
        'mixed_apneas': 0,  # Требует дополнительного анализа
        'total_hypopneas': respiratory_events.get('hypopneas', 0),
        'total_desaturations': respiratory_events.get('desaturations', 0),
        'total_snores': respiratory_events.get('snoring', 0),

        # Дыхательные индексы
        'ahi': round(sleep_indices.get('ahi', 0), 2),
        'ahi_obstructive': round(sleep_indices.get('ahi', 0), 2),  # Упрощенно
        'ahi_central': 0,  # Требует дополнительного анализа
        'odi': round(sleep_indices.get('odi', 0), 2),
        'snore_index': round(sleep_indices.get('snoring_index', 0), 2),

        # Сатурация кислорода
        'avg_spo2': spo2_stats.get('avg_spo2'),
        'min_spo2': spo2_stats.get('min_spo2'),
        'spo2_baseline': spo2_stats.get('spo2_baseline'),
        'time_below_spo2_90': spo2_stats.get('time_below_spo2_90', 0),
        'time_below_spo2_85': spo2_stats.get('time_below_spo2_85', 0),

        # Сердечный ритм
        'avg_heart_rate': hr_stats.get('avg_heart_rate'),
        'min_heart_rate': hr_stats.get('min_heart_rate'),
        'max_heart_rate': hr_stats.get('max_heart_rate'),
        'heart_rate_variability': hr_stats.get('heart_rate_variability'),
        'tachycardia_events': hr_stats.get('tachycardia_events'),
        'bradycardia_events': hr_stats.get('bradycardia_events'),

        # Дыхательная система
        'avg_resp_rate': resp_stats.get('avg_resp_rate'),
        'min_resp_rate': resp_stats.get('min_resp_rate'),
        'max_resp_rate': resp_stats.get('max_resp_rate'),

        # Двигательные нарушения
        'total_limb_movements': fragmentation_data.get('total_limb_movements', 0),
        'periodic_limb_movements': fragmentation_data.get('periodic_limb_movements', 0),
        'plmi': round(fragmentation_data.get('periodic_limb_movements', 0) / (total_sleep_time / 60), 2) if total_sleep_time > 0 else 0,
        'bruxism_events': 0,  # Требует дополнительного анализа

        # Активации и фрагментация
        'total_arousals': fragmentation_data.get('activations', 0),
        'arousal_index': round(fragmentation_data.get('arousal_index', 0), 2),
        'sleep_fragmentation_index': round(fragmentation_data.get('fragmentation_index', 0), 2),

        # Общая оценка качества сна
        'overall_sleep_quality': overall_quality.get('overall_score'),
        'sleep_quality_status': overall_quality.get('status'),

        # Гипнограмма
        'hypnogram_data': json.dumps(hypnogram_data) if hypnogram_data else None,

        # Метаданные
        'data_quality': 'good',
        'analysis_notes': f"Автоматический анализ файла: {os.path.basename(edf_path)}",
        'calculated_at': 'NOW()'
    }

    # Генерация SQL UPDATE запроса для sleep_statistics
    set_parts = []
    for key, value in sql_data.items():
        if key == 'calculated_at':
            set_parts.append(f"`{key}` = {value}")
        elif value is None:
            set_parts.append(f"`{key}` = NULL")
        elif isinstance(value, str):
            # Экранирование кавычек в строках
            escaped_value = value.replace("'", "''")
            set_parts.append(f"`{key}` = '{escaped_value}'")
        else:
            set_parts.append(f"`{key}` = {value}")

    # Генерация SQL UPDATE запросов для обеих таблиц
    sql = f"""-- Обновление статистики сна
UPDATE `sleep_statistics` ss
JOIN `psg_studies` ps ON ss.study_id = ps.study_id
SET {', '.join(set_parts)}
WHERE ps.edf_uuid = '{uuid}';

-- Обновление информации об артефактах в psg_studies
UPDATE `psg_studies` 
SET `artifact_count` = {artifact_count}, 
    `artifact_duration_minutes` = {round(artifact_duration_minutes, 2)}
WHERE `edf_uuid` = '{uuid}';"""

    # Создание SQL файла
    sql_filename = f"sleep_stats_{uuid}.sql"
    try:
        with open(sql_filename, 'w', encoding='utf-8') as f:
            f.write("-- SQL запрос для обновления статистики сна\n")
            f.write(f"-- UUID исследования: {uuid}\n")
            f.write(f"-- Файл: {os.path.basename(edf_path)}\n")
            f.write(f"-- Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("-- ВАЖНО: Эта запись должна выполняться ПОСЛЕ импорта исследования через процедуру\n")
            f.write("-- Исследование должно уже существовать в таблицах psg_studies и sleep_statistics\n\n")
            f.write(sql + "\n")

        print(f"✅ SQL файл создан: {sql_filename}")
        print(f"📊 UUID исследования: {uuid}")
        print(f"📝 Тип запроса: UPDATE (обновление существующей записи)")
        print(f"🚫 Артефакты: {artifact_count} регионов, {round(artifact_duration_minutes, 2)} минут")

        return sql_filename

    except Exception as e:
        print(f"❌ Ошибка создания SQL файла: {e}")
        return None

def analyze_heart_rate_comprehensive(raw):
    """
    Комплексный анализ сердечного ритма из ЭКГ:
    - базовые статистики ЧСС
    - вариабельность сердечного ритма
    - эпизоды тахикардии и брадикардии
    - исключение артефактов
    """

    results = {
        # Базовые статистики
        'avg_heart_rate': None,
        'min_heart_rate': None,
        'max_heart_rate': None,
        'heart_rate_variability': None,
        'artifact_regions_excluded': 0,

        # Эпизоды
        'tachycardia_events': 0,
        'bradycardia_events': 0,

        # Метод анализа
        'analysis_method': 'ecg'  # 'ecg' или 'markers'
    }

    try:
        # Поиск ЭКГ канала (общая логика)
        ecg_keywords = ['ecg', 'ekg', 'electrocardiogram', 'экг', 'кардиограмма']
        ecg_channels = [
            ch for ch in raw.ch_names
            if any(keyword in ch.lower() for keyword in ecg_keywords)
        ]

        if not ecg_channels:
            # Если нет ЭКГ каналов - используем анализ по маркерам
            return _analyze_heart_rate_from_markers(raw)

        # Настройки анализа
        ecg_ch = ecg_channels[0]
        ecg_idx = raw.ch_names.index(ecg_ch)
        sfreq = raw.info['sfreq']
        max_samples = len(raw.times)

        # Получаем данные
        data, times = raw[ecg_idx, :max_samples]
        if len(data) == 0:
            return _analyze_heart_rate_from_markers(raw)

        ecg_signal = data[0]

        # Обнаружение артефактов и R-пиков
        artifact_mask, artifact_regions = get_artifact_masks(raw)
        r_peaks = _get_clean_r_peaks(ecg_signal, sfreq, artifact_mask, artifact_regions, max_samples)
        results['artifact_regions_excluded'] = len(artifact_regions)

        if len(r_peaks) <= 100:
            return _analyze_heart_rate_from_markers(raw)

        # Расчет RR-интервалов и ЧСС
        rr_intervals, heart_rates = _calculate_heart_rate_metrics(r_peaks, sfreq)

        if len(heart_rates) > 5:
            # Базовые статистики
            results.update(_calculate_basic_stats(heart_rates, rr_intervals))

            # Анализ эпизодов
            results.update(_detect_heart_rate_episodes(heart_rates))

        else:
            results = _analyze_heart_rate_from_markers(raw)

    except Exception as e:
        print(f"⚠️ Ошибка комплексного анализа ЭКГ: {e}")
        results = _analyze_heart_rate_from_markers(raw)

    return results

def _get_clean_r_peaks(ecg_signal, sfreq, artifact_mask, artifact_regions, max_samples):
    """Обнаружение R-пиков с исключением артефактов"""
    if artifact_mask is not None and len(artifact_mask) >= max_samples:
        segment_mask = artifact_mask[:max_samples]
        valid_segments = find_continuous_segments(segment_mask, min_segment_length=int(sfreq * 10))

        all_r_peaks = []
        for start, end in valid_segments:
            segment_ecg = ecg_signal[start:end]
            if len(segment_ecg) > int(sfreq * 5):
                segment_peaks = detect_r_peaks(segment_ecg, sfreq)
                segment_peaks += start
                all_r_peaks.extend(segment_peaks)

        return np.array(all_r_peaks)
    else:
        return detect_r_peaks(ecg_signal, sfreq)

def _calculate_heart_rate_metrics(r_peaks, sfreq):
    """Расчет RR-интервалов и ЧСС с фильтрацией"""
    rr_intervals = np.diff(r_peaks) / sfreq

    # Фильтрация физиологически возможных интервалов
    valid_rr_mask = (rr_intervals > 0.3) & (rr_intervals < 2.0)
    valid_rr = rr_intervals[valid_rr_mask]

    if len(valid_rr) > 1:
        heart_rates = 60.0 / valid_rr
        # Фильтрация физиологически возможных значений ЧСС
        valid_hr_mask = (heart_rates >= 40) & (heart_rates <= 180)
        valid_hr = heart_rates[valid_hr_mask]
        return valid_rr, valid_hr

    return np.array([]), np.array([])

def _calculate_basic_stats(heart_rates, rr_intervals):
    """Расчет базовых статистик ЧСС"""
    return {
        'avg_heart_rate': round(float(np.median(heart_rates)), 2),
        'min_heart_rate': round(float(np.percentile(heart_rates, 5)), 2),
        'max_heart_rate': round(float(np.percentile(heart_rates, 95)), 2),
        'heart_rate_variability': round(float(np.std(rr_intervals * 1000)), 2)
    }

def _detect_heart_rate_episodes(heart_rates):
    """Обнаружение эпизодов тахикардии и брадикардии"""
    episodes = {
        'tachycardia_events': 0,
        'bradycardia_events': 0
    }

    tachycardia_threshold = 100
    bradycardia_threshold = 50
    min_consecutive = 7

    tachy_count = 0
    brady_count = 0
    tachy_episode = False
    brady_episode = False

    for hr in heart_rates:
        # Тахикардия
        if hr > tachycardia_threshold:
            tachy_count += 1
            if tachy_count >= min_consecutive and not tachy_episode:
                episodes['tachycardia_events'] += 1
                tachy_episode = True
        else:
            tachy_count = 0
            tachy_episode = False

        # Брадикардия
        if hr < bradycardia_threshold:
            brady_count += 1
            if brady_count >= min_consecutive and not brady_episode:
                episodes['bradycardia_events'] += 1
                brady_episode = True
        else:
            brady_count = 0
            brady_episode = False

    return episodes

def _analyze_heart_rate_from_markers(raw):
    """Резервный анализ по маркерам аннотаций"""
    results = {
        'avg_heart_rate': None,
        'min_heart_rate': None,
        'max_heart_rate': None,
        'heart_rate_variability': None,
        'artifact_regions_excluded': 0,
        'tachycardia_events': 0,
        'bradycardia_events': 0,
        'analysis_method': 'markers'
    }

    try:
        annotations = raw.annotations
        for desc in annotations.description:
            desc_str = str(desc)
            if 'Тахикардия' in desc_str:
                results['tachycardia_events'] += 1
            elif 'Брадикардия' in desc_str:
                results['bradycardia_events'] += 1
    except Exception as e:
        print(f"⚠️ Ошибка анализа по маркерам: {e}")

    return results

def find_continuous_segments(mask, min_segment_length=1):
    """Находит непрерывные сегменты в маске"""
    segments = []
    start = None

    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= min_segment_length:
                segments.append((start, i))
            start = None

    # Проверяем последний сегмент
    if start is not None and len(mask) - start >= min_segment_length:
        segments.append((start, len(mask)))

    return segments

def detect_r_peaks(ecg_signal, sfreq):
    """Детекция R-зубцов в ЭКГ сигнале"""
    try:
        ecg_clean = ecg_signal - np.median(ecg_signal)

        if len(ecg_clean) > 100:
            b, a = signal.butter(3, [0.5 / (sfreq / 2), 40 / (sfreq / 2)], btype='band')
            ecg_filtered = signal.filtfilt(b, a, ecg_clean)
        else:
            ecg_filtered = ecg_clean

        ecg_squared = np.square(ecg_filtered)
        window_size = int(0.1 * sfreq)
        if window_size % 2 == 0:
            window_size += 1
        ecg_smoothed = signal.medfilt(ecg_squared, kernel_size=window_size)

        threshold = np.percentile(ecg_smoothed, 85)
        peaks, _ = signal.find_peaks(ecg_smoothed, height=threshold, distance=int(0.3 * sfreq))

        return peaks

    except Exception as e:
        return np.array([])

def get_artifact_masks(raw, artifact_marker='Артефакт(blockArtefact)'):
    """Создание масок для исключения участков с артефактами"""
    if not raw or not hasattr(raw, 'annotations'):
        return None, None

    annotations = raw.annotations
    sfreq = raw.info['sfreq']
    total_samples = len(raw.times)

    # Создаем маску (True = валидные данные, False = артефакт)
    valid_mask = np.ones(total_samples, dtype=bool)

    current_time = 0
    artifact_regions = []

    for desc, duration, onset in zip(annotations.description, annotations.duration, annotations.onset):
        desc_str = str(desc)

        if artifact_marker in desc_str:
            start_sample = int(onset * sfreq)
            end_sample = int((onset + duration) * sfreq)
            end_sample = min(end_sample, total_samples - 1)

            # Помечаем регион артефакта как невалидный
            if start_sample < total_samples:
                valid_mask[start_sample:end_sample] = False
                artifact_regions.append({
                    'start_time': onset,
                    'end_time': onset + duration,
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'duration': duration
                })

    return valid_mask, artifact_regions

def analyze_spo2_channel_fast(raw):
    """Быстрый анализ SpO2 с исключением артефактов"""
    spo2_stats = {
        'avg_spo2': None, 'min_spo2': None,
        'time_below_spo2_90': 0, 'time_below_spo2_85': 0,
        'spo2_baseline': None,
        'artifact_regions_excluded': 0
    }

    try:
        # Получаем маску артефактов
        artifact_mask, artifact_regions = get_artifact_masks(raw)

        spo2_channels = [ch for ch in raw.ch_names if any(x in ch.lower() for x in ['spo2', 'sao2', 'sat'])]
        if spo2_channels:
            spo2_idx = raw.ch_names.index(spo2_channels[0])
            data, times = raw[spo2_idx, :]
            if len(data) > 0:
                spo2_values = data[0]

                # Применяем маску артефактов
                if artifact_mask is not None:
                    valid_spo2 = spo2_values[(spo2_values >= 50) & (spo2_values <= 100) & artifact_mask]
                    spo2_stats['artifact_regions_excluded'] = len(artifact_regions)
                else:
                    valid_spo2 = spo2_values[(spo2_values >= 50) & (spo2_values <= 100)]

                if len(valid_spo2) > 0:
                    spo2_stats['avg_spo2'] = round(float(np.median(valid_spo2)), 1)
                    spo2_stats['min_spo2'] = round(float(np.percentile(valid_spo2, 1)), 1)
                    spo2_stats['spo2_baseline'] = round(float(np.percentile(valid_spo2, 90)), 1)

                    # Расчет времени ниже порогов (только на валидных данных)
                    if artifact_mask is not None:
                        total_valid_samples = np.sum(artifact_mask)
                        if total_valid_samples > 0:
                            samples_below_90 = np.sum(
                                (spo2_values < 90) & artifact_mask & (spo2_values >= 50) & (spo2_values <= 100))
                            samples_below_85 = np.sum(
                                (spo2_values < 85) & artifact_mask & (spo2_values >= 50) & (spo2_values <= 100))

                            total_duration_seconds = raw.times[-1]
                            valid_duration_ratio = total_valid_samples / len(raw.times)

                            time_below_90 = (samples_below_90 / total_valid_samples) * (
                                        total_duration_seconds / 60) * valid_duration_ratio
                            time_below_85 = (samples_below_85 / total_valid_samples) * (
                                        total_duration_seconds / 60) * valid_duration_ratio

                            spo2_stats['time_below_spo2_90'] = int(time_below_90)
                            spo2_stats['time_below_spo2_85'] = int(time_below_85)

    except Exception as e:
        print(f"  ⚠️ Ошибка анализа SpO2: {e}")

    return spo2_stats

def load_edf_file(edf_path):
    """Загрузка EDF файла"""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        return raw
    except Exception as e:
        print(f"❌ Ошибка загрузки файла: {e}")
        return None

def extract_all_annotations(edf_path):
    """Извлечение всех аннотаций"""
    raw = load_edf_file(edf_path)
    if not raw or not hasattr(raw, 'annotations') or raw.annotations is None or len(raw.annotations) == 0:
        return None, None

    annotations = raw.annotations
    annotation_counts = Counter(str(desc) for desc in annotations.description)

    return raw, annotation_counts

def extract_patient_info_from_edf(edf_path):
    """Извлечение UUID из EDF файла (упрощенная версия)"""
    try:
        with open(edf_path, 'rb') as f:
            header = f.read(256).decode('latin-1', errors='ignore')
            patient_info = header[8:168].strip()

            # Поиск UUID
            uuid_pattern = r'([a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})'
            uuid_match = re.search(uuid_pattern, patient_info)

            if uuid_match:
                return {'uuid': uuid_match.group(1)}

            return {'uuid': None}

    except Exception as e:
        print(f"❌ Ошибка чтения EDF файла: {e}")
        return {'uuid': None}

def format_time(seconds):
    """Форматирование времени в ЧЧ:ММ:СС"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def analyze_signal_quality(raw, channel_groups):
    """Анализ качества сигналов по каналам"""
    try:
        # Анализ ЭКГ каналов
        if channel_groups['ЭКГ (ECG)']:
            ecg_channel = channel_groups['ЭКГ (ECG)'][0][0]
            ecg_idx = raw.ch_names.index(ecg_channel)
            data, times = raw[ecg_idx, :5000]  # Берем больше samples
            ecg_signal = data[0]

            # Нормализуем сигнал
            ecg_normalized = ecg_signal - np.mean(ecg_signal)
            signal_range = np.ptp(ecg_normalized)  # Peak-to-peak amplitude
            noise_level = np.std(ecg_normalized)

            if signal_range > 0:
                snr = signal_range / noise_level
                quality = "отличный" if snr > 20 else "хороший" if snr > 10 else "удовлетворительный"
                print(f"  • ЭКГ канал '{ecg_channel}': {quality} сигнал")
                print(f"    - Диапазон сигнала: {signal_range:.2f}")
                print(f"    - Уровень шума: {noise_level:.2f}")
                print(f"    - Отношение сигнал/шум: {snr:.1f}")
            else:
                print(f"  • ЭКГ канал '{ecg_channel}': плохой сигнал (нулевой диапазон)")

        # Анализ SpO2 каналов
        if channel_groups['Сатурация (SpO2)']:
            spo2_channel = channel_groups['Сатурация (SpO2)'][0][0]
            spo2_idx = raw.ch_names.index(spo2_channel)
            data, times = raw[spo2_idx, :10000]  # Берем больше samples
            spo2_signal = data[0]

            valid_spo2 = spo2_signal[(spo2_signal >= 50) & (spo2_signal <= 100)]
            if len(valid_spo2) > 0:
                coverage = (len(valid_spo2) / len(spo2_signal)) * 100
                quality = "отличный" if coverage > 95 else "хороший" if coverage > 80 else "удовлетворительный"
                print(f"  • SpO2 канал '{spo2_channel}': {quality} сигнал")
                print(f"    - Покрытие валидными данными: {coverage:.1f}%")
                print(f"    - Диапазон значений: {np.min(valid_spo2):.1f}-{np.max(valid_spo2):.1f}%")
            else:
                print(f"  • SpO2 канал '{spo2_channel}': нет валидных данных")

        # Анализ дыхательных каналов
        if channel_groups['Дыхание (Respiratory)']:
            for resp_channel, _ in channel_groups['Дыхание (Respiratory)'][:2]:
                resp_idx = raw.ch_names.index(resp_channel)
                data, times = raw[resp_idx, :5000]
                resp_signal = data[0]

                # Нормализуем сигнал
                resp_normalized = resp_signal - np.mean(resp_signal)
                signal_variance = np.var(resp_normalized)

                quality = "отличный" if signal_variance > 100 else "хороший" if signal_variance > 10 else "удовлетворительный"
                print(f"  • Дыхательный канал '{resp_channel}': {quality} сигнал")
                print(f"    - Дисперсия сигнала: {signal_variance:.2f}")

        # Анализ пульса
        if channel_groups['Пульс']:
            pulse_channel = channel_groups['Пульс'][0][0]
            pulse_idx = raw.ch_names.index(pulse_channel)
            data, times = raw[pulse_idx, :5000]
            pulse_signal = data[0]

            valid_pulse = pulse_signal[(pulse_signal >= 40) & (pulse_signal <= 180)]
            if len(valid_pulse) > 0:
                coverage = (len(valid_pulse) / len(pulse_signal)) * 100
                quality = "отличный" if coverage > 90 else "хороший" if coverage > 70 else "удовлетворительный"
                print(f"  • Пульс канал '{pulse_channel}': {quality} сигнал")
                print(f"    - Покрытие валидными данными: {coverage:.1f}%")
                print(f"    - Диапазон значений: {np.min(valid_pulse):.1f}-{np.max(valid_pulse):.1f} уд/мин")

    except Exception as e:
        print(f"  ⚠️ Ошибка анализа качества сигналов: {e}")

def get_detailed_channel_info(raw, channel_name):
    """Получение детальной информации о конкретном канале"""
    try:
        if channel_name not in raw.ch_names:
            print(f"❌ Канал '{channel_name}' не найден")
            return None

        idx = raw.ch_names.index(channel_name)
        channel_type = raw.get_channel_types()[idx]
        sfreq = raw.info['sfreq']

        # Получаем данные канала
        data, times = raw[idx, :]
        signal = data[0]

        info = {
            'name': channel_name,
            'type': channel_type,
            'sampling_rate': sfreq,
            'samples_count': len(signal),
            'duration_seconds': len(signal) / sfreq,
            'duration_minutes': len(signal) / sfreq / 60,
            'min_value': np.min(signal),
            'max_value': np.max(signal),
            'mean_value': np.mean(signal),
            'std_value': np.std(signal),
            'unit': raw._orig_units.get(channel_name, 'unknown')
        }

        return info

    except Exception as e:
        print(f"❌ Ошибка получения информации о канале: {e}")
        return None

def analyze_respiratory_channels_improved(raw):
    """Улучшенный анализ дыхательных каналов"""
    resp_stats = {
        'avg_resp_rate': None,
        'min_resp_rate': None,
        'max_resp_rate': None,
        'signal_quality': 'unknown'
    }

    try:
        # Поиск дыхательных каналов с приоритетом
        resp_patterns = ['resp', 'breath', 'дыхание', 'thorax', 'chest', 'abdomen', 'flow']
        resp_channels = [
            ch for ch in raw.ch_names
            if any(pattern in ch.lower() for pattern in resp_patterns)
        ]

        if not resp_channels:
            resp_stats['signal_quality'] = 'no_channel'
            return resp_stats

        # Пробуем разные каналы по порядку
        best_rates = []
        for resp_ch in resp_channels[:2]:  # Анализируем первые 2 канала
            rates = analyze_single_resp_channel(raw, resp_ch)
            if rates:
                best_rates.extend(rates)

        if not best_rates:
            resp_stats['signal_quality'] = 'no_signal'
            return resp_stats

        # Фильтрация реалистичных значений
        valid_rates = [r for r in best_rates if 8 <= r <= 25]  # Более строгие пределы

        if len(valid_rates) < 5:
            # Если мало валидных значений, используем менее строгие пределы
            valid_rates = [r for r in best_rates if 6 <= r <= 30]

        if not valid_rates:
            resp_stats['signal_quality'] = 'invalid_rates'
            return resp_stats

        # Статистика с отсечением выбросов
        valid_rates = np.array(valid_rates)
        q1, q3 = np.percentile(valid_rates, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        final_rates = valid_rates[(valid_rates >= lower_bound) & (valid_rates <= upper_bound)]

        if len(final_rates) < 3:
            final_rates = valid_rates  # Используем все, если после фильтрации мало данных

        resp_stats['avg_resp_rate'] = round(float(np.median(final_rates)), 1)
        resp_stats['min_resp_rate'] = round(float(np.percentile(final_rates, 10)), 1)
        resp_stats['max_resp_rate'] = round(float(np.percentile(final_rates, 90)), 1)
        resp_stats['signal_quality'] = 'good' if len(final_rates) >= 10 else 'moderate'

    except Exception as e:
        print(f"  ⚠️ Ошибка улучшенного анализа дыхания: {e}")
        resp_stats['signal_quality'] = 'error'

    return resp_stats

def analyze_single_resp_channel(raw, channel_name):
    """Анализ одного дыхательного канала с исключением артефактов"""
    try:
        # Получаем маску артефактов
        artifact_mask, artifact_regions = get_artifact_masks(raw)

        channel_idx = raw.ch_names.index(channel_name)
        sfreq = raw.info['sfreq']

        # Берем больше данных для анализа (первые 30 минут)
        max_samples = min(int(sfreq * 1800), len(raw.times))
        data, times = raw[channel_idx, :max_samples]

        if len(data) == 0:
            return []

        resp_signal = data[0]

        # Если есть артефакты, обрабатываем только валидные сегменты
        if artifact_mask is not None and len(artifact_mask) >= max_samples:
            segment_mask = artifact_mask[:max_samples]
            valid_segments = find_continuous_segments(segment_mask, min_segment_length=int(sfreq * 30))  # мин 30 секунд

            breathing_rates = []
            for start, end in valid_segments:
                segment_signal = resp_signal[start:end]
                if len(segment_signal) > int(sfreq * 30):  # сегменты не менее 30 секунд
                    # Предобработка сегмента
                    resp_clean = preprocess_resp_signal(segment_signal, sfreq)
                    if resp_clean is not None:
                        # Анализ сегмента
                        rate_peaks = analyze_breathing_peaks_improved(resp_clean, sfreq)
                        if rate_peaks:
                            breathing_rates.extend(rate_peaks)

            return breathing_rates
        else:
            # Старая логика, если нет артефактов
            resp_clean = preprocess_resp_signal(resp_signal, sfreq)
            if resp_clean is None:
                return []

            breathing_rates = []

            # Метод 1: Анализ по пикам (основной)
            rate_peaks = analyze_breathing_peaks_improved(resp_clean, sfreq)
            if rate_peaks:
                breathing_rates.extend(rate_peaks)

            # Метод 2: Спектральный анализ
            rate_spectral = analyze_breathing_spectral_improved(resp_clean, sfreq)
            if rate_spectral:
                breathing_rates.append(rate_spectral)

            # Метод 3: Сегментный анализ
            rate_segmented = analyze_breathing_segmented_improved(resp_clean, sfreq)
            if rate_segmented:
                breathing_rates.extend(rate_segmented)

            return breathing_rates

    except Exception as e:
        print(f"    ⚠️ Ошибка анализа канала {channel_name}: {e}")
        return []

def preprocess_resp_signal(signal_data, sfreq):
    """Предобработка дыхательного сигнала с исправленными импортами"""
    try:
        # Удаление выбросов
        signal_clean = np.copy(signal_data)
        median = np.median(signal_clean)
        mad = np.median(np.abs(signal_clean - median))

        # Отсечение выбросов (5 MAD)
        outlier_mask = np.abs(signal_clean - median) > 5 * mad
        signal_clean[outlier_mask] = median

        # Дыхательный фильтр 0.1-1.0 Гц (6-60 дых/мин)
        low_freq = 0.1  # 6 дых/мин
        high_freq = 1.0  # 60 дых/мин

        # Проверка частоты дискретизации
        if sfreq <= 2 * high_freq:
            return None

        # ИСПРАВЛЕНИЕ: используем scipy.signal вместо signal
        b, a = signal.butter(3, [low_freq / (sfreq / 2), high_freq / (sfreq / 2)], btype='band')
        resp_filtered = signal.filtfilt(b, a, signal_clean)

        # Нормализация
        resp_normalized = (resp_filtered - np.mean(resp_filtered)) / (np.std(resp_filtered) + 1e-8)

        return resp_normalized

    except Exception as e:
        print(f"      ⚠️ Ошибка предобработки: {e}")
        return None

def analyze_breathing_peaks_improved(resp_signal, sfreq):
    """Улучшенный анализ дыхания через пики"""
    try:
        # Поиск пиков вдохов
        peaks, properties = signal.find_peaks(
            resp_signal,
            distance=int(0.8 * sfreq),  # Минимум 0.8 сек между вдохами
            prominence=0.3,
            height=0.2,
            width=int(0.3 * sfreq)  # Минимальная длительность пика
        )

        if len(peaks) < 4:
            return []

        # Расчет интервалов между вдохами
        breath_intervals = np.diff(peaks) / sfreq

        # Фильтрация реалистичных интервалов (1.0-7.5 сек = 8-60 дых/мин)
        valid_intervals = breath_intervals[
            (breath_intervals >= 1.0) & (breath_intervals <= 7.5)
            ]

        if len(valid_intervals) < 3:
            return []

        # Расчет частоты дыхания
        breathing_rates = 60.0 / valid_intervals

        # Дополнительная фильтрация частот
        valid_rates = breathing_rates[
            (breathing_rates >= 8) & (breathing_rates <= 25)  # Более строгие пределы
            ]

        return valid_rates.tolist()

    except Exception as e:
        return []

def analyze_breathing_spectral_improved(resp_signal, sfreq):
    """Улучшенный спектральный анализ с исправленными импортами"""
    try:
        # Параметры Уэлча
        nperseg = min(1024, len(resp_signal))
        if nperseg < 256:
            return None

        # ИСПРАВЛЕНИЕ: используем scipy.signal вместо signal
        f, Pxx = signal.welch(resp_signal, fs=sfreq, nperseg=nperseg)

        # Дыхательный диапазон 0.1-0.7 Гц (6-42 дых/мин)
        breath_mask = (f >= 0.1) & (f <= 0.7)
        breath_freq = f[breath_mask]
        breath_power = Pxx[breath_mask]

        if len(breath_power) == 0:
            return None

        # Находим пик в дыхательном диапазоне
        peak_idx = np.argmax(breath_power)
        dominant_freq = breath_freq[peak_idx]
        breathing_rate = dominant_freq * 60

        # Проверка амплитуды пика
        if breath_power[peak_idx] < np.max(Pxx) * 0.1:
            return None

        if 8 <= breathing_rate <= 25:
            return breathing_rate

    except Exception as e:
        print(f"      ⚠️ Ошибка спектрального анализа: {e}")
        return None

def analyze_breathing_segmented_improved(resp_signal, sfreq):
    """Улучшенный сегментный анализ"""
    try:
        segment_duration = 60 * sfreq  # 60-секундные сегменты
        breathing_rates = []

        for i in range(0, len(resp_signal) - int(segment_duration), int(segment_duration // 2)):
            segment = resp_signal[i:i + int(segment_duration)]

            # Анализ пиков в сегменте
            rate_peaks = analyze_breathing_peaks_improved(segment, sfreq)
            if rate_peaks:
                breathing_rates.append(np.median(rate_peaks))

            # Спектральный анализ сегмента
            rate_spectral = analyze_breathing_spectral_improved(segment, sfreq)
            if rate_spectral:
                breathing_rates.append(rate_spectral)

        return breathing_rates

    except Exception as e:
        return []

def analyze_breathing_segmented(resp_signal, sfreq):
    """Анализ дыхания по сегментам"""
    try:
        segment_duration = 30 * sfreq
        breathing_rates = []

        for i in range(0, len(resp_signal) - int(segment_duration), int(segment_duration)):
            segment = resp_signal[i:i + int(segment_duration)]
            rate = analyze_breathing_peaks(segment, sfreq)
            if rate and 8 <= rate <= 40:
                breathing_rates.append(rate)

        return breathing_rates
    except:
        return []

def analyze_breathing_peaks(resp_signal, sfreq):
    """Анализ дыхания через поиск пиков"""
    try:
        low_freq = 0.1
        high_freq = 0.7

        b, a = signal.butter(3, [low_freq / (sfreq / 2), high_freq / (sfreq / 2)], btype='band')
        resp_filtered = signal.filtfilt(b, a, resp_signal)
        resp_normalized = (resp_filtered - np.mean(resp_filtered)) / (np.std(resp_filtered) + 1e-8)

        peaks, _ = signal.find_peaks(np.abs(resp_normalized),
                                     distance=int(0.8 * sfreq),
                                     prominence=0.3,
                                     height=0.2)

        if len(peaks) >= 4:
            breath_intervals = np.diff(peaks) / sfreq
            valid_intervals = breath_intervals[(breath_intervals >= 1.0) & (breath_intervals <= 7.5)]

            if len(valid_intervals) >= 3:
                breathing_rates = 60.0 / valid_intervals
                valid_rates = breathing_rates[(breathing_rates >= 8) & (breathing_rates <= 40)]

                if len(valid_rates) >= 2:
                    return np.median(valid_rates)
    except:
        pass
    return None

def analyze_breathing_spectral(resp_signal, sfreq):
    """Спектральный анализ дыхания"""
    try:
        low_freq = 0.1
        high_freq = 0.7

        b, a = signal.butter(3, [low_freq / (sfreq / 2), high_freq / (sfreq / 2)], btype='band')
        resp_filtered = signal.filtfilt(b, a, resp_signal)

        f, Pxx = signal.welch(resp_filtered, fs=sfreq, nperseg=min(1024, len(resp_filtered)))

        breath_mask = (f >= low_freq) & (f <= high_freq)
        breath_freq = f[breath_mask]
        breath_power = Pxx[breath_mask]

        if len(breath_power) > 0:
            peak_indices = signal.find_peaks(breath_power, height=np.max(breath_power) * 0.1)[0]

            if len(peak_indices) > 0:
                main_peak_idx = peak_indices[0]
                dominant_freq = breath_freq[main_peak_idx]
                breathing_rate = dominant_freq * 60

                if 8 <= breathing_rate <= 40:
                    return breathing_rate
    except:
        pass
    return None

def calculate_respiratory_events(raw):
    """Анализ дыхательных нарушений"""
    annotations = raw.annotations

    respiratory_events = {
        'apneas': 0,
        'hypopneas': 0,
        'desaturations': 0,
        'snoring': 0
    }

    event_mapping = {
        'Обструктивное апноэ(pointPolySomnographyObstructiveApnea)': 'apneas',
        'Обструктивное гипопноэ(pointPolySomnographyHypopnea)': 'hypopneas',
        'Десатурация(pointPolySomnographyDesaturation)': 'desaturations',
        'Храп(pointPolySomnographySnore)': 'snoring'
    }

    for desc in annotations.description:
        desc_str = str(desc)
        if desc_str in event_mapping:
            respiratory_events[event_mapping[desc_str]] += 1

    return respiratory_events

if __name__ == "__main__":
    edf_file_path = "EDF/test1.edf"

    # Создание отчета
    # print_sleep_report(edf_file_path)

    # Генерация SQL файла
    print("\n" + "=" * 60)
    print("🗃️  ГЕНЕРАЦИЯ SQL ДЛЯ БАЗЫ ДАННЫХ")
    print("=" * 60)

    sql_file = generate_sql_insert(edf_file_path)
    # if sql_file:
    #     print(f"📁 Файл для импорта: {sql_file}")