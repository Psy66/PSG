import mne
import pandas as pd
from tabulate import tabulate
from collections import defaultdict

def get_channel_display_name(ch_name):
	"""Преобразует техническое название канала в понятное"""
	ch_upper = ch_name.upper()

	if 'EEG F3-A2' in ch_name:
		return 'EEG Лобный левый'
	elif 'EEG C3-A2' in ch_name:
		return 'EEG Центральный левый'
	elif 'EEG O1-A2' in ch_name:
		return 'EEG Затылочный левый'
	elif 'EEG F4-A2' in ch_name:
		return 'EEG Лобный правый'
	elif 'EEG C4-A2' in ch_name:
		return 'EEG Центральный правый'
	elif 'EEG O2-A2' in ch_name:
		return 'EEG Затылочный правый'
	elif 'EMG F7-F8' in ch_upper:
		return 'EMG Подбородочные мышцы'
	elif 'EOGL' in ch_upper or 'FP1-A2' in ch_upper:
		return 'EOG Левый глаз'
	elif 'EOGR' in ch_upper or 'FP2-A2' in ch_upper:
		return 'EOG Правый глаз'
	elif 'LMl' in ch_upper or 'T3-T5' in ch_upper:
		return 'EMG Левая нога'
	elif 'LMr' in ch_upper or 'T4-T6' in ch_upper:
		return 'EMG Правая нога'
	elif 'SOUND' in ch_upper:
		return 'Микрофон (храп)'
	elif 'ECG' in ch_upper:
		return 'ЭКГ'
	elif 'CHASTOTA' in ch_upper or 'PULSA' in ch_upper:
		return 'Пульс'
	elif 'CHEST' in ch_upper:
		return 'Дыхание грудное'
	elif 'ABDOMEN' in ch_upper:
		return 'Дыхание брюшное'
	elif 'POSITION' in ch_upper:
		return 'Положение тела'
	elif 'BREATH' in ch_upper:
		return 'Дыхательный поток'
	elif 'SAO2' in ch_upper or 'SPO2' in ch_upper:
		return 'Кислород крови'
	elif 'PPG' in ch_upper:
		return 'Пульсовая волна'
	else:
		return ch_name

def analyze_sleep_stages(annotations):
	"""Анализ длительности стадий сна"""
	if not annotations:
		return None

	stage_names = {
		'Sleep stage W': 'Бодрствование',
		'Sleep stage 1': 'N1',
		'Sleep stage 2': 'N2',
		'Sleep stage 3': 'N3',
		'Sleep stage 4': 'N3',
		'Sleep stage R': 'REM',
		'БДГ': 'REM',
		'Sleep stage W(eventUnknown)': 'Бодрствование',
		'Sleep stage 1(eventUnknown)': 'N1',
		'Sleep stage 2(eventUnknown)': 'N2',
		'Sleep stage 3(eventUnknown)': 'N3',
		'Sleep stage 4(eventUnknown)': 'N3',
		'Sleep stage R(eventUnknown)': 'REM',
		'БДГ(pointPolySomnographyREM)': 'REM'
	}

	# Собираем стадии сна
	sleep_events = []
	for ann in annotations:
		if ann['description'] in stage_names:
			sleep_events.append({
				'stage': stage_names[ann['description']],
				'duration': ann['duration']
			})

	if not sleep_events:
		return None

	# Считаем общую длительность по стадиям
	stage_durations = defaultdict(float)
	for event in sleep_events:
		stage_durations[event['stage']] += event['duration']

	total_sleep_time = sum(stage_durations.values())

	return {
		'durations': dict(stage_durations),
		'total_sleep_time': total_sleep_time
	}

def read_edf_optimized(filename):
	"""Оптимизированное чтение EDF файла"""
	try:
		raw = mne.io.read_raw_edf(filename, preload=False, verbose=False)

		print("ОСНОВНАЯ ИНФОРМАЦИЯ")
		print(f"Длительность: {raw.times[-1] / 60:.1f} мин | Каналы: {len(raw.ch_names)}")
		print(f"Дата: {raw.info['meas_date'].strftime('%d.%m.%Y %H:%M')}")

		# Таблица каналов с понятными названиями
		channels_data = []
		for i, ch_name in enumerate(raw.ch_names):
			display_name = get_channel_display_name(ch_name)
			channels_data.append([i + 1, display_name])

		print("\nКАНАЛЫ:")
		print(tabulate(channels_data, headers=['№', 'Канал'], tablefmt='simple'))

		# Анализ сна
		if hasattr(raw, 'annotations') and raw.annotations:
			sleep_analysis = analyze_sleep_stages(raw.annotations)
			if sleep_analysis:
				print(f"\nСТАТИСТИКА СНА:")
				print(f"Общее время сна: {sleep_analysis['total_sleep_time'] / 60:.1f} мин")

				print("\nДлительность стадий:")
				for stage in ['Бодрствование', 'N1', 'N2', 'N3', 'REM']:
					if stage in sleep_analysis['durations']:
						duration_min = sleep_analysis['durations'][stage] / 60
						percent = sleep_analysis['durations'][stage] / sleep_analysis['total_sleep_time'] * 100
						print(f"  {stage}: {duration_min:.1f} мин ({percent:.1f}%)")

		return raw

	except Exception as e:
		print(f"Ошибка: {e}")
		return None

# Запуск
if __name__ == "__main__":
	filename = "test.edf"
	raw_data = read_edf_optimized(filename)