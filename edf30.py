import os
import re
from datetime import datetime
from collections import Counter
import json
import mne
import numpy as np
from scipy import signal

# Конфигурация анализа
ANALYSIS_CONFIG = {
	# Настройки ЭКГ анализа
	'ecg': {
		'tachycardia_threshold': 100,  # порог тахикардии (уд/мин)
		'bradycardia_threshold': 50,  # порог брадикардии (уд/мин)
		'min_consecutive_events': 10,  # минимальное последовательных событий для эпизода
		'rr_min': 0.3,  # минимальный RR интервал (сек)
		'rr_max': 2.0,  # максимальный RR интервал (сек)
		'hr_min': 40,  # минимальная ЧСС (уд/мин)
		'hr_max': 150,  # максимальная ЧСС (уд/мин)
	},

	# Настройки дыхательного анализа
	'respiration': {
		'min_rate': 8,  # минимальная частота дыхания (дых/мин)
		'max_rate': 25,  # максимальная частота дыхания (дых/мин)
		'filter_low': 0.1,  # нижняя граница фильтра (Гц)
		'filter_high': 1.0,  # верхняя граница фильтра (Гц)
		'min_segment_length': 30,  # минимальная длина сегмента (сек)
	},

	# Настройки SpO2 анализа
	'spo2': {
		'min_valid': 75,  # минимальное валидное SpO2 (%)
		'max_valid': 100,  # максимальное валидное SpO2 (%)
		'threshold_90': 90,  # порог для времени <90% сатурации
		'threshold_85': 85,  # порог для времени <85% сатурации
	},

	# Настройки качества сна
	'sleep_quality': {
		'efficiency_weights': {85: 25, 70: 20, 50: 10},
		'n3_threshold': 15,  # порог для глубокого сна (%)
		'rem_threshold': 20,  # порог для REM сна (%)
		'ahi_weights': {5: 30, 15: 20, 30: 10},
		'arousal_weights': {10: 15, 20: 10},
	}
}

class SleepAnalyzer:
	"""Основной класс для анализа сна"""

	def __init__(self, config=None):
		self.config = config or ANALYSIS_CONFIG
		self.raw = None
		self.stages = None

	def load_edf_file(self, edf_path):
		"""Загрузка EDF файла"""
		try:
			self.raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
			return self.raw
		except Exception as e:
			print(f"❌ Ошибка загрузки файла: {e}")
			return None

	def extract_patient_info_from_edf(self, edf_path):
		"""Извлечение UUID из EDF файла"""
		try:
			with open(edf_path, 'rb') as f:
				header = f.read(256).decode('latin-1', errors='ignore')
				patient_info = header[8:168].strip()

				uuid_pattern = r'([a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})'
				uuid_match = re.search(uuid_pattern, patient_info)

				if uuid_match:
					return {'uuid': uuid_match.group(1)}

				return {'uuid': None}

		except Exception as e:
			print(f"❌ Ошибка чтения EDF файла: {e}")
			return {'uuid': None}

	def calculate_sleep_stages(self):
		"""Расчет стадий сна по 30-секундным эпохам"""
		if not self.raw or not hasattr(self.raw, 'annotations'):
			return None

		annotations = self.raw.annotations

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

		self.stages = stages
		return stages

	def calculate_rem_quality(self):
		"""Расчет качества REM-сна"""
		if not self.raw or not hasattr(self.raw, 'annotations'):
			return None

		annotations = self.raw.annotations

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

	def calculate_sleep_efficiency(self):
		"""Расчет эффективности сна"""
		if not self.stages:
			return None

		total_sleep_time = sum(self.stages[stage]['minutes'] for stage in ['N1', 'N2', 'N3', 'REM'])
		total_bed_time = sum(stage['minutes'] for stage in self.stages.values())

		sleep_efficiency = (total_sleep_time / total_bed_time * 100) if total_bed_time > 0 else 0

		return {
			'sleep_efficiency': sleep_efficiency,
			'total_sleep_time': total_sleep_time,
			'total_bed_time': total_bed_time,
			'wake_after_sleep_onset': self.stages['Wake']['minutes']
		}

	def calculate_sleep_latencies(self):
		"""Расчет различных латентностей"""
		annotations = self.raw.annotations

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

	def calculate_sleep_architecture(self):
		"""Анализ архитектуры сна"""
		if not self.stages:
			return None

		total_sleep_time = sum(self.stages[stage]['minutes'] for stage in ['N1', 'N2', 'N3', 'REM'])

		if total_sleep_time == 0:
			return None

		architecture = {
			'n1_percentage': (self.stages['N1']['minutes'] / total_sleep_time) * 100,
			'n2_percentage': (self.stages['N2']['minutes'] / total_sleep_time) * 100,
			'n3_percentage': (self.stages['N3']['minutes'] / total_sleep_time) * 100,
			'rem_percentage': (self.stages['REM']['minutes'] / total_sleep_time) * 100,
			'deep_sleep_ratio': self.stages['N3']['minutes'] / total_sleep_time,
			'rem_nrem_ratio': self.stages['REM']['minutes'] / (
					self.stages['N1']['minutes'] + self.stages['N2']['minutes'] + self.stages['N3']['minutes'])
		}

		return architecture

	def calculate_sleep_fragmentation(self):
		"""Анализ фрагментации сна с разделением типов движений"""
		annotations = self.raw.annotations

		activations = sum(1 for desc in annotations.description
		                  if str(desc) == 'Активация(pointPolySomnographyActivation)')

		limb_movements = sum(1 for desc in annotations.description
		                     if str(desc) == 'Движение конечностей(pointPolySomnographyLegsMovements)')

		periodic_limb_movements = sum(1 for desc in annotations.description
		                              if
		                              str(desc) == 'Периодические движения конечностей(pointPolySomnographyPeriodicalLegsMovements)')

		# ДОБАВЛЕНО: Подсчет бруксизмов
		bruxism_events = sum(1 for desc in annotations.description
		                     if str(desc) == 'Бруксизм(pointBruxism)')

		total_sleep_time = sum(self.stages[stage]['minutes'] for stage in ['N1', 'N2', 'N3', 'REM'])

		total_movements = limb_movements + periodic_limb_movements
		fragmentation_index = (activations + total_movements) / (total_sleep_time / 60) if total_sleep_time > 0 else 0

		return {
			'fragmentation_index': fragmentation_index,
			'activations': activations,
			'limb_movements': limb_movements,
			'periodic_limb_movements': periodic_limb_movements,
			'bruxism_events': bruxism_events,  # ДОБАВЛЕНО
			'total_limb_movements': total_movements,
			'arousal_index': activations / (total_sleep_time / 60) if total_sleep_time > 0 else 0
		}

	def calculate_sleep_indices(self):
		"""Расчет различных индексов"""
		if not self.raw or not self.stages:
			return {}

		respiratory_events = self.calculate_respiratory_events() or {}
		total_sleep_time = sum(self.stages[stage]['minutes'] for stage in ['N1', 'N2', 'N3', 'REM'])

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

	def calculate_respiratory_events(self):
		"""Анализ дыхательных нарушений"""
		annotations = self.raw.annotations

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

	def calculate_rem_cycles(self):
		"""Расчет количества REM-циклов за ночь"""
		if not self.raw or not hasattr(self.raw, 'annotations'):
			return 0

		annotations = self.raw.annotations

		stage_mapping = {
			'Sleep stage W(eventUnknown)': 'W',
			'Sleep stage 1(eventUnknown)': 'N1',
			'Sleep stage 2(eventUnknown)': 'N2',
			'Sleep stage 3(eventUnknown)': 'N3',
			'Sleep stage R(eventUnknown)': 'R'
		}

		stages_sequence = []
		current_time = 0

		for desc, duration in zip(annotations.description, annotations.duration):
			desc_str = str(desc)
			if desc_str in stage_mapping and abs(duration - 30) < 1:
				stages_sequence.append(stage_mapping[desc_str])

		if not stages_sequence:
			return 0

		rem_cycles = 0
		in_rem_cycle = False
		rem_cycle_started = False

		for i in range(1, len(stages_sequence) - 1):
			current_stage = stages_sequence[i]
			prev_stage = stages_sequence[i - 1]
			next_stage = stages_sequence[i + 1]

			if current_stage == 'R' and prev_stage in ['N1', 'N2', 'N3'] and not rem_cycle_started:
				rem_cycle_started = True
				in_rem_cycle = True

			elif current_stage == 'R' and next_stage in ['N1', 'N2', 'N3'] and in_rem_cycle:
				rem_cycles += 1
				in_rem_cycle = False
				rem_cycle_started = False

			elif current_stage == 'W' and in_rem_cycle:
				in_rem_cycle = False
				rem_cycle_started = False

		return rem_cycles

	def calculate_overall_sleep_quality(self):
		"""Комплексная оценка качества сна"""
		if not self.raw or not self.stages:
			return {}

		efficiency_data = self.calculate_sleep_efficiency() or {}
		architecture_data = self.calculate_sleep_architecture() or {}
		sleep_indices = self.calculate_sleep_indices() or {}
		fragmentation_data = self.calculate_sleep_fragmentation() or {}
		rem_quality = self.calculate_rem_quality() or {}
		hr_stats = self.analyze_heart_rate_comprehensive() or {}
		rem_cycles = self.calculate_rem_cycles()

		score = 0
		cfg = self.config['sleep_quality']

		# Эффективность сна
		efficiency = efficiency_data.get('sleep_efficiency', 0)
		for threshold, points in cfg['efficiency_weights'].items():
			if efficiency >= threshold:
				score += points
				break

		# Архитектура сна
		n3_percentage = architecture_data.get('n3_percentage', 0)
		rem_percentage = architecture_data.get('rem_percentage', 0)

		if n3_percentage >= cfg['n3_threshold']:
			score += 15
		if rem_percentage >= cfg['rem_threshold']:
			score += 15

		# Дыхательные нарушения
		ahi = sleep_indices.get('ahi', 0)
		for threshold, points in cfg['ahi_weights'].items():
			if ahi < threshold:
				score += points
				break

		# Фрагментация
		arousal_index = fragmentation_data.get('arousal_index', 0)
		for threshold, points in cfg['arousal_weights'].items():
			if arousal_index < threshold:
				score += points
				break

		# REM качество
		rem_score = rem_quality.get('rem_quality_score', 0)
		score += rem_score * 0.15

		# Штраф за тахикардию
		tachycardia_events = hr_stats.get('tachycardia_events', 0)
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

	def get_artifact_masks(self, artifact_marker='Артефакт(blockArtefact)'):
		"""Создание масок для исключения участков с артефактами"""
		if not self.raw or not hasattr(self.raw, 'annotations'):
			return None, None

		annotations = self.raw.annotations
		sfreq = self.raw.info['sfreq']
		total_samples = len(self.raw.times)

		valid_mask = np.ones(total_samples, dtype=bool)
		current_time = 0
		artifact_regions = []

		for desc, duration, onset in zip(annotations.description, annotations.duration, annotations.onset):
			desc_str = str(desc)

			if artifact_marker in desc_str:
				start_sample = int(onset * sfreq)
				end_sample = int((onset + duration) * sfreq)
				end_sample = min(end_sample, total_samples - 1)

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

	def analyze_heart_rate_comprehensive(self):
		"""Комплексный анализ сердечного ритма из ЭКГ"""
		results = {
			'avg_heart_rate': None,
			'min_heart_rate': None,
			'max_heart_rate': None,
			'heart_rate_variability': None,
			'artifact_regions_excluded': 0,
			'tachycardia_events': 0,
			'bradycardia_events': 0,
			'analysis_method': 'ecg'
		}

		try:
			ecg_keywords = ['ecg', 'ekg', 'electrocardiogram', 'экг', 'кардиограмма']
			ecg_channels = [
				ch for ch in self.raw.ch_names
				if any(keyword in ch.lower() for keyword in ecg_keywords)
			]

			if not ecg_channels:
				return self._analyze_heart_rate_from_markers()

			ecg_ch = ecg_channels[0]
			ecg_idx = self.raw.ch_names.index(ecg_ch)
			sfreq = self.raw.info['sfreq']
			max_samples = len(self.raw.times)

			data, times = self.raw[ecg_idx, :max_samples]
			if len(data) == 0:
				return self._analyze_heart_rate_from_markers()

			ecg_signal = data[0]

			artifact_mask, artifact_regions = self.get_artifact_masks()
			r_peaks = self._get_clean_r_peaks(ecg_signal, sfreq, artifact_mask, artifact_regions, max_samples)
			results['artifact_regions_excluded'] = len(artifact_regions)

			if len(r_peaks) <= 100:
				return self._analyze_heart_rate_from_markers()

			rr_intervals, heart_rates = self._calculate_heart_rate_metrics(r_peaks, sfreq)

			if len(heart_rates) > 5:
				results.update(self._calculate_basic_stats(heart_rates, rr_intervals))
				results.update(self._detect_heart_rate_episodes(heart_rates))

			else:
				results = self._analyze_heart_rate_from_markers()

		except Exception as e:
			print(f"⚠️ Ошибка комплексного анализа ЭКГ: {e}")
			results = self._analyze_heart_rate_from_markers()

		return results

	def _get_clean_r_peaks(self, ecg_signal, sfreq, artifact_mask, artifact_regions, max_samples):
		"""Обнаружение R-пиков с исключением артефактов"""
		if artifact_mask is not None and len(artifact_mask) >= max_samples:
			segment_mask = artifact_mask[:max_samples]
			valid_segments = self.find_continuous_segments(segment_mask, min_segment_length=int(sfreq * 10))

			all_r_peaks = []
			for start, end in valid_segments:
				segment_ecg = ecg_signal[start:end]
				if len(segment_ecg) > int(sfreq * 5):
					segment_peaks = self.detect_r_peaks(segment_ecg, sfreq)
					segment_peaks += start
					all_r_peaks.extend(segment_peaks)

			return np.array(all_r_peaks)
		else:
			return self.detect_r_peaks(ecg_signal, sfreq)

	def _calculate_heart_rate_metrics(self, r_peaks, sfreq):
		"""Расчет RR-интервалов и ЧСС с фильтрацией"""
		cfg = self.config['ecg']
		rr_intervals = np.diff(r_peaks) / sfreq

		valid_rr_mask = (rr_intervals > cfg['rr_min']) & (rr_intervals < cfg['rr_max'])
		valid_rr = rr_intervals[valid_rr_mask]

		if len(valid_rr) > 1:
			heart_rates = 60.0 / valid_rr
			valid_hr_mask = (heart_rates >= cfg['hr_min']) & (heart_rates <= cfg['hr_max'])
			valid_hr = heart_rates[valid_hr_mask]
			return valid_rr, valid_hr

		return np.array([]), np.array([])

	def _calculate_basic_stats(self, heart_rates, rr_intervals):
		"""Расчет базовых статистик ЧСС"""
		return {
			'avg_heart_rate': round(float(np.median(heart_rates)), 2),
			'min_heart_rate': round(float(np.percentile(heart_rates, 5)), 2),
			'max_heart_rate': round(float(np.percentile(heart_rates, 95)), 2),
			'heart_rate_variability': round(float(np.std(rr_intervals * 1000)), 2)
		}

	def _detect_heart_rate_episodes(self, heart_rates):
		"""Обнаружение эпизодов тахикардии и брадикардии"""
		cfg = self.config['ecg']
		episodes = {
			'tachycardia_events': 0,
			'bradycardia_events': 0
		}

		tachy_count = 0
		brady_count = 0
		tachy_episode = False
		brady_episode = False

		for hr in heart_rates:
			# Тахикардия
			if hr > cfg['tachycardia_threshold']:
				tachy_count += 1
				if tachy_count >= cfg['min_consecutive_events'] and not tachy_episode:
					episodes['tachycardia_events'] += 1
					tachy_episode = True
			else:
				tachy_count = 0
				tachy_episode = False

			# Брадикардия
			if hr < cfg['bradycardia_threshold']:
				brady_count += 1
				if brady_count >= cfg['min_consecutive_events'] and not brady_episode:
					episodes['bradycardia_events'] += 1
					brady_episode = True
			else:
				brady_count = 0
				brady_episode = False

		return episodes

	def _analyze_heart_rate_from_markers(self):
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
			annotations = self.raw.annotations
			for desc in annotations.description:
				desc_str = str(desc)
				if 'Тахикардия' in desc_str:
					results['tachycardia_events'] += 1
				elif 'Брадикардия' in desc_str:
					results['bradycardia_events'] += 1
		except Exception as e:
			print(f"⚠️ Ошибка анализа по маркерам: {e}")

		return results

	def find_continuous_segments(self, mask, min_segment_length=1):
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

		if start is not None and len(mask) - start >= min_segment_length:
			segments.append((start, len(mask)))

		return segments

	def detect_r_peaks(self, ecg_signal, sfreq):
		"""Детекция R-зубцов в ЭКГ сигнале с адаптивным порогом"""
		try:
			ecg_clean = ecg_signal - np.median(ecg_signal)

			if len(ecg_clean) > 100:
				b, a = signal.butter(3, [5 / (sfreq / 2), 35 / (sfreq / 2)], btype='band')
				ecg_filtered = signal.filtfilt(b, a, ecg_clean)
			else:
				ecg_filtered = ecg_clean

			ecg_squared = np.square(ecg_filtered)
			window_size = int(0.12 * sfreq)
			if window_size % 2 == 0:
				window_size += 1
			ecg_smoothed = signal.medfilt(ecg_squared, kernel_size=window_size)

			# 3. АДАПТИВНЫЙ ПОРОГ ДЛЯ ДИНАМИЧЕСКИХ ИЗМЕНЕНИЙ
			window_size_adaptive = 5 * sfreq  # 5-секундное окно
			overlap = window_size_adaptive // 2  # 50% перекрытие

			adaptive_thresholds = np.zeros_like(ecg_smoothed)

			for i in range(0, len(ecg_smoothed), overlap):
				end_idx = min(i + window_size_adaptive, len(ecg_smoothed))
				window = ecg_smoothed[i:end_idx]

				if len(window) > 0:
					# Используем 85-й процентиль для каждого окна
					window_threshold = np.percentile(window, 85)
					# Заполняем соответствующий сегмент этим порогом
					adaptive_thresholds[i:end_idx] = window_threshold

			# Если массив слишком короткий для скользящего окна, используем глобальный порог
			if len(ecg_smoothed) < window_size_adaptive:
				adaptive_thresholds[:] = np.percentile(ecg_smoothed, 85)

			# Поиск пиков с адаптивным порогом
			peaks = []
			for i in range(len(ecg_smoothed)):
				# Находим пики, превышающие локальный порог
				if (ecg_smoothed[i] > adaptive_thresholds[i] and
						(i == 0 or ecg_smoothed[i] > ecg_smoothed[i - 1]) and
						(i == len(ecg_smoothed) - 1 or ecg_smoothed[i] > ecg_smoothed[i + 1])):

					# Проверяем минимальное расстояние от предыдущего пика
					if len(peaks) == 0 or (i - peaks[-1]) >= int(0.3 * sfreq):
						peaks.append(i)

			peaks = np.array(peaks)

			# ВАЛИДАЦИЯ НАЙДЕННЫХ ПИКОВ (дополнительное улучшение)
			if len(peaks) > 0:
				valid_peaks = []
				for peak in peaks:
					# Проверяем, что амплитуда в исходном сигнале значима
					if (ecg_signal[peak] > np.median(ecg_signal) + 0.1 * np.std(ecg_signal)):
						valid_peaks.append(peak)
				peaks = np.array(valid_peaks)

			return peaks

		except Exception as e:
			print(f"Ошибка в детекции R-пиков: {e}")
			return np.array([])

	def analyze_spo2_channel_fast(self):
		"""Быстрый анализ SpO2 с исключением артефактов"""
		spo2_stats = {
			'avg_spo2': None, 'min_spo2': None,
			'time_below_spo2_90': 0, 'time_below_spo2_85': 0,
			'spo2_baseline': None,
			'artifact_regions_excluded': 0
		}

		try:
			artifact_mask, artifact_regions = self.get_artifact_masks()
			cfg = self.config['spo2']

			spo2_channels = [ch for ch in self.raw.ch_names if any(x in ch.lower() for x in ['spo2', 'sao2', 'sat'])]
			if spo2_channels:
				spo2_idx = self.raw.ch_names.index(spo2_channels[0])
				data, times = self.raw[spo2_idx, :]
				if len(data) > 0:
					spo2_values = data[0]

					if artifact_mask is not None:
						valid_spo2 = spo2_values[(spo2_values >= cfg['min_valid']) &
						                         (spo2_values <= cfg['max_valid']) & artifact_mask]
						spo2_stats['artifact_regions_excluded'] = len(artifact_regions)
					else:
						valid_spo2 = spo2_values[(spo2_values >= cfg['min_valid']) &
						                         (spo2_values <= cfg['max_valid'])]

					if len(valid_spo2) > 0:
						spo2_stats['avg_spo2'] = round(float(np.median(valid_spo2)), 1)
						spo2_stats['min_spo2'] = round(float(np.percentile(valid_spo2, 1)), 1)
						spo2_stats['spo2_baseline'] = round(float(np.percentile(valid_spo2, 90)), 1)

						if artifact_mask is not None:
							total_valid_samples = np.sum(artifact_mask)
							if total_valid_samples > 0:
								samples_below_90 = np.sum(
									(spo2_values < cfg['threshold_90']) & artifact_mask &
									(spo2_values >= cfg['min_valid']) & (spo2_values <= cfg['max_valid']))
								samples_below_85 = np.sum(
									(spo2_values < cfg['threshold_85']) & artifact_mask &
									(spo2_values >= cfg['min_valid']) & (spo2_values <= cfg['max_valid']))

								total_duration_seconds = self.raw.times[-1]
								valid_duration_ratio = total_valid_samples / len(self.raw.times)

								time_below_90 = (samples_below_90 / total_valid_samples) * (
										total_duration_seconds / 60) * valid_duration_ratio
								time_below_85 = (samples_below_85 / total_valid_samples) * (
										total_duration_seconds / 60) * valid_duration_ratio

								spo2_stats['time_below_spo2_90'] = int(time_below_90)
								spo2_stats['time_below_spo2_85'] = int(time_below_85)

		except Exception as e:
			print(f"  ⚠️ Ошибка анализа SpO2: {e}")

		return spo2_stats

	def analyze_respiratory_channels_improved(self):
		"""Улучшенный анализ дыхательных каналов"""
		resp_stats = {
			'avg_resp_rate': None,
			'min_resp_rate': None,
			'max_resp_rate': None,
			'signal_quality': 'unknown'
		}

		try:
			resp_patterns = ['resp', 'breath', 'дыхание', 'thorax', 'chest', 'abdomen', 'flow']
			resp_channels = [
				ch for ch in self.raw.ch_names
				if any(pattern in ch.lower() for pattern in resp_patterns)
			]

			if not resp_channels:
				resp_stats['signal_quality'] = 'no_channel'
				return resp_stats

			best_rates = []
			for resp_ch in resp_channels[:2]:
				rates = self.analyze_single_resp_channel(resp_ch)
				if rates:
					best_rates.extend(rates)

			if not best_rates:
				resp_stats['signal_quality'] = 'no_signal'
				return resp_stats

			cfg = self.config['respiration']
			valid_rates = [r for r in best_rates if cfg['min_rate'] <= r <= cfg['max_rate']]

			if len(valid_rates) < 5:
				valid_rates = [r for r in best_rates if 6 <= r <= 30]

			if not valid_rates:
				resp_stats['signal_quality'] = 'invalid_rates'
				return resp_stats

			valid_rates = np.array(valid_rates)
			q1, q3 = np.percentile(valid_rates, [25, 75])
			iqr = q3 - q1
			lower_bound = q1 - 1.5 * iqr
			upper_bound = q3 + 1.5 * iqr

			final_rates = valid_rates[(valid_rates >= lower_bound) & (valid_rates <= upper_bound)]

			if len(final_rates) < 3:
				final_rates = valid_rates

			resp_stats['avg_resp_rate'] = round(float(np.median(final_rates)), 1)
			resp_stats['min_resp_rate'] = round(float(np.percentile(final_rates, 10)), 1)
			resp_stats['max_resp_rate'] = round(float(np.percentile(final_rates, 90)), 1)
			resp_stats['signal_quality'] = 'good' if len(final_rates) >= 10 else 'moderate'

		except Exception as e:
			print(f"  ⚠️ Ошибка улучшенного анализа дыхания: {e}")
			resp_stats['signal_quality'] = 'error'

		return resp_stats

	def analyze_single_resp_channel(self, channel_name):
		"""Анализ одного дыхательного канала с исключением артефактов"""
		try:
			artifact_mask, artifact_regions = self.get_artifact_masks()
			channel_idx = self.raw.ch_names.index(channel_name)
			sfreq = self.raw.info['sfreq']
			cfg = self.config['respiration']

			max_samples = len(self.raw.times)
			data, times = self.raw[channel_idx, :max_samples]

			if len(data) == 0:
				return []

			resp_signal = data[0]

			if artifact_mask is not None and len(artifact_mask) >= max_samples:
				segment_mask = artifact_mask[:max_samples]
				valid_segments = self.find_continuous_segments(segment_mask,
				                                               min_segment_length=int(
					                                               sfreq * cfg['min_segment_length']))

				breathing_rates = []
				for start, end in valid_segments:
					segment_signal = resp_signal[start:end]
					if len(segment_signal) > int(sfreq * cfg['min_segment_length']):
						resp_clean = self.preprocess_resp_signal(segment_signal, sfreq)
						if resp_clean is not None:
							rate_peaks = self.analyze_breathing_peaks_improved(resp_clean, sfreq)
							if rate_peaks is not None and len(rate_peaks) > 0:  # ИСПРАВЛЕНИЕ: проверка на None и длину
								breathing_rates.extend(rate_peaks)  # ИСПРАВЛЕНИЕ: используем extend вместо append

				if breathing_rates:
					return breathing_rates

			# Если нет артефактов или после обработки артефактов нет данных
			resp_clean = self.preprocess_resp_signal(resp_signal, sfreq)
			if resp_clean is not None:
				rate_peaks = self.analyze_breathing_peaks_improved(resp_clean, sfreq)
				return rate_peaks if rate_peaks is not None else []  # ИСПРАВЛЕНИЕ: проверка на None

		except Exception as e:
			print(f"    ⚠️ Ошибка анализа канала {channel_name}: {e}")

		return []

	def analyze_breathing_peaks_improved(self, resp_signal, sfreq):
		"""Улучшенный анализ дыхательных пиков"""
		try:
			# ИСПРАВЛЕНИЕ: используем более простой подход для анализа дыхания
			resp_normalized = (resp_signal - np.mean(resp_signal)) / (np.std(resp_signal) + 1e-8)

			# Поиск пиков в нормализованном сигнале
			peaks, properties = signal.find_peaks(
				resp_normalized,
				distance=int(1.0 * sfreq),  # минимум 1 секунда между вдохами
				prominence=0.3,
				height=0.2
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
				(breathing_rates >= 8) & (breathing_rates <= 25)
				]

			return valid_rates.tolist() if len(valid_rates) > 0 else []

		except Exception as e:
			print(f"      ⚠️ Ошибка анализа дыхательных пиков: {e}")
			return []

	def preprocess_resp_signal(self, resp_signal, sfreq):
		"""Предобработка дыхательного сигнала"""
		try:
			resp_clean = resp_signal - np.median(resp_signal)
			cfg = self.config['respiration']
			b, a = signal.butter(3, [cfg['filter_low'] / (sfreq / 2), cfg['filter_high'] / (sfreq / 2)], btype='band')
			resp_filtered = signal.filtfilt(b, a, resp_clean)
			return resp_filtered
		except Exception as e:
			return None

	def export_minimal_hypnogram(self):
		"""Экспорт гипнограммы в минимальном формате для SQL"""
		if not self.raw or not hasattr(self.raw, 'annotations'):
			return None

		annotations = self.raw.annotations

		stage_mapping = {
			'Sleep stage W(eventUnknown)': 'W',
			'Sleep stage 1(eventUnknown)': '1',
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

		return minimal_data

	def print_annotation_statistics(self):
		"""Вывод статистики аннотаций в порядке убывания количества"""
		if not self.raw or not hasattr(self.raw, 'annotations'):
			print("❌ Нет аннотаций для анализа")
			return

		annotations = self.raw.annotations
		annotation_counts = Counter(str(desc) for desc in annotations.description)

		print("\n📊 СТАТИСТИКА АННОТАЦИЙ")
		print("=" * 50)
		print(f"Всего аннотаций: {len(annotations)}")
		print(f"Уникальных типов: {len(annotation_counts)}")
		print("\nТипы аннотаций (по убыванию):")

		for desc, count in sorted(annotation_counts.items(), key=lambda x: x[1], reverse=True):
			print(f"  {count:>5} × {desc}")

	def generate_sql_insert_statements(self, edf_path, patient_info):
		"""Генерация SQL UPDATE запросов в правильном формате"""
		if not self.raw:
			return None

		# Основные расчеты
		stages = self.calculate_sleep_stages() or {}
		efficiency = self.calculate_sleep_efficiency() or {}
		hypnogram_data = self.export_minimal_hypnogram()
		total_sleep_time = efficiency.get('total_sleep_time', 0)
		architecture = self.calculate_sleep_architecture() or {}
		fragmentation = self.calculate_sleep_fragmentation() or {}
		respiratory_events = self.calculate_respiratory_events() or {}
		sleep_indices = self.calculate_sleep_indices() or {}
		rem_quality = self.calculate_rem_quality() or {}
		hr_stats = self.analyze_heart_rate_comprehensive() or {}
		spo2_stats = self.analyze_spo2_channel_fast() or {}
		resp_stats = self.analyze_respiratory_channels_improved() or {}
		latencies = self.calculate_sleep_latencies() or {}
		overall_quality = self.calculate_overall_sleep_quality() or {}
		rem_cycles = self.calculate_rem_cycles()

		# Расчет статистики артефактов
		artifact_mask, artifact_regions = self.get_artifact_masks()
		artifact_count = len(artifact_regions) if artifact_regions else 0
		artifact_duration_minutes = sum(
			region['duration'] for region in artifact_regions) / 60 if artifact_regions else 0

		# Подготовка данных для SQL UPDATE
		sql_data = {
			# Общие параметры сна
			'total_sleep_time': int(efficiency.get('total_sleep_time', 0)),
			'total_bed_time': int(efficiency.get('total_bed_time', 0)),
			'sleep_efficiency': round(efficiency.get('sleep_efficiency', 0), 2),
			'sleep_latency': int(latencies.get('sleep_onset_latency', 0)) if latencies.get(
				'sleep_onset_latency') else 0,
			'wake_after_sleep_onset': int(efficiency.get('wake_after_sleep_onset', 0)),

			# Стадии сна (минуты)
			'n1_minutes': int(stages['N1']['minutes']) if stages else 0,
			'n2_minutes': int(stages['N2']['minutes']) if stages else 0,
			'n3_minutes': int(stages['N3']['minutes']) if stages else 0,
			'rem_minutes': int(stages['REM']['minutes']) if stages else 0,

			# Стадии сна (проценты)
			'n1_percentage': round(architecture.get('n1_percentage', 0), 2),
			'n2_percentage': round(architecture.get('n2_percentage', 0), 2),
			'n3_percentage': round(architecture.get('n3_percentage', 0), 2),
			'rem_percentage': round(architecture.get('rem_percentage', 0), 2),

			# REM-сон
			'rem_latency': int(latencies.get('rem_latency')) if latencies.get('rem_latency') else None,
			'rem_epochs': stages['REM']['count'] if stages else None,
			'rem_cycles': rem_cycles,
			'rem_events': rem_quality.get('rem_events'),
			'rem_density': round(rem_quality.get('rem_density', 0), 2) if rem_quality.get('rem_density') else None,
			'rem_quality_score': rem_quality.get('rem_quality_score'),

			# Дыхательные нарушения (события)
			'total_apneas': respiratory_events.get('apneas', 0),
			'obstructive_apneas': respiratory_events.get('apneas', 0),
			'central_apneas': 0,  # Требует дополнительного анализа
			'mixed_apneas': 0,  # Требует дополнительного анализа
			'total_hypopneas': respiratory_events.get('hypopneas', 0),
			'total_desaturations': respiratory_events.get('desaturations', 0),
			'total_snores': respiratory_events.get('snoring', 0),

			# Дыхательные индексы
			'ahi': round(sleep_indices.get('ahi', 0), 2),
			'ahi_obstructive': round(sleep_indices.get('ahi', 0), 2),
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
			'total_limb_movements': fragmentation.get('total_limb_movements', 0),
			'periodic_limb_movements': fragmentation.get('periodic_limb_movements', 0),
			'plmi': round(fragmentation.get('periodic_limb_movements', 0) / (total_sleep_time / 60),
			              2) if total_sleep_time > 0 else 0,
			'bruxism_events': fragmentation.get('bruxism_events', 0),

			# Активации и фрагментация
			'total_arousals': fragmentation.get('activations', 0),
			'arousal_index': round(fragmentation.get('arousal_index', 0), 2),
			'sleep_fragmentation_index': round(fragmentation.get('fragmentation_index', 0), 2),

			# Общая оценка качества сна
			'overall_sleep_quality': overall_quality.get('overall_score'),
			'sleep_quality_status': overall_quality.get('status'),
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

		# Генерация полного SQL
		uuid = patient_info.get('uuid', 'unknown')
		sql = f"""-- SQL запрос для обновления статистики сна
	-- UUID исследования: {uuid}
	-- Файл: {os.path.basename(edf_path)}
	-- Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

	-- ВАЖНО: Эта запись должна выполняться ПОСЛЕ импорта исследования через процедуру
	-- Исследование должно уже существовать в таблицах psg_studies и sleep_statistics

	-- Обновление статистики сна
	UPDATE `sleep_statistics` ss
	JOIN `psg_studies` ps ON ss.study_id = ps.study_id
	SET {', '.join(set_parts)}
	WHERE ps.edf_uuid = '{uuid}';

	-- Обновление информации об артефактах в psg_studies
	UPDATE `psg_studies` 
	SET `artifact_count` = {artifact_count}, 
	    `artifact_duration_minutes` = {round(artifact_duration_minutes, 2)}
	WHERE `edf_uuid` = '{uuid}';"""

		# Сохранение в файл
		sql_filename = f"sleep_stats_{uuid}.sql"
		try:
			with open(sql_filename, 'w', encoding='utf-8') as f:
				f.write(sql + "\n")

			print(f"✅ SQL файл создан: {sql_filename}")
			print(f"📊 UUID исследования: {uuid}")
			print(f"📝 Тип запроса: UPDATE (обновление существующей записи)")
			print(f"🚫 Артефакты: {artifact_count} регионов, {round(artifact_duration_minutes, 2)} минут")

			return sql_filename

		except Exception as e:
			print(f"❌ Ошибка создания SQL файла: {e}")
			return None

	def process_edf_folder(self, folder_path, output_folder="sql_output"):
		"""Пакетная обработка всех EDF файлов в папке"""
		import glob

		# Создание папки для выходных файлов
		os.makedirs(output_folder, exist_ok=True)

		# Поиск всех EDF файлов в папке
		edf_files = glob.glob(os.path.join(folder_path, "*.edf"))

		if not edf_files:
			print(f"❌ В папке {folder_path} не найдено EDF файлов")
			return []

		print(f"📁 Найдено {len(edf_files)} EDF файлов для обработки")

		processed_files = []
		failed_files = []

		for i, edf_path in enumerate(edf_files, 1):
			print(f"\n{'=' * 60}")
			print(f"🔍 Обработка файла {i}/{len(edf_files)}: {os.path.basename(edf_path)}")
			print(f"{'=' * 60}")

			try:
				# Загрузка файла
				raw = self.load_edf_file(edf_path)
				if not raw:
					print(f"❌ Не удалось загрузить файл: {os.path.basename(edf_path)}")
					failed_files.append(edf_path)
					continue

				# Извлечение информации о пациенте
				patient_info = self.extract_patient_info_from_edf(edf_path)

				# Генерация SQL
				sql_filename = self.generate_sql_insert_statements(edf_path, patient_info)

				if sql_filename:
					# Перемещение SQL файла в выходную папку
					new_sql_path = os.path.join(output_folder, os.path.basename(sql_filename))
					os.rename(sql_filename, new_sql_path)
					processed_files.append(new_sql_path)

					print(f"✅ Успешно обработан: {os.path.basename(edf_path)}")
					print(f"📄 SQL файл: {os.path.basename(new_sql_path)}")
				else:
					print(f"❌ Не удалось создать SQL для: {os.path.basename(edf_path)}")
					failed_files.append(edf_path)

			except Exception as e:
				print(f"❌ Ошибка при обработке {os.path.basename(edf_path)}: {e}")
				failed_files.append(edf_path)

		# Сводка обработки
		print(f"\n{'=' * 60}")
		print("📊 СВОДКА ОБРАБОТКИ")
		print(f"{'=' * 60}")
		print(f"✅ Успешно обработано: {len(processed_files)} файлов")
		print(f"❌ Не удалось обработать: {len(failed_files)} файлов")

		if processed_files:
			print(f"\n📁 SQL файлы сохранены в папке: {output_folder}")
			for file in processed_files:
				print(f"  • {os.path.basename(file)}")

		if failed_files:
			print(f"\n🚫 Файлы с ошибками:")
			for file in failed_files:
				print(f"  • {os.path.basename(file)}")

		return processed_files

def main():
	"""Основная функция для демонстрации работы"""
	analyzer = SleepAnalyzer()

	# Обработка одного файла
	# edf_path = "EDF/test2.edf"
	# raw = analyzer.load_edf_file(edf_path)
	# if raw:
	#     analyzer.print_annotation_statistics()
	#     patient_info = analyzer.extract_patient_info_from_edf(edf_path)
	#     sql_filename = analyzer.generate_sql_insert_statements(edf_path, patient_info)

	# Пакетная обработка всех файлов в папке
	folder_path = "EDF"  # замените на вашу папку
	processed_files = analyzer.process_edf_folder(folder_path)

	if processed_files:
		print(f"\n🎉 Пакетная обработка завершена! Создано {len(processed_files)} SQL файлов")

if __name__ == "__main__":
	main()