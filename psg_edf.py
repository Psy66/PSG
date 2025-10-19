import os
import re
import json
import concurrent.futures
from datetime import datetime
from threading import Lock
import mne
import numpy as np
from scipy import signal

CONFIG = {
	'ecg': {'rr_min': 0.3, 'rr_max': 2.0, 'hr_min': 40, 'hr_max': 150},
	'respiration': {'min_rate': 8, 'max_rate': 25, 'filter_low': 0.1, 'filter_high': 1.0},
	'spo2': {'min_valid': 75, 'max_valid': 100, 'threshold_90': 90, 'threshold_85': 85},
	'sleep_quality': {
		'efficiency_weights': {85: 25, 70: 20, 50: 10},
		'n3_threshold': 15, 'rem_threshold': 20,
		'ahi_weights': {5: 30, 15: 20, 30: 10},
		'arousal_weights': {10: 15, 20: 10}
	}
}

class ArtifactProcessor:
	def get_artifact_mask(self, raw, artifact_marker='Артефакт(blockArtefact)'):
		if not raw or not hasattr(raw, 'annotations'):
			return None, []

		sfreq = raw.info['sfreq']
		total_samples = len(raw.times)
		valid_mask = np.ones(total_samples, dtype=bool)
		artifact_regions = []

		for desc, onset, duration in zip(raw.annotations.description, raw.annotations.onset, raw.annotations.duration):
			if artifact_marker in str(desc):
				start = int(onset * sfreq)
				end = min(int((onset + duration) * sfreq), total_samples)
				if start < total_samples:
					valid_mask[start:end] = False
					artifact_regions.append({'start_time': onset, 'end_time': onset + duration, 'duration': duration})

		gap_mask, gap_regions = self.get_heartbeat_gaps(raw)
		if gap_mask is not None:
			valid_mask &= ~gap_mask
			artifact_regions.extend(gap_regions)

		return valid_mask, artifact_regions

	def get_heartbeat_gaps(self, raw, marker='pointIlluminationSensorValue', max_gap=5.0, min_duration=10.0):
		if not raw or not hasattr(raw, 'annotations'):
			return None, []

		annotations = raw.annotations
		sfreq = raw.info['sfreq']
		total_samples = len(raw.times)

		times = annotations.onset[annotations.description == marker]
		if len(times) < 2:
			return None, []

		gap_mask = np.zeros(total_samples, dtype=bool)
		gap_regions = []

		intervals = np.diff(times)
		gap_indices = np.where(intervals > max_gap)[0]

		for idx in gap_indices:
			start_time = times[idx]
			end_time = times[idx + 1]
			duration = intervals[idx]

			if duration >= min_duration:
				start_sample = int(start_time * sfreq)
				end_sample = min(int(end_time * sfreq), total_samples)
				if start_sample < total_samples:
					gap_mask[start_sample:end_sample] = True
					gap_regions.append({
						'start_time': start_time, 'end_time': end_time,
						'duration': duration, 'type': 'heartbeat_gap'
					})

		return gap_mask, gap_regions

class SignalAnalyzer:
	def __init__(self, config):
		self.config = config
		self.artifact_processor = ArtifactProcessor()

	def analyze_ecg(self, raw):
		results = {'avg_heart_rate': None, 'min_heart_rate': None, 'max_heart_rate': None,
		           'heart_rate_variability': None, 'tachycardia_events': 0, 'bradycardia_events': 0}

		try:
			if raw and hasattr(raw, 'annotations'):
				for desc in raw.annotations.description:
					desc_str = str(desc)
					if 'Тахикардия' in desc_str:
						results['tachycardia_events'] += 1
					elif 'Брадикардия' in desc_str:
						results['bradycardia_events'] += 1

			ecg_channels = [ch for ch in raw.ch_names if
			                any(kw in ch.lower() for kw in ['ecg', 'ekg', 'electrocardiogram'])]
			if not ecg_channels:
				return results

			artifact_mask, _ = self.artifact_processor.get_artifact_mask(raw)
			ecg_idx = raw.ch_names.index(ecg_channels[0])
			sfreq = raw.info['sfreq']

			data, _ = raw[ecg_idx, :]
			if len(data) == 0:
				return results

			ecg_signal = data[0]

			if artifact_mask is not None:
				ecg_signal = ecg_signal[artifact_mask[:len(ecg_signal)]]
				if len(ecg_signal) == 0:
					return results

			r_peaks = self.detect_r_peaks(ecg_signal, sfreq)
			if len(r_peaks) > 100:
				rr_intervals = np.diff(r_peaks) / sfreq
				cfg = self.config['ecg']
				valid_rr = rr_intervals[(rr_intervals > cfg['rr_min']) & (rr_intervals < cfg['rr_max'])]

				if len(valid_rr) > 1:
					hr = 60.0 / valid_rr
					valid_hr = hr[(hr >= cfg['hr_min']) & (hr <= cfg['hr_max'])]

					if len(valid_hr) > 5:
						results.update({
							'avg_heart_rate': round(float(np.median(valid_hr)), 2),
							'min_heart_rate': round(float(np.percentile(valid_hr, 5)), 2),
							'max_heart_rate': round(float(np.percentile(valid_hr, 95)), 2),
							'heart_rate_variability': round(float(np.std(valid_rr * 1000)), 2)
						})

		except Exception as e:
			print(f"ECG analysis error: {e}")

		return results

	def detect_r_peaks(self, ecg_signal, sfreq):
		try:
			ecg_clean = ecg_signal - np.median(ecg_signal)

			if len(ecg_clean) > 100:
				# Явное использование scipy.signal
				import scipy.signal as scipy_signal
				b, a = scipy_signal.butter(3, [5 / (sfreq / 2), 35 / (sfreq / 2)], btype='band')
				ecg_filtered = scipy_signal.filtfilt(b, a, ecg_clean)
			else:
				ecg_filtered = ecg_clean

			ecg_squared = np.square(ecg_filtered)
			window_size = int(0.1 * sfreq)
			if window_size % 2 == 0:
				window_size += 1

			ecg_smoothed = scipy_signal.medfilt(ecg_squared, kernel_size=window_size)

			threshold = np.percentile(ecg_smoothed, 85)
			peaks, _ = scipy_signal.find_peaks(ecg_smoothed, height=threshold, distance=int(0.3 * sfreq))

			return peaks
		except Exception as e:
			print(f"R-peak detection error: {e}")
			return np.array([])

	def analyze_spo2(self, raw):
		stats = {'avg_spo2': None, 'min_spo2': None, 'spo2_baseline': None, 'time_below_spo2_90': 0,
		         'time_below_spo2_85': 0}

		try:
			artifact_mask, artifact_regions = self.artifact_processor.get_artifact_mask(raw)
			cfg = self.config['spo2']

			spo2_channels = [ch for ch in raw.ch_names if any(x in ch.lower() for x in ['spo2', 'sao2', 'sat'])]
			if spo2_channels:
				spo2_idx = raw.ch_names.index(spo2_channels[0])
				data, _ = raw[spo2_idx, :]
				if len(data) > 0:
					spo2_values = data[0]

					if artifact_mask is not None:
						valid_spo2 = spo2_values[(spo2_values >= cfg['min_valid']) &
						                         (spo2_values <= cfg['max_valid']) & artifact_mask]
					else:
						valid_spo2 = spo2_values[(spo2_values >= cfg['min_valid']) &
						                         (spo2_values <= cfg['max_valid'])]

					if len(valid_spo2) > 0:
						stats['avg_spo2'] = round(float(np.median(valid_spo2)), 1)
						stats['min_spo2'] = round(float(np.percentile(valid_spo2, 1)), 1)
						stats['spo2_baseline'] = round(float(np.percentile(valid_spo2, 90)), 1)

						if artifact_mask is not None:
							total_valid = np.sum(artifact_mask)
							if total_valid > 0:
								below_90 = np.sum((spo2_values < cfg['threshold_90']) & artifact_mask &
								                  (spo2_values >= cfg['min_valid']) & (spo2_values <= cfg['max_valid']))
								below_85 = np.sum((spo2_values < cfg['threshold_85']) & artifact_mask &
								                  (spo2_values >= cfg['min_valid']) & (spo2_values <= cfg['max_valid']))

								total_duration = raw.times[-1]
								valid_ratio = total_valid / len(raw.times)

								stats['time_below_spo2_90'] = int(
									(below_90 / total_valid) * (total_duration / 60) * valid_ratio)
								stats['time_below_spo2_85'] = int(
									(below_85 / total_valid) * (total_duration / 60) * valid_ratio)

		except Exception as e:
			print(f"SpO2 analysis error: {e}")

		return stats

	def analyze_breathing(self, resp_signal, sfreq):
		try:
			if len(resp_signal) < int(sfreq * 20):
				return []

			normalized = (resp_signal - np.mean(resp_signal)) / (np.std(resp_signal) + 1e-8)
			min_distance = int(0.6 * sfreq)

			peaks, properties = signal.find_peaks(
				normalized,
				distance=min_distance,
				prominence=0.05,
				height=0.02,
				width=int(0.2 * sfreq),
				wlen=int(2 * sfreq)
			)

			if len(peaks) < 3:
				peaks, properties = signal.find_peaks(
					normalized,
					distance=min_distance,
					prominence=0.02,
					height=0.01
				)

			if len(peaks) < 3:
				return []

			breath_intervals = np.diff(peaks) / sfreq
			valid_intervals = breath_intervals[(breath_intervals >= 0.5) & (breath_intervals <= 10.0)]

			if len(valid_intervals) < 2:
				return []

			breathing_rates = 60.0 / valid_intervals
			valid_rates = breathing_rates[(breathing_rates >= 6) & (breathing_rates <= 60)]

			if len(valid_rates) > 5:
				q1, q3 = np.percentile(valid_rates, [25, 75])
				iqr = q3 - q1
				lower_bound = q1 - 1.5 * iqr
				upper_bound = q3 + 1.5 * iqr
				valid_rates = valid_rates[(valid_rates >= lower_bound) & (valid_rates <= upper_bound)]

			return valid_rates.tolist() if len(valid_rates) > 0 else []

		except Exception as e:
			print(f"Breathing analysis error: {e}")
			return []

	def analyze_resp_channel(self, raw, channel_name):
		try:
			artifact_mask, _ = self.artifact_processor.get_artifact_mask(raw)
			ch_idx = raw.ch_names.index(channel_name)
			sfreq = raw.info['sfreq']

			data, _ = raw[ch_idx, :]
			if len(data) == 0:
				return []

			resp_signal = data[0]

			if artifact_mask is not None and len(artifact_mask) == len(resp_signal):
				resp_signal = resp_signal[artifact_mask]
				if len(resp_signal) == 0:
					return []

			resp_clean = self.preprocess_resp(resp_signal, sfreq)
			return self.analyze_breathing(resp_clean, sfreq) if resp_clean is not None else []

		except Exception as e:
			print(f"Resp channel {channel_name} error: {e}")
			return []

	def preprocess_resp(self, signal_data, sfreq):
		try:
			if len(signal_data) == 0:
				return None

			cleaned = signal_data - np.median(signal_data)
			signal_std = np.std(cleaned)
			if signal_std < 1e-8:
				return None
			normalized = cleaned / signal_std

			cfg = self.config['respiration']
			low_freq = cfg['filter_low'] / (sfreq / 2)
			high_freq = cfg['filter_high'] / (sfreq / 2)

			if low_freq >= 1.0 or high_freq >= 1.0:
				return normalized

			import scipy.signal as scipy_signal
			b, a = scipy_signal.butter(3, [low_freq, high_freq], btype='band')
			filtered = scipy_signal.filtfilt(b, a, normalized)

			return filtered

		except Exception as e:
			print(f"Resp preprocessing error: {e}")
			return None

	def analyze_respiration(self, raw):
		stats = {'avg_resp_rate': None, 'min_resp_rate': None, 'max_resp_rate': None}

		try:
			resp_keywords = ['resp', 'breath', 'дыхание', 'thorax', 'chest', 'abdomen', 'flow', 'rip', 'pleth']
			resp_channels = [ch for ch in raw.ch_names if any(p in ch.lower() for p in resp_keywords)]

			if not resp_channels:
				return stats

			best_rates = []
			for ch in resp_channels[:3]:
				rates = self.analyze_resp_channel(raw, ch)
				if rates:
					best_rates.extend(rates)

			if not best_rates:
				return stats

			cfg = self.config['respiration']
			valid_rates = [r for r in best_rates if cfg['min_rate'] <= r <= cfg['max_rate']]

			if len(valid_rates) < 5:
				valid_rates = [r for r in best_rates if 6 <= r <= 40]

			if not valid_rates:
				return stats

			valid_rates = np.array(valid_rates)

			if len(valid_rates) > 5:
				q1, q3 = np.percentile(valid_rates, [25, 75])
				iqr = q3 - q1
				final_rates = valid_rates[(valid_rates >= q1 - 1.5 * iqr) & (valid_rates <= q3 + 1.5 * iqr)]
			else:
				final_rates = valid_rates

			final_rates = final_rates if len(final_rates) >= 3 else valid_rates

			if len(final_rates) > 0:
				stats.update({
					'avg_resp_rate': round(float(np.median(final_rates)), 1),
					'min_resp_rate': round(float(np.percentile(final_rates, 10)), 1),
					'max_resp_rate': round(float(np.percentile(final_rates, 90)), 1)
				})

		except Exception as e:
			print(f"Respiration analysis error: {e}")

		return stats

class SleepAnalyzer:
	def __init__(self, config=None):
		self.config = config or CONFIG
		self.raw = None
		self.stages = None
		self.signal_analyzer = SignalAnalyzer(self.config)
		self.artifact_processor = ArtifactProcessor()

	def load_edf(self, path):
		try:
			self.raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
			return self.raw
		except Exception as e:
			print(f"Load error {path}: {e}")
			return None

	def extract_uuid(self, path):
		try:
			with open(path, 'rb') as f:
				header = f.read(256).decode('latin-1', errors='ignore')
				patient_info = header[8:168].strip()
				match = re.search(r'([a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})',
				                  patient_info)
				return match.group(1) if match else None
		except Exception as e:
			print(f"UUID extract error {path}: {e}")
			return None

	def calculate_stages(self):
		if not self.raw or not hasattr(self.raw, 'annotations'):
			return None

		annotations = self.raw.annotations
		mapping = {
			'Sleep stage W(eventUnknown)': 'Wake',
			'Sleep stage 1(eventUnknown)': 'N1',
			'Sleep stage 2(eventUnknown)': 'N2',
			'Sleep stage 3(eventUnknown)': 'N3',
			'Sleep stage R(eventUnknown)': 'REM',
			'Sleep stage Unknown(eventUnknown)': 'Unknown'
		}

		stages = {s: {'count': 0, 'minutes': 0} for s in ['Wake', 'N1', 'N2', 'N3', 'REM', 'Unknown']}

		for desc, duration in zip(annotations.description, annotations.duration):
			desc_str = str(desc)
			if desc_str in mapping and abs(duration - 30) < 1:
				stage = mapping[desc_str]
				stages[stage]['count'] += 1
				stages[stage]['minutes'] += 0.5

		self.stages = stages
		return stages

	def calculate_efficiency(self):
		if not self.stages:
			return None

		total_sleep = sum(self.stages[s]['minutes'] for s in ['N1', 'N2', 'N3', 'REM'])
		total_bed = sum(s['minutes'] for s in self.stages.values())
		efficiency = (total_sleep / total_bed * 100) if total_bed > 0 else 0

		return {
			'sleep_efficiency': efficiency,
			'total_sleep_time': total_sleep,
			'total_bed_time': total_bed,
			'wake_after_sleep_onset': self.stages['Wake']['minutes']
		}

	def calculate_architecture(self):
		if not self.stages:
			return None

		total_sleep = sum(self.stages[s]['minutes'] for s in ['N1', 'N2', 'N3', 'REM'])
		if total_sleep == 0:
			return None

		return {
			'n1_percentage': (self.stages['N1']['minutes'] / total_sleep) * 100,
			'n2_percentage': (self.stages['N2']['minutes'] / total_sleep) * 100,
			'n3_percentage': (self.stages['N3']['minutes'] / total_sleep) * 100,
			'rem_percentage': (self.stages['REM']['minutes'] / total_sleep) * 100,
		}

	def calculate_latencies(self):
		annotations = self.raw.annotations
		first_sleep = first_rem = None
		current_time = 0

		for desc, duration in zip(annotations.description, annotations.duration):
			desc_str = str(desc)
			if desc_str in ['Sleep stage 1(eventUnknown)', 'Sleep stage 2(eventUnknown)',
			                'Sleep stage 3(eventUnknown)', 'Sleep stage R(eventUnknown)']:
				if first_sleep is None and abs(duration - 30) < 1:
					first_sleep = current_time

			if desc_str == 'Sleep stage R(eventUnknown)' and first_rem is None and abs(duration - 30) < 1:
				first_rem = current_time

			current_time += duration

		rem_latency = (first_rem - first_sleep) / 60 if first_sleep and first_rem else None
		return {
			'sleep_onset_latency': first_sleep / 60 if first_sleep else None,
			'rem_latency': rem_latency
		}

	def calculate_fragmentation(self):
		annotations = self.raw.annotations

		activations = sum(
			1 for desc in annotations.description if str(desc) == 'Активация(pointPolySomnographyActivation)')
		limb_movements = sum(1 for desc in annotations.description if
		                     str(desc) == 'Движение конечностей(pointPolySomnographyLegsMovements)')
		periodic_movements = sum(1 for desc in annotations.description if
		                         str(desc) == 'Периодические движения конечностей(pointPolySomnographyPeriodicalLegsMovements)')
		bruxism = sum(1 for desc in annotations.description if str(desc) == 'Бруксизм(pointBruxism)')

		total_sleep = sum(self.stages[s]['minutes'] for s in ['N1', 'N2', 'N3', 'REM'])
		total_movements = limb_movements + periodic_movements
		fragmentation_index = (activations + total_movements) / (total_sleep / 60) if total_sleep > 0 else 0

		return {
			'fragmentation_index': fragmentation_index,
			'activations': activations,
			'limb_movements': limb_movements,
			'periodic_limb_movements': periodic_movements,
			'bruxism_events': bruxism,
			'total_limb_movements': total_movements,
			'arousal_index': activations / (total_sleep / 60) if total_sleep > 0 else 0
		}

	def calculate_respiratory_events(self):
		annotations = self.raw.annotations
		mapping = {
			'Обструктивное апноэ(pointPolySomnographyObstructiveApnea)': 'obstructive_apneas',
			'Центральное апноэ(pointPolySomnographyCentralApnea)': 'central_apneas',
			'Смешанное апноэ(pointPolySomnographyMixedApnea)': 'mixed_apneas',
			'Обструктивное гипопноэ(pointPolySomnographyHypopnea)': 'obstructive_hypopneas',
			'Центральное гипопноэ(pointPolySomnographyCentralHypopnea)': 'central_hypopneas',
			'Смешанное гипопноэ(pointPolySomnographyMixedHypopnea)': 'mixed_hypopneas',
			'Десатурация(pointPolySomnographyDesaturation)': 'desaturations',
			'Храп(pointPolySomnographySnore)': 'snoring',
			'Дыхание Чейна-Стокса(pointPolySomnographyCheyneStokesRespiration)': 'cheyne_stokes'
		}

		events = {k: 0 for k in ['apneas', 'obstructive_apneas', 'central_apneas', 'mixed_apneas',
		                         'hypopneas', 'obstructive_hypopneas', 'central_hypopneas', 'mixed_hypopneas',
		                         'desaturations', 'snoring', 'cheyne_stokes']}

		for desc in annotations.description:
			desc_str = str(desc)
			if desc_str in mapping:
				events[mapping[desc_str]] += 1

		events['apneas'] = events['obstructive_apneas'] + events['central_apneas'] + events['mixed_apneas']
		events['hypopneas'] = events['obstructive_hypopneas'] + events['central_hypopneas'] + events['mixed_hypopneas']

		return events

	def calculate_indices(self):
		if not self.raw or not self.stages:
			return {}

		respiratory_events = self.calculate_respiratory_events() or {}
		total_sleep = sum(self.stages[s]['minutes'] for s in ['N1', 'N2', 'N3', 'REM'])

		if total_sleep == 0:
			return {}

		sleep_hours = total_sleep / 60
		ahi = (respiratory_events.get('apneas', 0) + respiratory_events.get('hypopneas', 0)) / sleep_hours

		ahi_obstructive = (respiratory_events.get('obstructive_apneas', 0) +
		                   respiratory_events.get('obstructive_hypopneas', 0)) / sleep_hours
		ahi_central = (respiratory_events.get('central_apneas', 0) +
		               respiratory_events.get('central_hypopneas', 0)) / sleep_hours
		ahi_mixed = (respiratory_events.get('mixed_apneas', 0) +
		             respiratory_events.get('mixed_hypopneas', 0)) / sleep_hours

		odi = respiratory_events.get('desaturations', 0) / sleep_hours
		snoring_index = respiratory_events.get('snoring', 0) / sleep_hours

		return {
			'ahi': ahi, 'ahi_obstructive': ahi_obstructive, 'ahi_central': ahi_central, 'ahi_mixed': ahi_mixed,
			'odi': odi, 'snoring_index': snoring_index,
			'total_apneas': respiratory_events.get('apneas', 0),
			'total_obstructive_apneas': respiratory_events.get('obstructive_apneas', 0),
			'total_central_apneas': respiratory_events.get('central_apneas', 0),
			'total_mixed_apneas': respiratory_events.get('mixed_apneas', 0),
			'total_hypopneas': respiratory_events.get('hypopneas', 0),
			'total_obstructive_hypopneas': respiratory_events.get('obstructive_hypopneas', 0),
			'total_central_hypopneas': respiratory_events.get('central_hypopneas', 0),
			'total_mixed_hypopneas': respiratory_events.get('mixed_hypopneas', 0),
			'total_desaturations': respiratory_events.get('desaturations', 0),
			'total_snores': respiratory_events.get('snoring', 0),
			'cheyne_stokes_episodes': respiratory_events.get('cheyne_stokes', 0)
		}

	def calculate_rem_quality(self):
		if not self.raw or not hasattr(self.raw, 'annotations'):
			return None

		annotations = self.raw.annotations
		rem_epochs = sum(1 for desc, duration in zip(annotations.description, annotations.duration)
		                 if str(desc) == 'Sleep stage R(eventUnknown)' and abs(duration - 30) < 1)
		rem_events = sum(1 for desc in annotations.description if str(desc) == 'БДГ(pointPolySomnographyREM)')

		rem_minutes = rem_epochs * 0.5
		rem_density = rem_events / rem_minutes if rem_minutes > 0 else 0

		time_score = 40 if rem_minutes >= 15 else 20 if rem_minutes >= 5 else 0
		density_score = 60 if rem_density >= 1.5 else 30 if rem_density >= 0.5 else 0
		quality_score = min(time_score + density_score, 100)

		status = "отлично" if quality_score >= 80 else "хорошо" if quality_score >= 60 else "удовлетворительно" if quality_score >= 40 else "низкое"

		return {
			'rem_quality_score': int(quality_score),
			'rem_minutes': rem_minutes,
			'rem_events': rem_events,
			'rem_density': rem_density,
			'status': status
		}

	def calculate_rem_cycles(self):
		if not self.raw or not hasattr(self.raw, 'annotations'):
			return 0

		annotations = self.raw.annotations
		mapping = {
			'Sleep stage W(eventUnknown)': 'W',
			'Sleep stage 1(eventUnknown)': 'N1',
			'Sleep stage 2(eventUnknown)': 'N2',
			'Sleep stage 3(eventUnknown)': 'N3',
			'Sleep stage R(eventUnknown)': 'R'
		}

		sequence = []
		for desc, duration in zip(annotations.description, annotations.duration):
			desc_str = str(desc)
			if desc_str in mapping and abs(duration - 30) < 1:
				sequence.append(mapping[desc_str])

		if not sequence:
			return 0

		cycles = 0
		in_rem = False
		rem_started = False

		for i in range(1, len(sequence) - 1):
			current, prev, next_ = sequence[i], sequence[i - 1], sequence[i + 1]

			if current == 'R' and prev in ['N1', 'N2', 'N3'] and not rem_started:
				rem_started = True
				in_rem = True
			elif current == 'R' and next_ in ['N1', 'N2', 'N3'] and in_rem:
				cycles += 1
				in_rem = False
				rem_started = False
			elif current == 'W' and in_rem:
				in_rem = False
				rem_started = False

		return cycles

	def calculate_sleep_quality(self):
		if not self.raw or not self.stages:
			return {}

		efficiency = self.calculate_efficiency() or {}
		architecture = self.calculate_architecture() or {}
		sleep_indices = self.calculate_indices() or {}
		fragmentation = self.calculate_fragmentation() or {}
		rem_quality = self.calculate_rem_quality() or {}
		hr_stats = self.signal_analyzer.analyze_ecg(self.raw) or {}
		rem_cycles = self.calculate_rem_cycles()

		score = 0
		cfg = self.config['sleep_quality']

		efficiency_val = efficiency.get('sleep_efficiency', 0)
		for threshold, points in cfg['efficiency_weights'].items():
			if efficiency_val >= threshold:
				score += points
				break

		n3_percentage = architecture.get('n3_percentage', 0)
		rem_percentage = architecture.get('rem_percentage', 0)

		if n3_percentage >= cfg['n3_threshold']:
			score += 15
		if rem_percentage >= cfg['rem_threshold']:
			score += 15

		ahi = sleep_indices.get('ahi', 0)
		for threshold, points in cfg['ahi_weights'].items():
			if ahi < threshold:
				score += points
				break

		arousal_index = fragmentation.get('arousal_index', 0)
		for threshold, points in cfg['arousal_weights'].items():
			if arousal_index < threshold:
				score += points
				break

		rem_score = rem_quality.get('rem_quality_score', 0)
		score += rem_score * 0.15

		tachycardia_events = hr_stats.get('tachycardia_events', 0)
		bradycardia_events = hr_stats.get('bradycardia_events', 0)

		if tachycardia_events > 10 or bradycardia_events > 10:
			score -= 15
		elif tachycardia_events > 5 or bradycardia_events > 5:
			score -= 10
		elif tachycardia_events > 0 or bradycardia_events > 0:
			score -= 5

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
			'overall_sleep_quality': int(overall_score),
			'sleep_quality_status': status
		}

	def export_hypnogram(self):
		if not self.raw or not hasattr(self.raw, 'annotations'):
			return None

		annotations = self.raw.annotations
		mapping = {
			'Sleep stage W(eventUnknown)': 'W',
			'Sleep stage 1(eventUnknown)': '1',
			'Sleep stage 2(eventUnknown)': '2',
			'Sleep stage 3(eventUnknown)': '3',
			'Sleep stage R(eventUnknown)': 'R'
		}

		sequence = []
		for desc, duration in zip(annotations.description, annotations.duration):
			desc_str = str(desc)
			if desc_str in mapping and abs(duration - 30) < 1:
				sequence.append(mapping[desc_str])

		return {'e': len(sequence), 'd': 30, 's': sequence}

	def generate_sql(self, edf_path, uuid):
		if not self.raw:
			return None

		stages = self.calculate_stages() or {}
		efficiency = self.calculate_efficiency() or {}
		architecture = self.calculate_architecture() or {}
		fragmentation = self.calculate_fragmentation() or {}
		respiratory_events = self.calculate_respiratory_events() or {}
		sleep_indices = self.calculate_indices() or {}
		rem_quality = self.calculate_rem_quality() or {}
		hr_stats = self.signal_analyzer.analyze_ecg(self.raw) or {}
		spo2_stats = self.signal_analyzer.analyze_spo2(self.raw) or {}
		resp_stats = self.signal_analyzer.analyze_respiration(self.raw) or {}
		latencies = self.calculate_latencies() or {}
		sleep_quality = self.calculate_sleep_quality() or {}
		hypnogram = self.export_hypnogram()
		rem_cycles = self.calculate_rem_cycles()

		_, artifact_regions = self.artifact_processor.get_artifact_mask(self.raw)
		artifact_count = len(artifact_regions)
		artifact_duration = sum(r['duration'] for r in artifact_regions) / 60

		total_sleep = efficiency.get('total_sleep_time', 0)

		sql_data = {
			'total_sleep_time': int(efficiency.get('total_sleep_time', 0)),
			'total_bed_time': int(efficiency.get('total_bed_time', 0)),
			'sleep_efficiency': round(efficiency.get('sleep_efficiency', 0), 2),
			'sleep_latency': int(latencies.get('sleep_onset_latency', 0)) if latencies.get(
				'sleep_onset_latency') else 0,
			'wake_after_sleep_onset': int(efficiency.get('wake_after_sleep_onset', 0)),
			'n1_minutes': int(stages['N1']['minutes']) if stages else 0,
			'n2_minutes': int(stages['N2']['minutes']) if stages else 0,
			'n3_minutes': int(stages['N3']['minutes']) if stages else 0,
			'rem_minutes': int(stages['REM']['minutes']) if stages else 0,
			'n1_percentage': round(architecture.get('n1_percentage', 0), 2),
			'n2_percentage': round(architecture.get('n2_percentage', 0), 2),
			'n3_percentage': round(architecture.get('n3_percentage', 0), 2),
			'rem_percentage': round(architecture.get('rem_percentage', 0), 2),
			'rem_latency': int(latencies.get('rem_latency')) if latencies.get('rem_latency') else None,
			'rem_epochs': stages['REM']['count'] if stages else None,
			'rem_cycles': rem_cycles,
			'rem_events': rem_quality.get('rem_events'),
			'rem_density': round(rem_quality.get('rem_density', 0), 2) if rem_quality.get('rem_density') else None,
			'rem_quality_score': rem_quality.get('rem_quality_score'),
			'total_apneas': respiratory_events.get('apneas', 0),
			'obstructive_apneas': respiratory_events.get('obstructive_apneas', 0),
			'central_apneas': respiratory_events.get('central_apneas', 0),
			'mixed_apneas': respiratory_events.get('mixed_apneas', 0),
			'total_hypopneas': respiratory_events.get('hypopneas', 0),
			'obstructive_hypopneas': respiratory_events.get('obstructive_hypopneas', 0),
			'central_hypopneas': respiratory_events.get('central_hypopneas', 0),
			'mixed_hypopneas': respiratory_events.get('mixed_hypopneas', 0),
			'total_desaturations': respiratory_events.get('desaturations', 0),
			'total_snores': respiratory_events.get('snoring', 0),
			'cheyne_stokes_episodes': respiratory_events.get('cheyne_stokes', 0),
			'ahi': round(sleep_indices.get('ahi', 0), 2),
			'ahi_obstructive': round(sleep_indices.get('ahi_obstructive', 0), 2),
			'ahi_central': round(sleep_indices.get('ahi_central', 0), 2),
			'ahi_mixed': round(sleep_indices.get('ahi_mixed', 0), 2),
			'odi': round(sleep_indices.get('odi', 0), 2),
			'snore_index': round(sleep_indices.get('snoring_index', 0), 2),
			'avg_spo2': spo2_stats.get('avg_spo2'),
			'min_spo2': spo2_stats.get('min_spo2'),
			'spo2_baseline': spo2_stats.get('spo2_baseline'),
			'time_below_spo2_90': spo2_stats.get('time_below_spo2_90', 0),
			'time_below_spo2_85': spo2_stats.get('time_below_spo2_85', 0),
			'avg_heart_rate': hr_stats.get('avg_heart_rate'),
			'min_heart_rate': hr_stats.get('min_heart_rate'),
			'max_heart_rate': hr_stats.get('max_heart_rate'),
			'heart_rate_variability': hr_stats.get('heart_rate_variability'),
			'tachycardia_events': hr_stats.get('tachycardia_events'),
			'bradycardia_events': hr_stats.get('bradycardia_events'),
			'avg_resp_rate': resp_stats.get('avg_resp_rate'),
			'min_resp_rate': resp_stats.get('min_resp_rate'),
			'max_resp_rate': resp_stats.get('max_resp_rate'),
			'total_limb_movements': fragmentation.get('total_limb_movements', 0),
			'periodic_limb_movements': fragmentation.get('periodic_limb_movements', 0),
			'plmi': round(fragmentation.get('periodic_limb_movements', 0) / (total_sleep / 60),
			              2) if total_sleep > 0 else 0,
			'bruxism_events': fragmentation.get('bruxism_events', 0),
			'total_arousals': fragmentation.get('activations', 0),
			'arousal_index': round(fragmentation.get('arousal_index', 0), 2),
			'sleep_fragmentation_index': round(fragmentation.get('fragmentation_index', 0), 2),
			'overall_sleep_quality': sleep_quality.get('overall_sleep_quality'),
			'sleep_quality_status': sleep_quality.get('sleep_quality_status'),
			'hypnogram_data': json.dumps(hypnogram) if hypnogram else None,
			'data_quality': 'good',
			'analysis_notes': f"Автоматический анализ файла: {os.path.basename(edf_path)}",
			'artifact_count': artifact_count,
			'artifact_duration_minutes': round(artifact_duration, 2)
		}

		return self.create_sql_update(sql_data, uuid, edf_path)

	def create_sql_update(self, data, uuid, edf_path):
		set_parts = []
		for key, value in data.items():
			if key in ['artifact_count', 'artifact_duration_minutes']:
				continue

			if value is None:
				set_parts.append(f"`{key}` = NULL")
			elif isinstance(value, str):
				escaped = value.replace("'", "''")
				set_parts.append(f"`{key}` = '{escaped}'")
			else:
				set_parts.append(f"`{key}` = {value}")

		sql = f"""-- SQL запрос для обновления статистики сна
-- UUID исследования: {uuid}
-- Файл: {os.path.basename(edf_path)}
-- Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

UPDATE `sleep_statistics` ss
JOIN `psg_studies` ps ON ss.study_id = ps.study_id
SET {', '.join(set_parts)}
WHERE ps.edf_uuid = '{uuid}';

UPDATE `psg_studies` 
SET `artifact_count` = {data['artifact_count']}, 
    `artifact_duration_minutes` = {data['artifact_duration_minutes']}
WHERE `edf_uuid` = '{uuid}';"""

		return sql

class SQLGenerator:
	def __init__(self, config=None):
		self.config = config or CONFIG
		self.sleep_analyzer = SleepAnalyzer(self.config)
		self.lock = Lock()

	def process_file(self, edf_path):
		try:
			print(f"Processing: {os.path.basename(edf_path)}")

			raw = self.sleep_analyzer.load_edf(edf_path)
			if not raw:
				return None

			uuid = self.sleep_analyzer.extract_uuid(edf_path)
			if not uuid:
				print(f"UUID not found: {edf_path}")
				return None

			sql = self.sleep_analyzer.generate_sql(edf_path, uuid)
			return sql

		except Exception as e:
			print(f"Error processing {edf_path}: {e}")
			return None

	def generate_sql_files(self, input_dir, output_dir, max_workers=10):
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		edf_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.edf')]

		with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
			results = list(executor.map(self.process_file, edf_files))

		valid_sql = [sql for sql in results if sql]

		for i, sql in enumerate(valid_sql):
			output_file = os.path.join(output_dir, f"sleep_stats_{i + 1}.sql")
			with open(output_file, 'w', encoding='utf-8') as f:
				f.write(sql)

		print(f"Generated {len(valid_sql)} SQL files in {output_dir}")

	def combine_sql_files(self, folder_path, output_file="combined_updates.sql"):
		"""Объединяет все SQL файлы в один компактный файл"""
		import glob

		sql_files = glob.glob(os.path.join(folder_path, "*.sql"))

		if not sql_files:
			print("SQL файлы не найдены!")
			return False

		with open(output_file, 'w', encoding='utf-8') as outfile:
			for sql_file in sorted(sql_files):
				with open(sql_file, 'r', encoding='utf-8') as infile:
					content = infile.read()
					lines = content.split('\n')
					sql_lines = []
					for line in lines:
						line = line.strip()
						if line and not line.startswith('--') and not line.startswith('#'):
							sql_lines.append(line)

					if sql_lines:
						outfile.write('\n'.join(sql_lines))
						outfile.write(';\n\n')

		print(f"Объединено {len(sql_files)} файлов в {output_file}")
		return True

def main():
	generator = SQLGenerator()
	generator.generate_sql_files('EDF', 'sql_output')
	combine = input("\nОбъединить все файлы в один? (y/n): ").lower().strip()
	if combine in ['y', 'yes', 'д', 'да']:
		generator.combine_sql_files('sql_output', 'combined_sleep_updates.sql')
	else:
		print("Объединение отменено.")

if __name__ == "__main__":
	main()