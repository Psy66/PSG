import struct
import pandas as pd
from tabulate import tabulate

def read_edf_header(filename):
	"""
	Чтение заголовка EDF файла и извлечение информации о каналах
	"""
	with open(filename, 'rb') as f:
		# Чтение основного заголовка (256 байт)
		header = f.read(256)

		# Парсинг основной информации
		version = header[0:8].decode('ascii', errors='ignore').strip()
		patient_id = header[8:88].decode('ascii', errors='ignore').strip()
		recording_id = header[88:168].decode('ascii', errors='ignore').strip()
		start_date = header[168:176].decode('ascii', errors='ignore').strip()
		start_time = header[176:184].decode('ascii', errors='ignore').strip()
		header_bytes = int(header[184:192].decode('ascii', errors='ignore').strip())
		reserved = header[192:236].decode('ascii', errors='ignore').strip()
		num_records = int(header[236:244].decode('ascii', errors='ignore').strip())
		record_duration = float(header[244:252].decode('ascii', errors='ignore').strip())
		num_channels = int(header[252:256].decode('ascii', errors='ignore').strip())

		print("=== ОСНОВНАЯ ИНФОРМАЦИЯ EDF ===")
		print(f"Версия: {version}")
		print(f"ID пациента: {patient_id}")
		print(f"ID записи: {recording_id}")
		print(f"Дата начала: {start_date}")
		print(f"Время начала: {start_time}")
		print(f"Размер заголовка: {header_bytes} байт")
		print(f"Количество записей: {num_records}")
		print(f"Длительность записи: {record_duration} сек")
		print(f"Количество каналов: {num_channels}")
		print()

		# Чтение информации о каналах
		channels = []

		# Чтение названий каналов (16 символов на канал)
		f.seek(256)
		channel_names = []
		for i in range(num_channels):
			name = f.read(16).decode('ascii', errors='ignore').strip()
			channel_names.append(name)

		# Чтение типов датчиков (80 символов на канал)
		f.seek(256 + 16 * num_channels)
		transducer_types = []
		for i in range(num_channels):
			transducer = f.read(80).decode('ascii', errors='ignore').strip()
			transducer_types.append(transducer)

		# Чтение физических размерностей (8 символов на канал)
		f.seek(256 + 16 * num_channels + 80 * num_channels)
		physical_dimensions = []
		for i in range(num_channels):
			dimension = f.read(8).decode('ascii', errors='ignore').strip()
			physical_dimensions.append(dimension)

		# Чтение физических минимумов (8 символов на канал)
		f.seek(256 + 16 * num_channels + 80 * num_channels + 8 * num_channels)
		physical_minima = []
		for i in range(num_channels):
			pmin = f.read(8).decode('ascii', errors='ignore').strip()
			try:
				physical_minima.append(float(pmin))
			except:
				physical_minima.append(0.0)

		# Чтение физических максимумов (8 символов на канал)
		f.seek(256 + 16 * num_channels + 80 * num_channels + 16 * num_channels)
		physical_maxima = []
		for i in range(num_channels):
			pmax = f.read(8).decode('ascii', errors='ignore').strip()
			try:
				physical_maxima.append(float(pmax))
			except:
				physical_maxima.append(0.0)

		# Чтение цифровых минимумов (8 символов на канал)
		f.seek(256 + 16 * num_channels + 80 * num_channels + 24 * num_channels)
		digital_minima = []
		for i in range(num_channels):
			dmin = f.read(8).decode('ascii', errors='ignore').strip()
			try:
				digital_minima.append(int(dmin))
			except:
				digital_minima.append(-32768)

		# Чтение цифровых максимумов (8 символов на канал)
		f.seek(256 + 16 * num_channels + 80 * num_channels + 32 * num_channels)
		digital_maxima = []
		for i in range(num_channels):
			dmax = f.read(8).decode('ascii', errors='ignore').strip()
			try:
				digital_maxima.append(int(dmax))
			except:
				digital_maxima.append(32767)

		# Чтение предфильтров (80 символов на канал)
		f.seek(256 + 16 * num_channels + 80 * num_channels + 40 * num_channels)
		prefiltering = []
		for i in range(num_channels):
			prefilter = f.read(80).decode('ascii', errors='ignore').strip()
			prefiltering.append(prefilter)

		# Чтение количества samples на запись (8 символов на канал)
		f.seek(256 + 16 * num_channels + 80 * num_channels + 120 * num_channels)
		samples_per_record = []
		for i in range(num_channels):
			samples = f.read(8).decode('ascii', errors='ignore').strip()
			try:
				samples_per_record.append(int(samples))
			except:
				samples_per_record.append(0)

		# Формирование информации о каналах
		for i in range(num_channels):
			channel_info = {
				'index': i + 1,
				'name': channel_names[i],
				'transducer': transducer_types[i],
				'physical_dimension': physical_dimensions[i],
				'physical_min': physical_minima[i],
				'physical_max': physical_maxima[i],
				'digital_min': digital_minima[i],
				'digital_max': digital_maxima[i],
				'prefiltering': prefiltering[i],
				'samples_per_record': samples_per_record[i],
				'sampling_rate': samples_per_record[i] / record_duration,
				'type': get_channel_type(channel_names[i])
			}
			channels.append(channel_info)

		return channels, {
			'version': version,
			'patient_id': patient_id,
			'recording_id': recording_id,
			'start_date': start_date,
			'start_time': start_time,
			'header_bytes': header_bytes,
			'num_records': num_records,
			'record_duration': record_duration,
			'num_channels': num_channels
		}

def get_channel_type(channel_name):
	"""Определяет тип канала по его названию"""
	name = channel_name.upper()
	if 'EEG' in name:
		return 'EEG'
	elif 'EOG' in name:
		return 'EOG'
	elif 'EMG' in name:
		return 'EMG'
	elif 'ECG' in name:
		return 'ECG'
	elif 'RESP' in name:
		return 'RESP'
	elif 'SPO2' in name or 'SAO2' in name:
		return 'SpO2'
	elif 'SOUND' in name:
		return 'Sound'
	elif 'POSITION' in name:
		return 'Position'
	elif 'PPG' in name:
		return 'PPG'
	elif 'ANNOTATION' in name:
		return 'Annotations'
	elif 'PULSA' in name or 'CHASTOTA' in name:
		return 'Pulse'
	else:
		return 'Other'

def create_channels_table(channels):
	"""Создает красивую таблицу с информацией о каналах"""
	table_data = []

	for channel in channels:
		table_data.append([
			channel['index'],
			channel['name'],
			channel['type'],
			channel['physical_dimension'],
			f"{channel['sampling_rate']:.0f} Hz",
			channel['samples_per_record'],
			f"[{channel['physical_min']:.1f}, {channel['physical_max']:.1f}]",
			channel['prefiltering'][:30] + '...' if len(channel['prefiltering']) > 30 else channel['prefiltering']
		])

	headers = [
		'№', 'Название', 'Тип', 'Ед.изм.',
		'Частота', 'Samples/rec', 'Физ.диапазон', 'Предфильтрация'
	]

	return tabulate(table_data, headers=headers, tablefmt='grid', stralign='left')

def print_channel_statistics(channels):
	"""Выводит статистику по каналам"""
	print("\n=== СТАТИСТИКА ПО КАНАЛАМ ===")

	# Группировка по типам
	type_stats = {}
	sampling_stats = {}

	for channel in channels:
		chan_type = channel['type']
		sampling_rate = channel['sampling_rate']

		if chan_type not in type_stats:
			type_stats[chan_type] = 0
		type_stats[chan_type] += 1

		if chan_type not in sampling_stats:
			sampling_stats[chan_type] = []
		sampling_stats[chan_type].append(sampling_rate)

	# Вывод статистики по типам
	print("Количество каналов по типам:")
	for chan_type, count in sorted(type_stats.items()):
		avg_sampling = sum(sampling_stats[chan_type]) / len(sampling_stats[chan_type])
		print(f"  {chan_type}: {count} каналов (средняя частота: {avg_sampling:.0f} Гц)")

	# Общая статистика
	total_samples_per_record = sum(channel['samples_per_record'] for channel in channels)
	print(f"\nОбщее количество samples на запись: {total_samples_per_record}")
	print(f"Общий размер данных на запись: {total_samples_per_record * 2} байт")

def save_channels_to_csv(channels, filename):
	"""Сохраняет информацию о каналах в CSV файл"""
	df_data = []
	for channel in channels:
		df_data.append({
			'Index': channel['index'],
			'Name': channel['name'],
			'Type': channel['type'],
			'Physical_Dimension': channel['physical_dimension'],
			'Sampling_Rate_Hz': channel['sampling_rate'],
			'Samples_Per_Record': channel['samples_per_record'],
			'Physical_Min': channel['physical_min'],
			'Physical_Max': channel['physical_max'],
			'Digital_Min': channel['digital_min'],
			'Digital_Max': channel['digital_max'],
			'Prefiltering': channel['prefiltering'],
			'Transducer': channel['transducer']
		})

	df = pd.DataFrame(df_data)
	csv_filename = filename.replace('.edf', '_channels.csv')
	df.to_csv(csv_filename, index=False, encoding='utf-8')
	print(f"\nИнформация о каналах сохранена в: {csv_filename}")

def read_annotations_simple(filename, channels, header_info):
	"""Простое чтение аннотаций - базовая версия"""
	annotation_index = None
	for i, channel in enumerate(channels):
		if 'annotation' in channel['name'].lower():
			annotation_index = i
			break

	if annotation_index is None:
		print("Канал аннотаций не найден!")
		return []

	print(f"\n=== ЧТЕНИЕ АННОТАЦИЙ (канал {annotation_index + 1}) ===")

	with open(filename, 'rb') as f:
		header_size = header_info['header_bytes']
		annotation_samples = channels[annotation_index]['samples_per_record']

		# Рассчитываем размер одной записи
		record_size_bytes = sum(channel['samples_per_record'] for channel in channels) * 2

		# Читаем первые несколько записей для анализа
		f.seek(header_size)
		first_record_bytes = f.read(min(record_size_bytes, 10000))

		print(f"Размер заголовка: {header_size} байт")
		print(f"Samples аннотаций на запись: {annotation_samples}")
		print(f"Размер одной записи: {record_size_bytes} байт")
		print(f"Прочитано байт для анализа: {len(first_record_bytes)}")

		# Пока просто возвращаем базовую информацию
		return [{
			'annotation_channel_index': annotation_index,
			'samples_per_record': annotation_samples,
			'record_size_bytes': record_size_bytes,
			'status': 'ready_for_reading'
		}]

# Основная программа
if __name__ == "__main__":
	filename = "test.edf"

	# Чтение заголовка и информации о каналах
	channels, header_info = read_edf_header(filename)

	# Создание и вывод таблицы каналов
	table = create_channels_table(channels)
	print("=== ТАБЛИЦА КАНАЛОВ ===")
	print(table)

	# Вывод статистики
	print_channel_statistics(channels)

	# Сохранение в CSV
	save_channels_to_csv(channels, filename)

	# Базовая информация об аннотациях
	annotations_info = read_annotations_simple(filename, channels, header_info)

	print(f"\n=== ИНФОРМАЦИЯ ДЛЯ ЧТЕНИЯ АННОТАЦИЙ ===")
	if annotations_info:
		info = annotations_info[0]
		print(f"Канал аннотаций: {info['annotation_channel_index'] + 1}")
		print(f"Samples аннотаций на запись: {info['samples_per_record']}")
		print(f"Размер данных аннотаций на запись: {info['samples_per_record'] * 2} байт")
		print(f"Общий размер одной записи: {info['record_size_bytes']} байт")