import mne
import os
import glob
import re

def show_annotations_with_encodings(edf_folder="EDF"):
	"""
	Пробует разные кодировки и выводит аннотации из скобок
	"""
	# Находим все EDF файлы
	edf_files = glob.glob(os.path.join(edf_folder, "*.edf"))

	if not edf_files:
		print(f"В папке '{edf_folder}' нет EDF файлов")
		return

	print(f"Найдено файлов: {len(edf_files)}")
	print("=" * 80)

	# Кодировки для проб
	encodings = ['utf-8', 'latin1', 'cp1251', 'iso-8859-1', None]

	all_keys = set()
	processed = 0
	failed = 0

	for i, edf_path in enumerate(edf_files, 1):
		filename = os.path.basename(edf_path)
		print(f"\n{i}. {filename}")

		file_loaded = False

		# Пробуем разные кодировки
		for enc in encodings:
			try:
				if enc:
					raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False, encoding=enc)
				else:
					raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

				# Если дошли сюда - файл загрузился
				file_loaded = True
				print(f"   ✅ Загружен с кодировкой: {enc if enc else 'auto'}")

				# Получаем аннотации
				if hasattr(raw, 'annotations') and raw.annotations:
					annotations = raw.annotations.description

					# Собираем ключи из скобок
					file_keys = set()
					for ann in annotations:
						ann_str = str(ann)
						# Ищем текст в скобках
						match = re.search(r'\(([^)]+)\)', ann_str)
						if match:
							key = match.group(1)
							all_keys.add(key)
							file_keys.add(key)

					print(f"   Аннотаций: {len(annotations)}")
					print(f"   Уникальных ключей в файле: {len(file_keys)}")

					# Покажем несколько примеров
					if file_keys:
						examples = list(file_keys)[:5]
						print(f"   Примеры: {', '.join(examples)}")
				else:
					print("   Аннотаций нет")

				# Выходим из цикла кодировок
				break

			except Exception as e:
				# Пробуем следующую кодировку
				continue

		if not file_loaded:
			failed += 1
			print(f"   ❌ Не удалось загрузить ни с одной кодировкой")
		else:
			processed += 1

	# Выводим итог
	print("\n" + "=" * 80)
	print(f"ИТОГ:")
	print(f"   Успешно загружено: {processed}")
	print(f"   Не загружено: {failed}")
	print(f"   Всего уникальных ключей: {len(all_keys)}")
	print("=" * 80)

	# Сортируем и выводим все ключи
	for i, key in enumerate(sorted(all_keys), 1):
		print(f"{i:3}. {key}")

if __name__ == "__main__":
	show_annotations_with_encodings("EDF2")