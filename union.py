import os
import glob

def quick_combine_sql_files(folder_path, output_file="combined_updates.sql"):
	"""Быстрое объединение SQL файлов"""

	sql_files = glob.glob(os.path.join(folder_path, "*.sql"))

	if not sql_files:
		print("SQL файлы не найдены!")
		return

	with open(output_file, 'w', encoding='utf-8') as outfile:
		outfile.write("-- Объединенные SQL запросы для обновления статистики сна\n\n")

		for sql_file in sorted(sql_files):
			filename = os.path.basename(sql_file)
			outfile.write(f"-- Файл: {filename}\n")

			with open(sql_file, 'r', encoding='utf-8') as infile:
				outfile.write(infile.read())
				outfile.write("\n\n")

	print(f"Объединено {len(sql_files)} файлов в {output_file}")

# Использование:
quick_combine_sql_files("sql_output")