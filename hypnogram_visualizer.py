import matplotlib

# Используем бэкенд который не пытается ничего показывать
matplotlib.use('Agg')  # ДОЛЖНО БЫТЬ ПЕРВОЙ СТРОКОЙ!
import matplotlib.pyplot as plt
import json

def generate_hypnogram():
	try:
		# Загрузка данных
		with open('hypnogram_compact.json', 'r', encoding='utf-8') as f:
			data = json.load(f)

		# Извлекаем последовательность стадий из нового формата
		stages = data['s']  # 's' вместо 'stages_sequence'
		total_epochs = data['e']  # 'e' вместо 'total_epochs'

		print(f"Обрабатывается {total_epochs} эпох...")

		# Создаем график
		fig, ax = plt.subplots(figsize=(15, 5))

		# Цвета и высоты (обновляем для числовых кодов)
		colors = {'W': '#FF6B6B', '1': '#4ECDC4', '2': '#45B7D1', '3': '#96CEB4', 'R': '#FFEAA7'}
		levels = {'W': 4, '1': 3, '2': 2, '3': 1, 'R': 0}
		labels = {'W': 'Бодрствование', '1': 'N1', '2': 'N2', '3': 'N3', 'R': 'REM'}

		# Рисуем
		for i, stage in enumerate(stages):
			ax.fill_between([i, i + 1], [levels[stage], levels[stage]],
			                color=colors[stage], alpha=0.8)

		# Настройки
		ax.set_yticks(list(levels.values()))
		ax.set_yticklabels([labels[stage] for stage in levels.keys()])
		ax.set_ylabel('Стадия сна')
		ax.set_xlabel('Эпохи (30 секунд)')

		total_hours = (total_epochs * 30) / 3600
		ax.set_title(f'Гипнограмма сна - {total_epochs} эпох ({total_hours:.1f} часов)')
		ax.grid(True, alpha=0.3)

		# Сохраняем
		plt.tight_layout()
		plt.savefig('hypnogram_basic.png', dpi=300, bbox_inches='tight')
		print("✅ Успешно! Файл: hypnogram_basic.png")

	except Exception as e:
		print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
	generate_hypnogram()