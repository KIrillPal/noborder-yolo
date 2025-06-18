import json
import os
import shutil

def process_images_folder(folder_path, speed_json_path, threshold_T, output_folder1, output_folder2):
    """
    Обрабатывает папку с изображениями, разделяя их на две папки по условию сдвига > T
    
    :param folder_path: путь к папке с изображениями
    :param speed_json_path: путь к файлу speed.json
    :param threshold_T: пороговое значение сдвига
    :param output_folder1: папка для изображений с сдвигом <= T
    :param output_folder2: папка для изображений с сдвигом > T
    """
    # Загрузка данных из speed.json
    with open(speed_json_path, 'r') as f:
        speed_data = json.load(f)
    
    # Сортировка путей к изображениям (если нужно)
    sorted_images = sorted(speed_data.keys())
    
    # Вычисление сдвигов для каждого изображения
    shifts = {}
    cumulative_shift = 0
    
    for img_path in sorted_images:
        speed = speed_data[img_path]
        cumulative_shift += speed
        shifts[img_path] = cumulative_shift
    
    # Создание выходных папок, если их нет
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)
    
    # Копирование изображений в соответствующие папки
    for img_path, shift in shifts.items():
        # Получаем только имя файла из полного пути
        img_name = os.path.basename(img_path)
        
        # Определяем в какую папку копировать
        if shift > threshold_T:
            dest_folder = output_folder2
        else:
            dest_folder = output_folder1
        
        # Полный путь к исходному изображению
        src_img_path = os.path.join(folder_path, img_name)
        
        # Проверяем существует ли исходный файл
        if not os.path.exists(src_img_path):
            print(f"Предупреждение: файл {src_img_path} не существует, пропускаем")
            continue
        
        # Копируем файл
        shutil.move(src_img_path, os.path.join(dest_folder, img_name))
    
    print(f"Готово! Изображения разделены между {output_folder1} и {output_folder2}")

# Пример использования
if __name__ == "__main__":
    # Параметры (замените на свои)
    images_folder = "/alpha/projects/wastie/code/kondrashov/delta/data/trainval/imgs"
    speed_json = "/alpha/projects/wastie/code/kondrashov/delta/data/trainval/speed.json"
    T = 150000  # Пороговое значение сдвига
    output1 = "/alpha/projects/wastie/code/kondrashov/delta/data/trainval/imgs1"  # Папка для изображений с shift <= T
    output2 = "/alpha/projects/wastie/code/kondrashov/delta/data/trainval/imgs2"  # Папка для изображений с shift > T
    
    process_images_folder(images_folder, speed_json, T, output1, output2)