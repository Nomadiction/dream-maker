from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import cv2
import numpy as np
import io, os, uuid
from rembg import remove
import mediapipe as mp
import face_recognition
import logging
import math
from typing import Optional

# Настройка логов
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anonymize_service")

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUT_DIR = os.path.join(BASE_DIR, 'out')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory=OUT_DIR), name="static")


class PersonAnonymizer:
    """Класс для продвинутой анонимизации изображений с удалением фона и применением фильтров"""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """Удаление фона с помощью rembg"""
        output_image = remove(image)
        return output_image
    
    def apply_cartoon_filter(self, image):
        """Мультяшный фильтр"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Мультяшный эффект
        color = cv2.bilateralFilter(image, 9, 300, 300)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10
        )
        
        # Объединение цветного изображения с краями
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon
    
    def apply_oil_painting_filter(self, image):
        """Эффект масляной живописи"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        return cv2.xphoto.oilPainting(image, 7, 1)
    
    def apply_stylization(self, image):
        """Художественная стилизация"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        return cv2.stylization(image, sigma_s=60, sigma_r=0.45)
    
    def soft_anonymize_faces(self, image):
        """Мягкая анонимизация лиц - сохраняет черты, но делает менее узнаваемыми"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Защита от выхода за границы
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                # Извлекаем область лица
                if width > 0 and height > 0:
                    face_region = image[y:y+height, x:x+width]
                    
                    # 1. Лёгкое размытие (сохраняет основные черты)
                    face_region = cv2.bilateralFilter(face_region, 15, 80, 80)
                    
                    # 2. Небольшое изменение цвета кожи
                    face_region = self.adjust_skin_tone(face_region)
                    
                    # 3. Лёгкое искажение пропорций
                    face_region = self.slightly_distort_face(face_region)
                    
                    # Возвращаем обработанную область
                    image[y:y+height, x:x+width] = face_region

        return image

    def apply_wax_figure_effect(self, image):
        """Эффект восковой фигуры - УЛУЧШЕННЫЙ: гладкая кожа, сохраняя черты лица"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1) УМЕРЕННОЕ сглаживание - сохраняем черты лица
        smooth = cv2.bilateralFilter(image, 9, 75, 75)
        smooth = cv2.bilateralFilter(smooth, 7, 60, 60)
        
        # 2) Стилизация вместо размытия - сохраняет структуру
        smooth = cv2.stylization(smooth, sigma_s=70, sigma_r=0.45)
        
        # 3) УМЕРЕННОЕ увеличение яркости и насыщенности
        hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        v = cv2.add(v, 30)  # умеренная яркость
        s = cv2.add(s, 20)  # умеренная насыщенность
        
        hsv = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 4) ЛЕГКИЕ блики - более естественные
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, highlight_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        highlight = cv2.GaussianBlur(highlight_mask, (15, 15), 0)
        highlight = cv2.cvtColor(highlight, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(result, 0.85, highlight, 0.15, 0)
        
        # 5) УБИРАЕМ постеризацию - она делает изображение темным!
        
        # 6) УБИРАЕМ detailEnhance - он создает контуры
        # result остается после бликов
        
        return result
    
    def apply_toon_effect(self, image):
        """Toon эффект - УЛУЧШЕННЫЙ: светлый мультяшный стиль БЕЗ черных контуров"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1) Мягкое сглаживание - сохраняем детали
        smooth = cv2.bilateralFilter(image, 7, 60, 60)
        
        # 2) Стилизация для художественного вида
        smooth = cv2.stylization(smooth, sigma_s=100, sigma_r=0.50)
        
        # 3) УБИРАЕМ постеризацию - она делает изображение темным!
        # Вместо этого просто используем результат стилизации
        
        # 4) УБИРАЕМ detailEnhance - он создает контуры
        result = smooth
        
        # 5) МАКСИМАЛЬНО ЯРКИЕ цвета
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)  # Умеренная насыщенность
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.5 + 50, 0, 255)  # СУПЕР ярко!
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # 6) Увеличение яркости и контраста
        result = cv2.convertScaleAbs(result, alpha=1.1, beta=25)
        
        return result
    
    def apply_clay_sculpture_effect(self, image):
        """Эффект глиняной скульптуры - УЛУЧШЕННЫЙ: светлая матовая текстура"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1) Преобразование в СВЕТЛЫЕ глиняные тона
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Легкий сдвиг к теплым тонам
        hsv[:, :, 0] = (hsv[:, :, 0] + 10) % 180
        # Легко уменьшаем насыщенность (матовость)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.75, 0, 255)
        # МАКСИМАЛЬНО увеличиваем яркость для светлого вида
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.6 + 55, 0, 255)
        
        clay_color = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # 2) Мягкое сглаживание - сохраняем черты
        smooth = cv2.bilateralFilter(clay_color, 7, 60, 60)
        
        # 3) Стилизация для скульптурного вида
        smooth = cv2.stylization(smooth, sigma_s=90, sigma_r=0.45)
        
        # 4) ОЧЕНЬ ЛЕГКАЯ текстура глины
        noise = np.random.randint(-3, 3, smooth.shape, dtype=np.int16)
        result = np.clip(smooth.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 5) УБИРАЕМ постеризацию - она делает изображение темным!
        
        # 6) УБИРАЕМ detailEnhance - он создает контуры
        # result остается после текстуры
        
        return result
    
    def apply_comic_style_effect(self, image):
        """Комиксный стиль - УЛУЧШЕННЫЙ: очень яркие цвета БЕЗ толстых контуров"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1) Мягкое сглаживание
        smooth = cv2.bilateralFilter(image, 7, 65, 65)
        
        # 2) Стилизация для comic-look
        smooth = cv2.stylization(smooth, sigma_s=100, sigma_r=0.50)
        
        # 3) УБИРАЕМ постеризацию - она делает изображение темным!
        
        # 4) УБИРАЕМ detailEnhance - он создает контуры
        result = smooth
        
        # 5) МАКСИМАЛЬНО ЯРКИЕ цвета (комиксный стиль)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)  # Высокая насыщенность
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.55 + 50, 0, 255)  # СУПЕР яркость!
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # 6) Увеличение яркости и контраста
        result = cv2.convertScaleAbs(result, alpha=1.1, beta=22)
        
        return result
    
    def apply_cgi_character_effect(self, image):
        """3D CGI рендер - УЛУЧШЕННЫЙ: светлый digital art БЕЗ контурного эффекта"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1) Мягкое сглаживание - сохраняем детали
        smooth = cv2.bilateralFilter(image, 7, 60, 60)
        
        # 2) Стилизация для 3D-look
        smooth = cv2.stylization(smooth, sigma_s=100, sigma_r=0.50)
        
        # 3) МАКСИМАЛЬНОЕ увеличение насыщенности и яркости
        hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)  # Умеренная насыщенность
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.55 + 50, 0, 255)  # СУПЕР ярко!
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # 4) МЯГКИЕ блики для 3D-эффекта
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, highlight_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        highlight = cv2.GaussianBlur(highlight_mask, (15, 15), 0)
        highlight = cv2.cvtColor(highlight, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(result, 0.88, highlight, 0.12, 0)
        
        # 5) УБИРАЕМ постеризацию - она делает изображение темным!
        
        # 6) УБИРАЕМ detailEnhance - он создает контуры
        # result остается после постеризации
        
        return result
    
    def apply_pastel_painting_effect(self, image):
        """Пастельный портрет - УЛУЧШЕННЫЙ: светлые нежные тона, сохраняя четкость"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1) Легкое сглаживание - сохраняем черты
        soft = cv2.bilateralFilter(image, 7, 55, 55)
        
        # 2) Стилизация для живописного вида
        soft = cv2.stylization(soft, sigma_s=95, sigma_r=0.50)
        
        # 3) Пастельные тона с ВЫСОКОЙ яркостью
        hsv = cv2.cvtColor(soft, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.65, 0, 255)  # Уменьшаем насыщенность (пастель)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.6 + 55, 0, 255)  # СУПЕР светлые тона!
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # 4) ОЧЕНЬ ЛЕГКАЯ текстура холста
        noise = np.random.randint(-4, 4, result.shape, dtype=np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 5) ОЧЕНЬ легкое размытие для мягкости
        result = cv2.GaussianBlur(result, (3, 3), 0.5)
        
        # 6) УБИРАЕМ постеризацию - она делает изображение темным!
        
        # 7) УБИРАЕМ detailEnhance - он создает контуры
        # result остается после размытия
        
        return result
    
    def realistic_anonymize_faces(self, image):
        """Реалистичная анонимизация - минимальные изменения для сохранения фотореалистичности"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)

        if not results.detections:
            return image

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x = int(max(0, bbox.xmin) * w)
            y = int(max(0, bbox.ymin) * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            if bw <= 0 or bh <= 0:
                continue

            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            face = image[y:y+bh, x:x+bw]

            # 1) Очень лёгкий сдвиг оттенка кожи (сохраняет реалистичность)
            hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            # Создаём маску для кожи
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Очень лёгкий сдвиг оттенка только для кожи
            hsv_shifted = hsv.copy()
            hsv_shifted[:, :, 0] = (hsv[:, :, 0] + 2) % 180  # Ещё более минимальный сдвиг
            hsv_shifted[:, :, 1] = np.clip(hsv[:, :, 1] + 3, 0, 255)  # Очень лёгкое увеличение насыщенности
            hsv_shifted[:, :, 2] = np.clip(hsv[:, :, 2] + 2, 0, 255)  # Лёгкое увеличение яркости
            
            # Применяем изменения только к коже
            face_result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            face_shifted = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)
            face_result[skin_mask > 0] = face_shifted[skin_mask > 0]

            # 2) Очень лёгкое изменение пропорций (сохраняет узнаваемость)
            center = (bw // 2, bh // 2)
            angle = 0.5  # Ещё более минимальный поворот
            scale = 1.005  # Ещё более минимальное масштабирование
            M = cv2.getRotationMatrix2D(center, angle, scale)
            face_result = cv2.warpAffine(face_result, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            # 3) Очень лёгкое сглаживание для естественности (не блюр!)
            face_result = cv2.bilateralFilter(face_result, 3, 30, 30)

            # Вклеиваем обратно
            image[y:y+bh, x:x+bw] = face_result

        return image

    def identity_shift_faces(self, image):
        """Анонимизация без блюра: лёгкая смена идентичности через цветовой сдвиг и геом. трансформации.
        Сохраняет читаемость черт, но делает лицо менее узнаваемым.
        """
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)

        if not results.detections:
            return image

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x = int(max(0, bbox.xmin) * w)
            y = int(max(0, bbox.ymin) * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            if bw <= 0 or bh <= 0:
                continue

            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            face = image[y:y+bh, x:x+bw]

            # 1) Сдвиг оттенка/температуры для всего лица (с сохранением деталей)
            hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + 12) % 180
            hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 10)
            recolored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # 2) Лёгкая геометрическая трансформация (поворот+масштаб вокруг центра ROI)
            center = (bw // 2, bh // 2)
            angle = 3.0  # небольшой угол
            scale = 1.03  # лёгкое масштабирование
            M = cv2.getRotationMatrix2D(center, angle, scale)
            warped = cv2.warpAffine(recolored, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            # 3) Edge-preserving фильтр, чтобы сохранить очертания без блюра
            shifted = cv2.edgePreservingFilter(warped, flags=1, sigma_s=30, sigma_r=0.35)

            # Вклеиваем обратно
            image[y:y+bh, x:x+bw] = shifted

        return image
    
    def adjust_skin_tone(self, face_region):
        """Изменение тона кожи для анонимизации"""
        # Конвертируем в HSV для работы с цветом
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        
        # Создаём маску для кожи (диапазон оттенков кожи)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Применяем небольшой сдвиг оттенка
        hsv[:, :, 0] = (hsv[:, :, 0] + 5) % 180  # Сдвигаем оттенок на 5 градусов
        
        # Конвертируем обратно в BGR
        adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Применяем изменения только к областям кожи
        face_region[skin_mask > 0] = adjusted[skin_mask > 0]
        
        return face_region
    
    def slightly_distort_face(self, face_region):
        """Лёгкое искажение пропорций лица"""
        h, w = face_region.shape[:2]
        
        # Создаём небольшую деформацию
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                # Небольшое искажение по синусоиде
                distortion_x = 2 * np.sin(i * 0.02) * np.cos(j * 0.02)
                distortion_y = 1.5 * np.cos(i * 0.03) * np.sin(j * 0.03)
                
                map_x[i, j] = j + distortion_x
                map_y[i, j] = i + distortion_y
        
        # Применяем деформацию
        distorted = cv2.remap(face_region, map_x, map_y, cv2.INTER_LINEAR)
        
        return distorted
    
    def blur_faces(self, image):
        """Размытие лиц для дополнительной анонимизации (старый метод)"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Защита от выхода за границы
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                # Размытие области лица
                if width > 0 and height > 0:
                    face_roi = image[y:y+height, x:x+width]
                    blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                    image[y:y+height, x:x+width] = blurred_face
        
        return image
    
    def change_hair_color(self, image, color=(128, 0, 128)):
        """Изменение цвета волос (базовый метод)"""
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Конвертация в HSV для более легкого выделения волос
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Диапазон для темных цветов (обычно волосы)
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 80])
        
        mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Применяем цвет к маске
        colored_hair = image.copy()
        colored_hair[mask > 0] = color
        
        return colored_hair
    
    def composite_with_artistic_background(self, foreground, background_style="gradient"):
        """Создание художественного фона"""
        if isinstance(foreground, Image.Image):
            # Конвертируем PIL Image в numpy array
            foreground_array = np.array(foreground)
            if foreground_array.shape[2] == 4:  # RGBA
                # Разделяем на RGB и альфа-канал
                foreground_rgb = cv2.cvtColor(foreground_array[:, :, :3], cv2.COLOR_RGB2BGR)
                mask = foreground_array[:, :, 3] / 255.0
            else:  # RGB
                foreground_rgb = cv2.cvtColor(foreground_array, cv2.COLOR_RGB2BGR)
                mask = np.ones((foreground_array.shape[0], foreground_array.shape[1]))
            foreground = foreground_rgb
        else:
            if foreground.shape[2] == 4:
                mask = foreground[:, :, 3] / 255.0
                foreground = foreground[:, :, :3]
            else:
                mask = np.ones((foreground.shape[0], foreground.shape[1]))
        
        h, w = foreground.shape[:2]
        
        if background_style == "gradient":
            # Градиентный фон
            background = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(h):
                color_val = int(255 * i / h)
                background[i, :] = [color_val, color_val//2, 255-color_val]
        
        elif background_style == "abstract":
            # Абстрактный фон
            background = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            background = cv2.GaussianBlur(background, (51, 51), 0)
        
        else:
            # Простой белый фон
            background = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Наложение на фон
        result = background.copy()
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - mask) + foreground[:, :, c] * mask
        
        return result.astype(np.uint8)
    
    def process_image(self, image: Image.Image, style="realistic", 
                     remove_bg=True, blur_face=True, 
                     background_style="gradient", anonymize_mode: str = "realistic") -> np.ndarray:
        """Основной метод обработки с полной анонимизацией"""
        
        # Шаг 1: Удаление фона (опционально)
        if remove_bg:
            no_bg_image = self.remove_background(image)
        else:
            no_bg_image = image
        
        # Шаг 2: Применение фильтров (только для нереалистичных стилей)
        if style == "realistic":
            # Для реалистичного стиля не применяем художественные фильтры
            filtered_image = no_bg_image
        elif style == "cartoon":
            filtered_image = self.apply_cartoon_filter(no_bg_image)
        elif style == "oil_painting":
            filtered_image = self.apply_oil_painting_filter(no_bg_image)
        elif style == "stylization":
            filtered_image = self.apply_stylization(no_bg_image)
        else:
            filtered_image = no_bg_image
        
        # Шаг 3: Анонимизация лиц (опционально)
        if anonymize_mode and anonymize_mode != "none":
            if anonymize_mode == "realistic":
                processed_face_image = self.realistic_anonymize_faces(filtered_image)
            elif anonymize_mode == "soft":
                processed_face_image = self.soft_anonymize_faces(filtered_image)
            elif anonymize_mode == "identity_shift":
                processed_face_image = self.identity_shift_faces(filtered_image)
            else:
                processed_face_image = filtered_image
        else:
            processed_face_image = filtered_image
        
        # Шаг 4: Создание фона
        if remove_bg:
            if style == "realistic":
                # Для реалистичного стиля используем простой белый фон
                final_image = self.composite_with_artistic_background(
                    processed_face_image, "white"
                )
            else:
                final_image = self.composite_with_artistic_background(
                    processed_face_image, background_style
                )
        else:
            final_image = processed_face_image
        
        return final_image
    
    def apply_advanced_anonymization(self, image: Image.Image) -> np.ndarray:
        """Продвинутая анонимизация - УЛУЧШЕННАЯ: светлая стилизация БЕЗ потери черт"""
        
        # Удаляем фон
        no_bg = self.remove_background(image)
        
        if isinstance(no_bg, Image.Image):
            img_array = cv2.cvtColor(np.array(no_bg), cv2.COLOR_RGB2BGR)
        else:
            img_array = no_bg
        
        # Легкое сглаживание - сохраняем детали
        smooth = cv2.bilateralFilter(img_array, 7, 60, 60)
        
        # Стилизация для digital art вида
        smooth = cv2.stylization(smooth, sigma_s=100, sigma_r=0.50)
        
        # ОЧЕНЬ ЛЕГКОЕ искажение - едва заметное
        h, w = smooth.shape[:2]
        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)
        
        # Создаем МИНИМАЛЬНОЕ искажение
        for i in range(h):
            for j in range(w):
                map_x[i, j] = j + 1.5 * np.sin(2 * np.pi * i / 140)  # ЕЩЁ меньше
                map_y[i, j] = i + 1.5 * np.cos(2 * np.pi * j / 140)
        
        distorted = cv2.remap(smooth, map_x, map_y, cv2.INTER_LINEAR)
        
        # СУПЕР ЯРКОЕ изменение цветовой схемы
        hsv = cv2.cvtColor(distorted, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + 18) % 180  # умеренный сдвиг hue
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)  # умеренная насыщенность
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.6 + 55, 0, 255)  # СУПЕР ярко!
        color_shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # УБИРАЕМ постеризацию - она делает изображение темным!
        
        # УБИРАЕМ detailEnhance - он создает контуры
        result = color_shifted
        
        # Финальное художественное оформление с БЕЛЫМ фоном
        final = self.composite_with_artistic_background(result, "white")
        
        return final


# Глобальный экземпляр анонимайзера
anonymizer = PersonAnonymizer()


def photo_to_3d_art(img: Image.Image) -> Image.Image:
    """
    Превращает реальное фото в СВЕТЛЫЙ фотореалистичный 3D-рендер (CGI персонажа).
    Использует K-means кластеризацию для профессионального результата.
    Для Sora 2 - выглядит как высококачественный 3D-рендер, а не реальный человек.
    """
    # Конвертируем PIL Image в numpy array для cv2
    img_array = np.array(img.convert("RGB"))
    # cv2 использует BGR, конвертируем из RGB
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Шаг 1: Мягкое сглаживание кожи (убираем текстуру)
    img_gb = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    
    # Легкое median blur
    img_mb = cv2.medianBlur(img_gb, 3)
    
    # Bilateral filter - сглаживает детали, но сохраняет резкие края
    img_bf = cv2.bilateralFilter(img_mb, 7, 60, 60)
    
    # Шаг 2: K-means кластеризация для умного упрощения цветов
    # Reshape the image для K-means
    img_reshaped = img_bf.reshape((-1, 3))
    # Convert to np.float32
    img_reshaped = np.float32(img_reshaped)
    
    # Set the K-means criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Set the amount of K (colors) - БОЛЬШЕ цветов = СВЕТЛЕЕ
    K = 24  # УВЕЛИЧИЛИ для более светлого изображения
    
    # Apply K-means
    _, labels, centers = cv2.kmeans(img_reshaped, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to np.uint8
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    # Reshape it back to an image
    img_kmeans = res.reshape((img_bf.shape))
    
    # Шаг 3: СУПЕР ЯРКАЯ цветокоррекция для CGI-вида
    img_hsv = cv2.cvtColor(img_kmeans, cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * 1.2, 0, 255)  # Умеренная насыщенность
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * 1.6 + 55, 0, 255)  # СУПЕР ЯРКО!
    img_hsv = img_hsv.astype(np.uint8)
    img_saturated = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    
    # Увеличение контраста
    img_contrast = cv2.convertScaleAbs(img_saturated, alpha=1.12, beta=22)
    
    # Шаг 4: УБИРАЕМ detailEnhance - он создает контуры!
    img_final = img_contrast
    
    # Convert from BGR back to RGB
    img_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
    
    # Конвертируем обратно в PIL Image
    result = Image.fromarray(img_rgb)
    
    return result

@app.get('/')
async def root():
    """Информация об API"""
    return {
        "message": "Dream Maker - Photo Anonymization & Stylization API",
        "description": "Converts real photos to artistic styles with anonymization for Sora AI",
        "endpoints": {
            "/stylize": "Original 3D art style (K-means based)",
            "/anonymize": "Advanced anonymization with filters (cartoon, oil_painting, stylization)",
            "/anonymize-advanced": "Maximum anonymization with distortions",
            "/remove-background": "Simple background removal"
        },
        "usage": "POST to any endpoint with form-data: image (file)"
    }

@app.post('/stylize')
async def stylize(image: UploadFile = File(...)):
    if not image:
        raise HTTPException(status_code=400, detail="no image")
    
    # Читаем изображение
    contents = await image.read()
    img = Image.open(io.BytesIO(contents))
    
    # Применяем конвертацию в 3D-арт (реальное фото → CGI персонаж)
    art_3d = photo_to_3d_art(img)
    
    # Сохраняем результат
    filename = f"{uuid.uuid4().hex}_3d_art.png"
    path = os.path.join(OUT_DIR, filename)
    art_3d.save(path, format='PNG')
    
    # Конвертируем путь в формат с прямыми слешами для корректного отображения
    path_normalized = path.replace('\\', '/')
    
    # Возвращаем относительный URL и путь для локальной разработки
    return {
        "stylized_url": f"http://localhost:8000/static/{filename}",
        "local_path": path_normalized
    }

@app.get('/static/{filename}')
async def static_file(filename: str):
    file_path = os.path.join(OUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.post('/anonymize')
async def anonymize(
    image: UploadFile = File(...),
    remove_bg: bool = Query(default=True, description="Remove background"),
    anonymize_mode: str = Query(default="realistic", description="Anonymization: realistic, soft, identity_shift")
):
    """
    Реалистичная анонимизация - сохраняет фотореалистичность с минимальными изменениями.
    
    Параметры:
    - remove_bg: удалить фон (True/False)
    - anonymize_mode: realistic (реалистичная), soft (мягкая), identity_shift (смена идентичности)
    """
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    try:
        # Читаем изображение
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        
        # Применяем анонимизацию
        result_array = anonymizer.process_image(
            img, 
            style="realistic",
            remove_bg=remove_bg,
            blur_face=True,
            background_style="white",
            anonymize_mode=anonymize_mode
        )
        
        # Конвертируем обратно в PIL Image
        result_rgb = cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB)
        result_img = Image.fromarray(result_rgb)
        
        # Сохраняем результат
        filename = f"{uuid.uuid4().hex}_anonymized.png"
        path = os.path.join(OUT_DIR, filename)
        result_img.save(path, format='PNG')
        
        # Конвертируем путь в формат с прямыми слешами
        path_normalized = path.replace('\\', '/')
        
        return {
            "message": "Realistic anonymization completed",
            "stylized_url": f"http://localhost:8000/static/{filename}",
            "local_path": path_normalized,
            "style": "realistic",
            "settings": {
                "remove_bg": remove_bg,
                "anonymize_mode": anonymize_mode,
                "background_style": "white"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post('/wax-figure')
async def wax_figure(
    image: UploadFile = File(...),
    remove_bg: bool = Query(default=True, description="Remove background")
):
    """
    Эффект восковой фигуры - гладкая кожа, идеальное освещение, как в музее мадам Тюссо.
    """
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    try:
        # Читаем изображение
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        
        # Удаляем фон если нужно
        if remove_bg:
            no_bg = anonymizer.remove_background(img)
        else:
            no_bg = img
        
        # Применяем эффект восковой фигуры
        result_array = anonymizer.apply_wax_figure_effect(no_bg)
        
        # Добавляем фон
        if remove_bg:
            result_array = anonymizer.composite_with_artistic_background(result_array, "white")
        
        # Конвертируем обратно в PIL Image
        result_rgb = cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_rgb)
        
        # Сохраняем результат
        filename = f"{uuid.uuid4().hex}_wax_figure.png"
        file_path = os.path.join(OUT_DIR, filename)
        result_image.save(file_path)
        
        # Нормализуем путь для Windows
        path_normalized = file_path.replace("\\", "/")
        
        return {
            "message": "Wax figure effect completed - smooth skin, perfect lighting",
            "stylized_url": f"http://localhost:8000/static/{filename}",
            "local_path": path_normalized,
            "style": "wax_figure",
            "settings": {
                "remove_bg": remove_bg,
                "effect": "wax_figure"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post('/toon')
async def toon(
    image: UploadFile = File(...),
    remove_bg: bool = Query(default=True, description="Remove background")
):
    """
    Toon эффект - графический персонаж с контурами и плоскими цветами.
    """
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        
        if remove_bg:
            no_bg = anonymizer.remove_background(img)
        else:
            no_bg = img
        
        result_array = anonymizer.apply_toon_effect(no_bg)
        
        if remove_bg:
            result_array = anonymizer.composite_with_artistic_background(result_array, "white")
        
        result_rgb = cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_rgb)
        
        filename = f"{uuid.uuid4().hex}_toon.png"
        file_path = os.path.join(OUT_DIR, filename)
        result_image.save(file_path)
        path_normalized = file_path.replace("\\", "/")
        
        return {
            "message": "Toon effect completed - graphic character with outlines",
            "stylized_url": f"http://localhost:8000/static/{filename}",
            "local_path": path_normalized,
            "style": "toon",
            "settings": {"remove_bg": remove_bg, "effect": "toon"}
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post('/clay-sculpture')
async def clay_sculpture(
    image: UploadFile = File(...),
    remove_bg: bool = Query(default=True, description="Remove background")
):
    """
    Эффект глиняной скульптуры - матовая поверхность, естественная текстура.
    """
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        
        if remove_bg:
            no_bg = anonymizer.remove_background(img)
        else:
            no_bg = img
        
        result_array = anonymizer.apply_clay_sculpture_effect(no_bg)
        
        if remove_bg:
            result_array = anonymizer.composite_with_artistic_background(result_array, "white")
        
        result_rgb = cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_rgb)
        
        filename = f"{uuid.uuid4().hex}_clay_sculpture.png"
        file_path = os.path.join(OUT_DIR, filename)
        result_image.save(file_path)
        path_normalized = file_path.replace("\\", "/")
        
        return {
            "message": "Clay sculpture effect completed - matte surface with natural texture",
            "stylized_url": f"http://localhost:8000/static/{filename}",
            "local_path": path_normalized,
            "style": "clay_sculpture",
            "settings": {"remove_bg": remove_bg, "effect": "clay_sculpture"}
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post('/comic-style')
async def comic_style(
    image: UploadFile = File(...),
    remove_bg: bool = Query(default=True, description="Remove background")
):
    """
    Комиксный стиль - яркие цвета, толстые чёрные контуры, как в комиксах.
    """
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        
        if remove_bg:
            no_bg = anonymizer.remove_background(img)
        else:
            no_bg = img
        
        result_array = anonymizer.apply_comic_style_effect(no_bg)
        
        if remove_bg:
            result_array = anonymizer.composite_with_artistic_background(result_array, "white")
        
        result_rgb = cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_rgb)
        
        filename = f"{uuid.uuid4().hex}_comic.png"
        file_path = os.path.join(OUT_DIR, filename)
        result_image.save(file_path)
        path_normalized = file_path.replace("\\", "/")
        
        return {
            "message": "Comic style effect completed - bright colors with thick black outlines",
            "stylized_url": f"http://localhost:8000/static/{filename}",
            "local_path": path_normalized,
            "style": "comic",
            "settings": {"remove_bg": remove_bg, "effect": "comic"}
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post('/cgi-character')
async def cgi_character(
    image: UploadFile = File(...),
    remove_bg: bool = Query(default=True, description="Remove background")
):
    """
    3D CGI рендер персонажа - как из современной игры или мультфильма Pixar.
    """
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        
        if remove_bg:
            no_bg = anonymizer.remove_background(img)
        else:
            no_bg = img
        
        result_array = anonymizer.apply_cgi_character_effect(no_bg)
        
        if remove_bg:
            result_array = anonymizer.composite_with_artistic_background(result_array, "white")
        
        result_rgb = cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_rgb)
        
        filename = f"{uuid.uuid4().hex}_cgi.png"
        file_path = os.path.join(OUT_DIR, filename)
        result_image.save(file_path)
        path_normalized = file_path.replace("\\", "/")
        
        return {
            "message": "CGI character effect completed - 3D rendered game/movie character",
            "stylized_url": f"http://localhost:8000/static/{filename}",
            "local_path": path_normalized,
            "style": "cgi",
            "settings": {"remove_bg": remove_bg, "effect": "cgi"}
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post('/pastel-painting')
async def pastel_painting(
    image: UploadFile = File(...),
    remove_bg: bool = Query(default=True, description="Remove background")
):
    """
    Пастельный портрет - мягкие цвета, художественная текстура.
    """
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        
        if remove_bg:
            no_bg = anonymizer.remove_background(img)
        else:
            no_bg = img
        
        result_array = anonymizer.apply_pastel_painting_effect(no_bg)
        
        if remove_bg:
            result_array = anonymizer.composite_with_artistic_background(result_array, "white")
        
        result_rgb = cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_rgb)
        
        filename = f"{uuid.uuid4().hex}_pastel.png"
        file_path = os.path.join(OUT_DIR, filename)
        result_image.save(file_path)
        path_normalized = file_path.replace("\\", "/")
        
        return {
            "message": "Pastel painting effect completed - soft colors with artistic texture",
            "stylized_url": f"http://localhost:8000/static/{filename}",
            "local_path": path_normalized,
            "style": "pastel",
            "settings": {"remove_bg": remove_bg, "effect": "pastel"}
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post('/anonymize-realistic')
async def anonymize_realistic(
    image: UploadFile = File(...),
    remove_bg: bool = Query(default=True, description="Remove background")
):
    """
    Реалистичная анонимизация - сохраняет фотореалистичность, минимальные изменения.
    """
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    try:
        # Читаем изображение
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        
        # Применяем реалистичную анонимизацию
        result_array = anonymizer.process_image(
            img, 
            style="realistic",
            remove_bg=remove_bg,
            blur_face=True,
            background_style="white",
            anonymize_mode="realistic"
        )
        
        # Конвертируем обратно в PIL Image
        result_rgb = cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_rgb)
        
        # Сохраняем результат
        filename = f"{uuid.uuid4().hex}_realistic_anonymized.png"
        file_path = os.path.join(OUT_DIR, filename)
        result_image.save(file_path)
        
        # Нормализуем путь для Windows
        path_normalized = file_path.replace("\\", "/")
        
        return {
            "message": "Realistic anonymization completed - photo-realistic with minimal changes",
            "stylized_url": f"http://localhost:8000/static/{filename}",
            "local_path": path_normalized,
            "style": "realistic",
            "settings": {
                "remove_bg": remove_bg,
                "blur_face": True,
                "background_style": "white",
                "anonymization_type": "realistic"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post('/anonymize-soft')
async def anonymize_soft(
    image: UploadFile = File(...),
    style: str = Query(default="cartoon", description="Style: cartoon, oil_painting, stylization"),
    remove_bg: bool = Query(default=True, description="Remove background"),
    background_style: str = Query(default="gradient", description="Background: gradient, abstract, white")
):
    """
    Мягкая анонимизация - сохраняет черты лица, но делает менее узнаваемыми.
    """
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    try:
        # Читаем изображение
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        
        # Применяем мягкую анонимизацию
        result_array = anonymizer.process_image(
            img, 
            style=style,
            remove_bg=remove_bg,
            blur_face=True,  # Использует soft_anonymize_faces
            background_style=background_style,
            anonymize_mode="soft"
        )
        
        # Конвертируем обратно в PIL Image
        result_rgb = cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_rgb)
        
        # Сохраняем результат
        filename = f"{uuid.uuid4().hex}_soft_anonymized.png"
        file_path = os.path.join(OUT_DIR, filename)
        result_image.save(file_path)
        
        # Нормализуем путь для Windows
        path_normalized = file_path.replace("\\", "/")
        
        return {
            "message": "Soft anonymization completed - facial features preserved",
            "stylized_url": f"http://localhost:8000/static/{filename}",
            "local_path": path_normalized,
            "style": style,
            "settings": {
                "remove_bg": remove_bg,
                "blur_face": True,
                "background_style": background_style,
                "anonymization_type": "soft"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post('/anonymize-advanced')
async def anonymize_advanced(image: UploadFile = File(...)):
    """
    Максимальная анонимизация с искажениями, цветовым сдвигом и абстрактным фоном.
    Идеально для полной трансформации изображения для Sora AI.
    """
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    try:
        # Читаем изображение
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        
        # Применяем продвинутую анонимизацию
        result_array = anonymizer.apply_advanced_anonymization(img)
        
        # Конвертируем обратно в PIL Image
        result_rgb = cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB)
        result_img = Image.fromarray(result_rgb)
        
        # Сохраняем результат
        filename = f"{uuid.uuid4().hex}_advanced_anonymized.png"
        path = os.path.join(OUT_DIR, filename)
        result_img.save(path, format='PNG')
        
        # Конвертируем путь в формат с прямыми слешами
        path_normalized = path.replace('\\', '/')
        
        return {
            "stylized_url": f"http://localhost:8000/static/{filename}",
            "local_path": path_normalized,
            "method": "advanced_anonymization",
            "description": "Maximum anonymization with distortions, color shift and abstract background"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post('/remove-background')
async def remove_background_only(image: UploadFile = File(...)):
    """
    Простое удаление фона без дополнительной обработки.
    Возвращает изображение с прозрачным фоном (PNG).
    """
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    try:
        # Читаем изображение
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        
        # Удаляем фон
        result_img = anonymizer.remove_background(img)
        
        # Сохраняем результат
        filename = f"{uuid.uuid4().hex}_no_bg.png"
        path = os.path.join(OUT_DIR, filename)
        result_img.save(path, format='PNG')
        
        # Конвертируем путь в формат с прямыми слешами
        path_normalized = path.replace('\\', '/')
        
        return {
            "stylized_url": f"http://localhost:8000/static/{filename}",
            "local_path": path_normalized,
            "method": "background_removal",
            "description": "Background removed, transparent PNG"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
