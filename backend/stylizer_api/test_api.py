"""
Профессиональный тестовый скрипт для Dream Maker API
Тестирует все эндпоинты анонимизации и стилизации
Запустите сервер (python app.py) перед запуском этого скрипта
"""

import requests
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

# ===============================
# Конфигурация
# ===============================
API_BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = os.path.join(
    "D:/WorkSpace/dream-maker/dream-maker/backend/stylizer_api/uploads",
    "bogdanREAL.jpg"
)

# Цвета для CLI
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"


def print_header(title: str):
    """Печать заголовка с форматированием"""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{YELLOW}{title}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")


def test_endpoint(
    endpoint: str,
    files: Dict,
    params: Optional[Dict] = None,
    description: str = ""
) -> Optional[Dict[str, Any]]:
    """
    Тестирование эндпоинта с подробным логированием
    
    Args:
        endpoint: URL эндпоинта (например, "/anonymize")
        files: Словарь файлов для загрузки
        params: Опциональные параметры запроса
        description: Описание теста
    
    Returns:
        Словарь с результатами или None при ошибке
    """
    print_header(f"🧪 Тестирование: {endpoint}")
    print(f"📝 {description}")
    if params:
        print(f"🔧 Параметры: {params}")
    print("-" * 80)

    url = f"{API_BASE_URL}{endpoint}"
    start_time = time.time()

    try:
        response = requests.post(url, files=files, params=params or {}, timeout=120)
        duration = time.time() - start_time
        
        print(f"⏱️  Время выполнения: {duration:.2f} сек")
        
        if response.status_code != 200:
            print(f"{RED}❌ Ошибка {response.status_code}{RESET}")
            print(f"📝 Ответ: {response.text[:500]}")
            return None
        
        data = response.json()
        print(f"{GREEN}✅ Успешно!{RESET}")
        print(f"{CYAN}📦 Ответ сервера:{RESET}")
        
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"   {key}: {value[:100]}...")
            else:
                print(f"   {key}: {value}")
        
        return data
            
    except requests.exceptions.ConnectionError:
        print(f"{RED}❌ Не удалось подключиться к {API_BASE_URL}{RESET}")
        print("💡 Убедитесь, что сервер запущен: python app.py")
        return None
    except requests.exceptions.Timeout:
        print(f"{RED}❌ Превышено время ожидания (120 сек){RESET}")
        return None
    except Exception as e:
        print(f"{RED}❌ Ошибка при запросе: {e}{RESET}")
        return None


def check_api_health() -> bool:
    """Проверка доступности API"""
    print("\n🔍 Проверка доступности API...")
    try:
        response = requests.get(API_BASE_URL, timeout=5)
        print(f"{GREEN}✅ API доступен (статус {response.status_code}){RESET}")
        return True
    except Exception:
        print(f"{RED}❌ API недоступен на {API_BASE_URL}{RESET}")
        print("💡 Запустите сервер: python app.py")
        return False


def find_test_image() -> Optional[str]:
    """Поиск тестового изображения"""
    test_path = TEST_IMAGE_PATH
    
    if os.path.exists(test_path):
        return test_path
    
    # Ищем альтернативу в папке uploads
    uploads = Path("uploads")
    if uploads.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            images = list(uploads.glob(ext))
            if images:
                found_path = str(images[0])
                print(f"{YELLOW}⚠️  {test_path} не найден, используем {found_path}{RESET}")
                return found_path
    
    print(f"{RED}❌ Тестовое изображение не найдено!{RESET}")
    return None


def main():
    print_header("🚀 Dream Maker API — Комплексное тестирование")
    print("Тестируем все эндпоинты анонимизации и стилизации")
    
    # Проверка API
    if not check_api_health():
        return
    
    # Поиск тестового изображения
    test_image_path = find_test_image()
    if not test_image_path:
        return
    
    print(f"📸 Используем изображение: {test_image_path}")
    
    results = []
    
    # ========================================
    # ТЕСТ 1: Основная анонимизация
    # ========================================
    with open(test_image_path, "rb") as f:
        data = test_endpoint(
            "/anonymize",
            files={"image": f},
            params={"remove_bg": True, "anonymize_mode": "realistic"},
            description="Основная анонимизация с удалением фона и realistic режимом"
        )
        results.append(("Anonymize (realistic)", data is not None))
    
    # ========================================
    # ТЕСТ 2: Soft анонимизация
    # ========================================
    with open(test_image_path, "rb") as f:
        data = test_endpoint(
            "/anonymize-soft",
            files={"image": f},
            description="Мягкая анонимизация с минимальными изменениями"
        )
        results.append(("Anonymize Soft", data is not None))
    
    # ========================================
    # ТЕСТ 3: Realistic анонимизация
    # ========================================
    with open(test_image_path, "rb") as f:
        data = test_endpoint(
            "/anonymize-realistic",
            files={"image": f},
            description="Реалистичная анонимизация с сохранением черт лица"
        )
        results.append(("Anonymize Realistic", data is not None))
    
    # ========================================
    # ТЕСТ 4: Advanced анонимизация
    # ========================================
    with open(test_image_path, "rb") as f:
        data = test_endpoint(
            "/anonymize-advanced",
            files={"image": f},
            description="Продвинутая анонимизация с максимальными изменениями"
        )
        results.append(("Anonymize Advanced", data is not None))
    
    # ========================================
    # ТЕСТ 5: Wax Figure (восковая фигура)
    # ========================================
    with open(test_image_path, "rb") as f:
        data = test_endpoint(
            "/wax-figure",
            files={"image": f},
            description="Эффект восковой фигуры - гладкая кожа, пластиковый вид"
        )
        results.append(("Wax Figure", data is not None))
    
    # ========================================
    # ТЕСТ 6: Toon (мультяшный стиль)
    # ========================================
    with open(test_image_path, "rb") as f:
        data = test_endpoint(
            "/toon",
            files={"image": f},
            description="Мультяшный стиль - яркие цвета, чёткие контуры"
        )
        results.append(("Toon", data is not None))
    
    # ========================================
    # ТЕСТ 7: Clay Sculpture (глиняная скульптура)
    # ========================================
    with open(test_image_path, "rb") as f:
        data = test_endpoint(
            "/clay-sculpture",
            files={"image": f},
            description="Эффект глиняной скульптуры - матовая текстура"
        )
        results.append(("Clay Sculpture", data is not None))
    
    # ========================================
    # ТЕСТ 8: Comic Style (комикс)
    # ========================================
    with open(test_image_path, "rb") as f:
        data = test_endpoint(
            "/comic-style",
            files={"image": f},
            description="Стиль комикса - контурные линии, постеризация"
        )
        results.append(("Comic Style", data is not None))
    
    # ========================================
    # ТЕСТ 9: CGI Character (3D персонаж)
    # ========================================
    with open(test_image_path, "rb") as f:
        data = test_endpoint(
            "/cgi-character",
            files={"image": f},
            description="CGI персонаж - эффект 3D рендера"
        )
        results.append(("CGI Character", data is not None))
    
    # ========================================
    # ТЕСТ 10: Pastel Painting (пастель)
    # ========================================
    with open(test_image_path, "rb") as f:
        data = test_endpoint(
            "/pastel-painting",
            files={"image": f},
            description="Пастельная живопись - мягкие цвета, нежная текстура"
        )
        results.append(("Pastel Painting", data is not None))
    
    # ========================================
    # ТЕСТ 11: Remove Background
    # ========================================
    with open(test_image_path, "rb") as f:
        data = test_endpoint(
            "/remove-background",
            files={"image": f},
            description="Удаление фона без анонимизации"
        )
        results.append(("Remove Background", data is not None))
    
    # ========================================
    # ИТОГИ
    # ========================================
    print_header("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = f"{GREEN}✅ PASS{RESET}" if success else f"{RED}❌ FAIL{RESET}"
        print(f"{status} - {name}")
    
    print(f"\n📈 Успешно: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print(f"\n{GREEN}🎉 Все тесты пройдены успешно!{RESET}")
    else:
        print(f"\n{YELLOW}⚠️  Некоторые тесты провалились ({total - passed} из {total}){RESET}")
    
    print(f"\n{CYAN}💡 Результаты сохранены в папке: out/{RESET}")
    print(f"{CYAN}🔍 Проверьте визуально качество обработки{RESET}")
    
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{CYAN}📝 Информация о тестах:{RESET}")
    print(f"   • Anonymize: Основная анонимизация с параметрами")
    print(f"   • Anonymize Soft: Минимальные изменения лица")
    print(f"   • Anonymize Realistic: Реалистичные изменения")
    print(f"   • Anonymize Advanced: Максимальные изменения")
    print(f"   • Wax Figure: Эффект восковой фигуры")
    print(f"   • Toon: Мультяшный стиль")
    print(f"   • Clay Sculpture: Глиняная скульптура")
    print(f"   • Comic Style: Стиль комикса")
    print(f"   • CGI Character: 3D персонаж")
    print(f"   • Pastel Painting: Пастельная живопись")
    print(f"   • Remove Background: Удаление фона")
    print(f"{BLUE}{'='*80}{RESET}")


if __name__ == "__main__":
    main()
