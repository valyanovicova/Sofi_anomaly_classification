# Cover Letter Anomaly Detector  

Инструмент на базе **FastAPI + CatBoost**, который анализирует тексты сопроводительных писем и определяет, содержат ли они аномалии (например, смешение языков, ошибки пола, лишние навыки и пр.).  

## Возможности  
- Классификация писем на «нормальные» и «аномальные»  
- Учет:  
  - транслита и смешения алфавитов  
  - несогласованности по полу (муж./жен. род)  
  - добавления новых/редких навыков  
- Fast API для интеграции в другие сервисы  

---

## Установка и запуск  

1. Клонировать репозиторий:  
   ```bash
   git clone https://github.com/valyanovicova/Sofi_anomaly_classification.git
   cd Sofi_anomaly_classification
## Создание виртуального окружения и установка зависимостей
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt

## Запуск FastAPI-сервиса
uvicorn main:app --reload

##Доступен по адресу: http://127.0.0.1:8000/predict

## Пример запроса
{
  "cover_letter": "Добрый день!

Меня зовут Никита, и я хочу откликнуться на вакансию Frontend-разработчика."
}

## Пример ответа
{
  "anomaly": true,
  "text_length": 1120,
  "score": 0.83,
  "threshold": 0.4
}
## Структура проекта
📂 project

 ┣ 📜 main.py              # FastAPI приложение
 
 ┣ 📜 cat_model_post.cbm   # первая модель
 
 ┣ 📜 cat_model_pre.cbm    # вторая модель
 
 ┣ 📜 tfidf_post_1.pkl     # векторизатор для первой модели
 
 ┣ 📜 tfidf_pre.pkl        # векторизатор для второй модели
 
 ┣ 📜 requirements.txt     # зависимости
 
 ┗ 📜 README.md            # описание проекта

