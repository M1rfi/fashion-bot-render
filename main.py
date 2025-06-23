import os
import json
import numpy as np
from PIL import Image
import torch
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from sentence_transformers import SentenceTransformer
import logging

# Отключаем GPU, чтобы не занимать память под cuda
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = os.getenv('TOKEN')
DB_FILE = "wardrobe_db.json"
IMG_DIR = "temp_images"
os.makedirs(IMG_DIR, exist_ok=True)

# Инициализация модели (один раз при старте)
model = SentenceTransformer('clip-ViT-B-32')

def load_db():
    try:
        if os.path.exists(DB_FILE):
            with open(DB_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки БД: {e}")
    return {"items": []}

def save_db(db):
    try:
        with open(DB_FILE, 'w') as f:
            json.dump(db, f)
    except Exception as e:
        logger.error(f"Ошибка сохранения БД: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text(
            "👗 *Гардероб Бот* v2.0\n\n"
            "1. Отправь фото вещи - я сохраню её\n"
            "2. Отправь текст (напр. 'вечерний образ') или фото-референс\n"
            "3. Команды:\n"
            "   /look - создать образ\n"
            "   /random - случайный образ\n"
            "   /wardrobe - показать вещи\n"
            "   /remove ID - удалить вещь",
            parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Ошибка в start: {e}")

async def save_item(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        photo = update.message.photo[-1]
        file = await photo.get_file()
        item_id = str(photo.file_id)
        filename = f"{IMG_DIR}/{item_id}.jpg"

        await file.download_to_drive(filename)

        # Открываем изображение и уменьшаем размер для экономии памяти
        img = Image.open(filename).convert('RGB')
        img = img.resize((224, 224))

        # Получаем эмбеддинг
        img_emb = model.encode(img)

        # Преобразуем в float16 для уменьшения размера
        img_emb_16 = img_emb.astype(np.float16)

        db = load_db()
        db["items"].append({
            "id": item_id,
            "file_path": filename,
            "embedding": img_emb_16.tolist(),
            "type": "clothes"
        })
        save_db(db)

        await update.message.reply_text(f"✅ Вещь добавлена! Всего: {len(db['items'])}")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")
        logger.error(f"Ошибка в save_item: {e}")

# Заглушки для остальных функций, оставь свои реализации
async def generate_look(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pass

async def random_look(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pass

async def show_wardrobe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pass

async def remove_item(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pass

async def handle_reference(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pass

def main():
    try:
        app = ApplicationBuilder().token(TOKEN).build()

        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("look", generate_look))
        app.add_handler(CommandHandler("random", random_look))
        app.add_handler(CommandHandler("wardrobe", show_wardrobe))
        app.add_handler(CommandHandler("remove", remove_item))
        app.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, save_item))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_reference))

        logger.info("Бот запущен!")
        app.run_polling()
    except Exception as e:
        logger.error(f"Фатальная ошибка: {e}")

if __name__ == "__main__":
    main()
