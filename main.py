import os
import json
import numpy as np
from io import BytesIO
from PIL import Image
import torch
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from sentence_transformers import SentenceTransformer
import logging
import gc

# Отключаем GPU и настраиваем PyTorch для экономии памяти
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(1)  # Ограничиваем количество потоков

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv('TOKEN')
DB_FILE = "wardrobe_db.json"
IMG_DIR = "temp_images"
os.makedirs(IMG_DIR, exist_ok=True)

# Инициализация модели с оптимизацией памяти
model = None

def load_model():
    global model
    if model is None:
        # Загружаем модель с пониженной точностью для экономии памяти
        model = SentenceTransformer('clip-ViT-B-32', device='cpu')
        # Переводим модель в режим eval и half-precision
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return model

def load_db():
    try:
        if os.path.exists(DB_FILE):
            with open(DB_FILE, 'r') as f:
                db = json.load(f)
                # Конвертируем эмбеддинги обратно в numpy array при загрузке
                for item in db.get("items", []):
                    if "embedding" in item:
                        item["embedding"] = np.array(item["embedding"], dtype=np.float16)
                return db
    except Exception as e:
        logger.error(f"Ошибка загрузки БД: {e}")
    return {"items": []}

def save_db(db):
    try:
        # Создаем копию без numpy массивов для сериализации
        db_copy = {"items": []}
        for item in db.get("items", []):
            item_copy = item.copy()
            if "embedding" in item_copy:
                item_copy["embedding"] = item_copy["embedding"].tolist()
            db_copy["items"].append(item_copy)
        
        with open(DB_FILE, 'w') as f:
            json.dump(db_copy, f, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения БД: {e}")

async def cleanup_memory():
    """Очистка памяти"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        # Загружаем модель только при первом использовании
        model = load_model()
        
        photo = update.message.photo[-1]  # Берем самое большое фото
        file = await photo.get_file()
        item_id = str(photo.file_id)
        
        # Загружаем фото в память без сохранения на диск
        image_bytes = await file.download_as_bytearray()
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Уменьшаем размер изображения для экономии памяти
        img = img.resize((224, 224))
        
        # Вычисляем эмбеддинг с отключенным градиентом
        with torch.no_grad():
            img_emb = model.encode(img, convert_to_tensor=False)
        
        # Преобразуем в float16 для экономии памяти
        img_emb = img_emb.astype(np.float16)
        
        # Сохраняем в базу данных
        db = load_db()
        db["items"].append({
            "id": item_id,
            "file_path": f"{IMG_DIR}/{item_id}.jpg",  # Путь, но файл не сохраняем
            "embedding": img_emb,
            "type": "clothes"
        })
        save_db(db)
        
        # Очищаем память
        del img, img_emb, image_bytes
        await cleanup_memory()
        
        await update.message.reply_text(f"✅ Вещь добавлена! Всего: {len(db['items'])}")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")
        logger.error(f"Ошибка в save_item: {e}")

async def generate_look(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        model = load_model()
        db = load_db()
        
        if not db["items"]:
            await update.message.reply_text("В гардеробе пока нет вещей!")
            return
        
        # Простейшая реализация - случайный выбор
        items = np.random.choice(db["items"], size=min(3, len(db["items"])), replace=False)
        
        response = "👗 *Предлагаемый образ:*\n"
        for i, item in enumerate(items, 1):
            response += f"{i}. Вещь ID: {item['id']}\n"
        
        await update.message.reply_text(response, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")
        logger.error(f"Ошибка в generate_look: {e}")
    finally:
        await cleanup_memory()

async def random_look(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await generate_look(update, context)

async def show_wardrobe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        db = load_db()
        
        if not db["items"]:
            await update.message.reply_text("Ваш гардероб пуст!")
            return
        
        response = "👚 *Ваш гардероб:*\n"
        for i, item in enumerate(db["items"], 1):
            response += f"{i}. ID: {item['id']}\n"
        
        await update.message.reply_text(response, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")
        logger.error(f"Ошибка в show_wardrobe: {e}")

async def remove_item(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not context.args:
            await update.message.reply_text("Укажите ID вещи для удаления: /remove ID")
            return
        
        item_id = context.args[0]
        db = load_db()
        
        initial_count = len(db["items"])
        db["items"] = [item for item in db["items"] if item["id"] != item_id]
        
        if len(db["items"]) < initial_count:
            save_db(db)
            await update.message.reply_text(f"✅ Вещь {item_id} удалена!")
        else:
            await update.message.reply_text(f"❌ Вещь с ID {item_id} не найдена!")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")
        logger.error(f"Ошибка в remove_item: {e}")

async def handle_reference(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text_query = update.message.text
        if not text_query:
            await update.message.reply_text("Отправьте текстовое описание для поиска")
            return
        
        model = load_model()
        db = load_db()
        
        if not db["items"]:
            await update.message.reply_text("В гардеробе пока нет вещей!")
            return
        
        # Кодируем текстовый запрос
        with torch.no_grad():
            query_embedding = model.encode(text_query, convert_to_tensor=False)
        query_embedding = query_embedding.astype(np.float16)
        
        # Находим ближайшие вещи
        similarities = []
        for item in db["items"]:
            if "embedding" in item:
                emb = item["embedding"]
                sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                similarities.append((sim, item))
        
        # Сортируем по убыванию сходства
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_items = [item for (sim, item) in similarities[:3]]
        
        if not top_items:
            await update.message.reply_text("Не найдено подходящих вещей")
            return
        
        response = f"🔍 *Результаты по запросу '{text_query}':*\n"
        for i, item in enumerate(top_items, 1):
            response += f"{i}. Вещь ID: {item['id']} (сходство: {similarities[i-1][0]:.2f})\n"
        
        await update.message.reply_text(response, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")
        logger.error(f"Ошибка в handle_reference: {e}")
    finally:
        await cleanup_memory()

def main():
    try:
        # Предварительная загрузка модели
        load_model()
        
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
