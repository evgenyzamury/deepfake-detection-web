from model import inception_model, preprocess
import torch
import logging
from io import BytesIO
from PIL import Image
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO, filename="tg_bot_logs.txt",
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

BOT_TOKEN = ""
TARGET_SIZE = (256, 256)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Отправь мне изображение, и я передам его модели для анализа."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатываем входящие фотографии."""
    try:
        user = update.effective_user

        logger.info(f"Получено фото от пользователя @{user.username} (ID: {user.id}, Имя: {user.first_name})")

        # Получаем файл фотографии
        photo_file = await update.message.photo[-1].get_file()

        # Скачиваем файл в память
        image_bytes = BytesIO()
        await photo_file.download_to_memory(image_bytes)
        image_bytes.seek(0)
        image = Image.open(image_bytes).convert('RGB')  # 3 канала RGB
        image_resized = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

        # prepocess image
        image_tensor = preprocess(image_resized).unsqueeze(0)


        # Вызываем модель
        with torch.no_grad():
            output = inception_model(image_tensor)

        result = int(torch.argmax(output))
        result = 'fake' if result == 1 else 'real'

        logger.info(
            f"Результат для пользователя"
            f" {user.id} {user.first_name} {user.last_name} {user.username}: (класс {result})"
        )
        await update.message.reply_text(f"Результат модели:\n{result}")

    except Exception as e:
        logger.error(f"Ошибка при обработке фото: {e}")
        await update.message.reply_text("Произошла ошибка при обработке изображения. Попробуйте ещё раз.")


def main():
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    application.run_polling()


if __name__ == "__main__":
    main()
