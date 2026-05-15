import os
import requests
import logging
from io import BytesIO
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO, filename="tg_bot_logs.txt",
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Отправь мне изображение, и я передам его модели для анализа."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатываем входящие фотографии и отправляем их в API модели."""
    try:
        user = update.effective_user

        logger.info(
            f"Получено фото от пользователя @{user.username} "
            f"(ID: {user.id}, Имя: {user.first_name})"
        )

        # Получаем файл фотографии
        photo_file = await update.message.photo[-1].get_file()

        # Скачиваем файл в память
        image_bytes = BytesIO()
        await photo_file.download_to_memory(image_bytes)
        image_bytes.seek(0)

        # Отправляем изображение в локальный API модели
        files = {
            "image": ("telegram_photo.jpg", image_bytes, "image/jpeg")
        }

        response = requests.post(
            'http://127.0.0.1:8081/api/predict',
            files=files,
            timeout=60
        )
        response.raise_for_status()

        data = response.json()

        result = data["label"]
        prob_real = data["prob_real"]
        prob_fake = data["prob_fake"]

        logger.info(
            f"Результат для пользователя "
            f"{user.id} {user.first_name} {user.last_name} {user.username}: "
            f"{result}, real={prob_real}%, fake={prob_fake}%"
        )

        await update.message.reply_text(
            f"Результат модели: {result}\n"
            f"Real: {prob_real}%\n"
            f"Fake: {prob_fake}%"
        )

    except Exception:
        logger.exception("Ошибка при обработке фото")
        await update.message.reply_text(
            "Произошла ошибка при обработке изображения. Попробуйте ещё раз."
        )


def main():
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    application.run_polling()


if __name__ == "__main__":
    main()
