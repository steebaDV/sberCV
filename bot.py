import telebot
from PIL import Image
import io
import os
from dotenv import load_dotenv

from pipeline import Model, Pipeline, transform

load_dotenv()
TOKEN = os.getenv('BOT_TOKEN')

bot = telebot.TeleBot(TOKEN)

pipeline = Pipeline(Model, transform)


@bot.message_handler(content_types=['photo'])
def send_classified_breed(message):
    file = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file.file_path)
    pil_image = Image.open(io.BytesIO(downloaded_file))
    label = pipeline.predict(pil_image)[0]

    bot.send_message(message.chat.id, label)


if __name__ == '__main__':
    bot.infinity_polling()
