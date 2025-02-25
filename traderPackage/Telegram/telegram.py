import requests
import time
from dotenv import load_dotenv
import os

load_dotenv()  # Automatically loads the .env file

# Telegram bot bilgileri
TELEGRAM_TOKEN = os.getenv('MY_TOKEN')
CHAT_ID = os.getenv('MY_ID')   # Chat ID'niz

if not TELEGRAM_TOKEN:
    raise ValueError("No API key found in .env file.")

if not CHAT_ID:
    raise ValueError("No API key found in .env file.")


# Mesaj gönderme fonksiyonu
def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    payload = {'chat_id': CHAT_ID, 'text': message}
    response = requests.post(url, data=payload)
    print(response.json())  # Yanıtı kontrol et

# Mesaj gönderme
send_telegram_message("API keylerini deniyorum")
# Sonsuz döngü ile her dakika mesaj gönderme
#while True:
#    send_telegram_message()
#    time.sleep(60)  # 60 saniye bekle