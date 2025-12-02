# sms_sender.py
import os
from twilio.rest import Client
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "twi.env"))

ACCOUNT_SID = os.getenv("TWILIO_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE")

if not (ACCOUNT_SID and AUTH_TOKEN and TWILIO_PHONE_NUMBER):
    print("Twilio credentials missing! Check twi.env")

client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_sms(to_number, message):
    try:
        msg = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_number
        )
        print(f"SMS sended: {msg.sid}")
    except Exception as e:
        print("SMS send failed:", e)

def send_sms_to_multiple(numbers, message):
    for num in numbers:
        send_sms(num, message)
