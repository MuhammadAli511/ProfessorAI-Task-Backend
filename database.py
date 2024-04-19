import sys
from pymongo import MongoClient
from config import settings


try:
    client = MongoClient(settings.DATABASE_URL)
    chat_db = client["professorai"]
    print("Connected to DB")

except Exception as e:
    print(e)
    sys.exit(1)
