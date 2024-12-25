import aiomysql
import os
from dotenv import load_dotenv
load_dotenv()

DATABASE_CONFIG = {
    "host": os.getenv('DB_HOST'),
    "port": int(os.getenv('DB_PORT')),
    "user": os.getenv('DB_USER'),
    "password": os.getenv('DB_PASSWORD'),
    "db": os.getenv('DB_SCHEMA'),
}

async def get_db():
    pool = await aiomysql.create_pool(**DATABASE_CONFIG)
    return pool