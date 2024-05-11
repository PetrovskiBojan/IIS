from dotenv import load_dotenv
import os
from pymongo import MongoClient
load_dotenv();

mongodb_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongodb_uri)

db = client.IIS

collection_name = db['requests_collection']