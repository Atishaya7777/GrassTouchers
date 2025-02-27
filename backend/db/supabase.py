from supabase import Client, create_client
from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables

SUPABASE_KEY = os.getenv("FASTAPI_SUPABASE_KEY")
SUPABASE_URL = os.getenv("FASTAPI_SUPABASE_URL")


if not all([SUPABASE_URL, SUPABASE_KEY]):
    raise ValueError(
        "Please set SUPABASE_URL and SUPABASE_KEY environment variables")


def create_supabase_client():
    supabase: Client = create_client(SUPABASE_URL or "", SUPABASE_KEY or "")
    return supabase
