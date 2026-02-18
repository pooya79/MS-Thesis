import os

os.environ.setdefault("APP_PASSWORD", "test-password")
os.environ.setdefault("APP_AUTH_SECRET", "test-auth-secret")

from server.app.core.config import get_settings

get_settings.cache_clear()
