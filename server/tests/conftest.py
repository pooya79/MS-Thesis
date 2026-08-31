import os

os.environ.setdefault("APP_PASSWORD", "test-password")

from server.app.core.config import get_settings

get_settings.cache_clear()
