from slowapi import Limiter
from slowapi.util import get_remote_address

# Default: per-IP rate limiting
limiter = Limiter(key_func=get_remote_address)
