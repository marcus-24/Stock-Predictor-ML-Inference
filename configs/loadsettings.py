from pydantic import HttpUrl
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):

    # FRONTEND_URL: HttpUrl
    FRONTEND_URL: str
    ENV: str
