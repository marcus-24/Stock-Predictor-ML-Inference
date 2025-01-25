from pydantic import HttpUrl, SecretStr
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):

    # FRONTEND_URL: HttpUrl
    FRONTEND_URL: str
    ENV: str


class HuggingFaceSettings(BaseSettings):
    MODEL_URL: str
    DATA_REPO: str


class EvidentlySettings(BaseSettings):

    EVIDENTLY_TOKEN: SecretStr
    EVIDENTLY_PROJECT_ID: SecretStr


class GitHubSettings(BaseSettings):
    GITHUB_TOKEN: SecretStr
