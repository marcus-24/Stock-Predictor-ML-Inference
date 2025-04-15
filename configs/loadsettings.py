from pydantic import SecretStr
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):

    # FRONTEND_URL: HttpUrl
    FRONTEND_URL: str
    ENV: str


class HuggingFaceSettings(BaseSettings):
    MODEL_URL: str


class EvidentlySettings(BaseSettings):

    EVIDENTLY_TOKEN: SecretStr
    EVIDENTLY_PROJECT_ID: SecretStr


class GitHubSettings(BaseSettings):
    GITHUB_TOKEN: SecretStr


class HopsworksSettings(BaseSettings):
    HOPSWORKS_KEY: SecretStr
