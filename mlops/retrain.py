# standard imports
import requests
from requests import Response
import json

# local imports
from configs.loadsettings import GitHubSettings

GIT_SETTINGS = GitHubSettings()
HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GIT_SETTINGS.GITHUB_TOKEN.get_secret_value()}",
    "X-GitHub-Api-Version": "2022-11-28",
}
OWNER = "marcus-24"
REPO = "Stock-Predictor-ML-Training"
WORKFLOW_ID = "model-train-pipeline.yml"


def trigger_ml_model_retrain(
    owner: str = OWNER,
    repo: str = REPO,
    workflow_id: str = WORKFLOW_ID,
    headers: dict[str, str] = HEADERS,
) -> Response:
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches"
    data = json.dumps({"ref": "main"})
    return requests.post(url=url, headers=headers, data=data)
