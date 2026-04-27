import dvc.api
import os

repo_root = "/workspaces/fof8-scout"
data_path = "fof8-gen/data/raw"

try:
    url = dvc.api.get_url(data_path, repo=repo_root)
    print(f"URL: {url}")
except Exception as e:
    print(f"Error: {e}")
