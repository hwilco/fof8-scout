import dvc.api
import os

repo_root = "/workspaces/fof8-scout"
data_path = "fof8-gen/data/raw"

print(f"Current Working Directory: {os.getcwd()}")

try:
    url = dvc.api.get_url(path=data_path, repo=repo_root, remote='origin')
    print(f"URL (remote='origin'): {url}")
except Exception as e:
    print(f"Error (remote='origin'): {e}")

try:
    url = dvc.api.get_url(path=data_path, repo=repo_root)
    print(f"URL (default remote): {url}")
except Exception as e:
    print(f"Error (default remote): {e}")
