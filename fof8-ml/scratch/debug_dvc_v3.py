import dvc.api
import os

repo_root = "/workspaces/fof8-scout"
data_path = "fof8-gen/data/raw"

print(f"Current Working Directory: {os.getcwd()}")

try:
    url = dvc.api.get_url(path=os.path.join(repo_root, data_path), repo=repo_root)
    print(f"URL (absolute path): {url}")
except Exception as e:
    print(f"Error (absolute path): {e}")

try:
    os.chdir(os.path.join(repo_root, "fof8-gen/data"))
    url = dvc.api.get_url(path="raw")
    print(f"URL (in subdir, path='raw'): {url}")
except Exception as e:
    print(f"Error (in subdir, path='raw'): {e}")
