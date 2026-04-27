import dvc.api
import os

repo_root = "/workspaces/fof8-scout"
data_path = "fof8-gen/data/raw"

try:
    url = dvc.api.get_url(path=data_path, repo=repo_root)
    print(f"URL: {url}")
except Exception as e:
    print(f"Error: {e}")

# Try with absolute path for repo
try:
    url = dvc.api.get_url(path=data_path, repo=os.path.abspath(repo_root))
    print(f"URL (abspath repo): {url}")
except Exception as e:
    print(f"Error (abspath repo): {e}")

# Try changing CWD
os.chdir(repo_root)
try:
    url = dvc.api.get_url(path=data_path)
    print(f"URL (no repo, CWD=root): {url}")
except Exception as e:
    print(f"Error (no repo, CWD=root): {e}")
