import dvc.api
import os

repo_root = "/workspaces/fof8-scout"
data_path_from_root = "fof8-gen/data/raw"

cwd = os.getcwd()
try:
    os.chdir(repo_root)
    url = dvc.api.get_url(data_path_from_root)
    print(f"URL: {url}")
finally:
    os.chdir(cwd)
