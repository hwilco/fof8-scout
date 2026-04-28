from dvc.repo import Repo
import os

repo_root = "/workspaces/fof8-scout"

try:
    repo = Repo(repo_root)
    print("Outputs found in repo:")
    for out in repo.index.outs:
        print(f" - {out}")
except Exception as e:
    print(f"Error: {e}")
