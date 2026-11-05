import os
import subprocess
import sys

REPO = "https://github.com/MTalhaKhan326/cv-portfolio1"
BRANCH = "main"
DATA = "/kaggle/input/garbage-detection"
OUT = "/kaggle/working/outputs"


def get_token():
    from kaggle_secrets import UserSecretsClient
    return UserSecretsClient().get_secret("GITHUB_TOKEN")


url = f"https://{get_token()}@github.com/{REPO}.git"
subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, url, "/tmp/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "/tmp/repo/requirements.txt"], check=True)
subprocess.run([sys.executable, "/tmp/repo/main.py", "--data", DATA, "--out", OUT], check=True)