import sys
import subprocess

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        subprocess.run(
            ["python3", "src/serving/serve.py"],
            check=True,
        )
    else:
        subprocess.run(sys.argv[1:], check=True)