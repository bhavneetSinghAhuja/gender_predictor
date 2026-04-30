from huggingface_hub import HfApi
from huggingface_hub import login
import os

def create_space(repo_id=None):
    print("=== Creating Hugging Face Space ===\n")
    
    token = os.getenv("HF_TOKEN")
    if not token:
        token = input("Enter your Hugging Face API token: ").strip()
    login(token)
    
    api = HfApi()
    
    if not repo_id:
        username = api.whoami(token)["name"]
        repo_id = f"{username}/gender-predictor"
    
    print(f"Creating Space: {repo_id}")
    
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True
        )
        print(f"Space repository created: {repo_id}")
    except Exception as e:
        print(f"Space creation: {e}")
        if "409" in str(e):
            print("Space already exists, uploading files...")
    
    print("\nUploading app.py...")
    api.upload_file(
        path_or_fileobj="app.py",
        path_in_repo="app.py",
        repo_id=repo_id,
        repo_type="space"
    )
    print("Uploaded app.py")
    
    print("\nUploading requirements.txt...")
    api.upload_file(
        path_or_fileobj="space_requirements.txt",
        path_in_repo="requirements.txt",
        repo_id=repo_id,
        repo_type="space"
    )
    print("Uploaded requirements.txt")
    
    print(f"\nSpace deployed: https://huggingface.co/spaces/{repo_id}")
    return repo_id

if __name__ == "__main__":
    create_space()
