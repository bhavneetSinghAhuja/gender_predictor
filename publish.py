from huggingface_hub import HfApi, login
import os
from publish_to_hub import publish_to_hub
from create_space import create_space

def publish_all():
    print("=" * 60)
    print("  PUBLISHING TO HUGGING FACE (Model Hub + Space)")
    print("=" * 60)
    print()
    
    token = os.getenv("HF_TOKEN")
    if not token:
        token = input("Enter your Hugging Face API token: ").strip()
    os.environ["HF_TOKEN"] = token
    
    login(token)
    api = HfApi()
    username = api.whoami(token)["name"]
    repo_id = f"{username}/gender-predictor"
    
    print("\n" + "-" * 50)
    print("Step 1: Upload model to Hub")
    print("-" * 50)
    model_url = publish_to_hub(repo_id=repo_id)
    
    print("\n" + "-" * 50)
    print("Step 2: Deploy Gradio Space")
    print("-" * 50)
    space_url = create_space(repo_id=repo_id)
    
    print("\n" + "=" * 60)
    print("  PUBLISHING COMPLETE!")
    print("=" * 60)
    print(f"\nModel Hub:  https://huggingface.co/{repo_id}")
    print(f"Space Demo: https://huggingface.co/spaces/{repo_id}")
    print()

if __name__ == "__main__":
    publish_all()
