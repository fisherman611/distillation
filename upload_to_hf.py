import os

# from dotenv import load_dotenv
from huggingface_hub import HfApi


# load_dotenv()
# token = os.getenv("HF_TOKEN_UPLOAD")

# if not token:
#     raise RuntimeError(
#         "Missing Hugging Face token. Add HF_TOKEN=... to .env or export HF_TOKEN."
#     )

# api = HfApi(token=token)
api = HfApi()

api.upload_folder(
    folder_path="results/",
    repo_id="fisherman611/t2c_qwen2.5",
    repo_type="model",
)
