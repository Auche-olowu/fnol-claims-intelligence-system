from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your model files
upload_folder(folder_path="C:/Users/akuom/OneDrive/Documents/Amdari/FNOL_Project/models", repo_id="Akuoma12/ultimate_claim_cost_model", repo_type="model")
