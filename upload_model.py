from huggingface_hub import HfApi, HfFolder, Repository
from transformers import AutoModelForSequenceClassification, AutoTokenizer

api = HfApi()
token = HfFolder.get_token()
repo_id = "eneon12345/edwin_model_001"
model_path = "./edwin_model"

api.create_repo(
    repo_id=repo_id,
    token=token,
)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.push_to_hub(repo_id=repo_id)
tokenizer.push_to_hub(repo_id=repo_id)
