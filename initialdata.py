from datasets import load_dataset
from torch.utils.data import DataLoader

cv_13 = load_dataset("mozilla-foundation/common_voice_13_0", "hi", split="train")
dataloader = DataLoader(cv_13, batch_size=32) # DOES NOT WORK! Do we need auth token to access data?


languages = ["en","fr","hi","ar","it","es"] # languages to be used 
