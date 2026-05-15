from vision_db import VisionDB
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

db = VisionDB(device)
db.observe(r'/home/yuval-rubin/Projects/vision_rag/dataset', r'/home/yuval-rubin/Projects/vision_rag/db')
