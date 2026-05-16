from varch.vision_db import VisionDB
from varch.vlm import VLM


class VisualArchive:

    def __init__(self, path, device, load_db=False):
        self.device = device
        self.db = VisionDB(path, device, load=load_db)
        self.vlm = None if not load_db else VLM(self.device)

    def observe(self, path):
        self.db.observe(path)

    def search(self, query, k=5):
        relevant_paths, scores = self.db.search(query, k)
        answer = self.vlm.generate_answer(query, relevant_paths, scores)
        return answer
    


