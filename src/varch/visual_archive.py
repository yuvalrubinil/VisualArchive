from varch.vision_db import VisionDB
from varch.vlm import VLM
from varch.slm import SLM

class VisualArchive:

    def __init__(self, path, device, load_db=False, load_vlm=False, load_slm=False):
        self.device = device
        self.db = VisionDB(path, device, load=load_db)
        self.vlm = None if not load_vlm else VLM(self.device)
        self.slm = None if not load_slm else SLM(self.device)

    def observe(self, path):
        self.db.observe(path)

    def search(self, query, image_path=None, k=5):
        rag_answer = None

        # fusing visual and textual queries
        if self.slm and self.vlm and image_path:
            image_as_text = self.vlm.image_to_text(image_path)
            print(f"image as text: {image_as_text}")
            query = self.slm.fuse_queries(image_as_text, query)
            print(f"updated query: {query}")

        # searching the db
        relevant_paths, scores = self.db.search(query, k)

        if self.vlm:
            # filtering images relevant to the query
            relevant_paths = self.vlm.rank_and_filter(query, relevant_paths)
            # refining into fine grained answer
            rag_answer = self.vlm.generate_answer(query, relevant_paths, dual_modality=self.slm is not None)

        return relevant_paths, scores, rag_answer
    


