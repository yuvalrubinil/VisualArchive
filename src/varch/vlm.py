import re
import torch
from qwen_vl_utils import process_vision_info
from transformers import (AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration)

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

class VLM:

    def __init__(self, device):
        print("loading vlm...")
        self.device = device
        self.processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL,
            torch_dtype=torch.float16,
            device_map="auto",  
            quantization_config=quantization_config,
            attn_implementation="sdpa",
            local_files_only=True
        )
        print("vlm loaded successfuly")

    
    @torch.no_grad()
    def image_to_text(self, image_path):
        content = [
            {"type": "image", "image": image_path},
            {"type": "text", "text": (
            "Task: Describe the primary subject and their main action in a single, ultra-concise sentence.\n\n"
            "CRITICAL CONSTRAINTS:\n"
            "1. NO BACKGROUNDS: Completely omit walls, floors, windows, furniture, and room layouts.\n"
            "2. NO SPECULATION: State only observable actions directly.\n"
            "3. NO SPECIFIC PATTERNS: Do not mention specific patterns or stripes of objects. Use generic terms (e.g., 'a white sailboat' instead of 'a white sailboat with striped sails').\n"
            "4. FORMAT: Output only the single raw sentence ending with a period. Keep it under 12 words."
        )}
        ]
        
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=24)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        base_description = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        del inputs, generated_ids, generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return base_description
    

    @torch.no_grad()
    def rank_and_filter(self, query, images_paths, threshold=4):
        """Stage 1: Evaluates all images sequentially and returns them sorted by

        relevancy score. Removes images below the threshold.
        """
        scored_images = []

        for path in images_paths:
            content = [{"type": "image", "image": path}]

            rank_prompt = (
                f"Analyze this image's relevancy to the user query.\n"
                f"Query: '{query}'\n\n"
                f"Assign a relevancy score from 0 to 5:\n"
                f"5: Perfect match.\n"
                f"3-4: Partial match.\n"
                f"1-2: Low match.\n"
                f"0: Completely irrelevant.\n\n"
                f"Output format: You MUST start with '[Score: X]'."
            )
            content.append({"type": "text", "text": rank_prompt})

            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            # Restrict tokens tightly since we only care about the prefix score
            generated_ids = self.model.generate(**inputs, max_new_tokens=10)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            match = re.search(r"\[Score:\s*([0-5])\]", output_text)
            score = int(match.group(1)) if match else 0

            if score >= threshold:
                scored_images.append((path, score))

            del inputs, generated_ids, generated_ids_trimmed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Sort descending by score
        scored_images.sort(key=lambda x: x[1], reverse=True)

        return [path for path, score in scored_images]
    

    @torch.no_grad()
    def generate_answer(self, query, images_paths, dual_modality=False):
        content = []
        for path in images_paths:
            content.append({"type": "image", "image": path})
            
        # RAG instruction
        rag_prompt = (
            f"You are a precise local image archive assistant.\n"
            f"Analyze the provided images to see if they match or answer the user's query.\n\n"
            
            f"RULES:\n"
            f"1. VISUAL IS DATA: Identify real-world objects, people, attributes (color, clothing, etc), scenes, and actions visible in the images as concrete facts.\n"
            f"2. DIRECT CONFIRMATION: Confirm what is present that matches the query. Do not say 'The image shows...'.\n"
            f"3. HONEST NEGATIVE: If none of the images match the query description at all, state clearly that the requested item/subject is not present.\n\n"
        )
        if dual_modality:
            rag_prompt += (
                f"4. NO ECHOING: Do not quote or repeat the exact phrasing of the user's query\n"
                f"5. GENERALIZTION: collapse the description down to its absolute core subject (the primary object and its color). Do not list specific backgrounds, actions, or secondary details in your negative response (e.g., instead of saying 'None of the images match the description of a green frog sitting on a gray leaf with its eyes wide open', state simply: 'None of the images match to a green frog').\n"
                f"6.NATURAL LANGUAGE ONLY: Speak naturally as if you are looking at the actual photos. NEVER use technical, meta, or text-processing language. BANNED WORDS: 'description', 'described', 'query', 'text', 'prompt', 'image matches', 'criteria'.\n"
            )

        rag_prompt += (
            f"\nUser Query: '{query}'\n\n"
            f"Answer:"
        )
        content.append({"type": "text", "text": rag_prompt})
        
        # package into standard qwen message format
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # process inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # generate answer 
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        rag_answer = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        
        # clean up cuda memory
        del inputs, generated_ids, generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return rag_answer