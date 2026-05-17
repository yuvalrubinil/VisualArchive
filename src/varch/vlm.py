import torch
from qwen_vl_utils import process_vision_info
from transformers import (AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration)

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

class VLM:

    def __init__(self, device):
        print("loading vlm...")
        self.device = device
        self.processor = AutoProcessor.from_pretrained(MODEL)

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
        )

    @torch.no_grad()
    def generate_answer(self, query, images_paths, scores):
        # 1. Build a single content list containing ALL retrieved images and the final RAG prompt
        content = []
        
        # Append all images into the context payload
        for path in images_paths:
            content.append({"type": "image", "image": path})
            
        # Append the strict RAG instruction text block at the end of the content array
        rag_prompt = (
            f"You are a precise Retrieval-Augmented Generation (RAG) assistant.\n"
            f"Answer the user's query using ONLY the factual data, text, charts, or visual information "
            f"visible across the provided images. Do not give generic descriptions of the images; "
            f"directly synthesize an answer to the query.\n\n"
            f"User Query: '{query}'\n\n"
            f"Answer:"
        )
        content.append({"type": "text", "text": rag_prompt})
        
        # 2. Package into standard Qwen message format
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # 3. Process inputs for the entire batch of images at once
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # 4. Generate the single comprehensive answer (increased max_new_tokens for a complete response)
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
        
        # Clean up CUDA memory
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        
        return rag_answer
    
    