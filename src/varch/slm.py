import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

class SLM:
    def __init__(self, device):
        print("loading slm...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=quantization_config,
            device_map=device
        )
        print("slm loaded successfuly")

    def fuse_queries(self, image_as_text, query):

        system_prompt = (
            "You are a deterministic text feature blending engine for an image retrieval database.\n"
            "Your job is to take a base description of an image and modify it according to the user's search query.\n\n"
            "RULES:\n"
            "1. FEATURE REPLACEMENT: Overwrite any specific objects, types, categories, or styles in the base caption with the new concepts requested by the user.\n"
            "2. EXCLUDE CONFLICTS: Do not include any old traits, brands, or descriptions from the base caption that contradict the user's modifications.\n"
            "3. STRIP SEARCH INTENT: Ignore conversational phrases like 'is there an image like this but'. Extract only the concrete physical changes.\n"
            "4. OUTPUT FORMAT: Output ONLY the final modified sentence. No introductions, explanations, or commentary."
        )

        user_content = (
            f"Base Caption: \"{image_as_text}\"\n"
            f"User Query: \"{query}\"\n\n"
            f"Modified Caption:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        fused_query = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
        return fused_query
