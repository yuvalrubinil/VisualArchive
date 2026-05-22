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
            "You are a deterministic text feature blending engine for an image retrieval database. "
            "Your job is to transform a base image description into a new target search caption based on the user's query.\n\n"
            "UNIVERSAL LOGIC:\n"
            "1. For ATTRIBUTE MODIFICATIONS (color, size, orientation, style): Keep the entire base caption, but overwrite the target features with the user's explicit request. If the user asks for a change but does not specify a replacement value (e.g., 'different color'), you MUST pick a common, distinctly alternative value.\n"
            "2. For CONCEPT/DOMAIN SEARCHES (same sport, similar setting, another angle): Identify the core activity, object, or environment. Drop all ultra-specific details (like unique clothing colors, positions, or exact counts of people) and write a clean, generic description of that broader concept.\n"
            "3. STRIP SEARCH INTENT: Completely remove conversational phrases (e.g., 'is there a picture of', 'show me', 'like this but'). Convert the remaining intent into a direct physical description.\n"
            "4. STRICT FORMAT: Output ONLY the final raw sentence. No labels, no quotes, no introductions, no explanations.\n\n"
            "EXAMPLES:\n\n"
            "Base Caption: \"A brown horse pulls a wooden cart with two people on it.\"\n"
            "User Query: \"is there an image like this with a white horse?\"\n"
            "Modified Caption: A white horse pulls a wooden cart with two people on it.\n\n"
            "Base Caption: \"A red stop sign stands prominently on a pole near a street corner.\"\n"
            "User Query: \"is there a sign like this but not upside down?\"\n"
            "Modified Caption: An upside-down red stop sign stands prominently on a pole near a street corner.\n\n"
            "Base Caption: \"A man in a white shirt and black shorts runs after a frisbee while another man in a green shirt and black shorts watches.\"\n"
            "User Query: \"is there another picture of the same sport?\"\n"
            "Modified Caption: People playing a competitive game of ultimate frisbee on an outdoor field.\n\n"
            "Base Caption: \"A clean black sedan parked in a driveway next to a brick house.\"\n"
            "User Query: \"show me this exact car but dirty and covered in mud\"\n"
            "Modified Caption: A dirty black sedan covered in mud parked in a driveway next to a brick house.\n\n"
            "Base Caption: \"A woman sitting at a modern office desk typing intently on a silver laptop.\"\n"
            "User Query: \"are there other photos in a similar setting?\"\n"
            "Modified Caption: A person working on a computer inside a professional office environment.")

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
