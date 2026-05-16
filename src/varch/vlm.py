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
        # Accumulate lines in a list
        output_lines = ["\n--- Retrieved Sources & Descriptions (Ctrl+Click to open) ---"]
        
        for i, (path, score) in enumerate(zip(images_paths, scores)):
            content = [
                {"type": "image", "image": path},
                {
                    "type": "text", 
                    "text": f"The user searched for: '{query}'. Provide a very brief, 1-sentence description of what is visible in this specific image relative to the search."
                }
            ]
            
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

            generated_ids = self.model.generate(**inputs, max_new_tokens=40)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            img_description = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            # Append the formatted lines to our collector list
            output_lines.append(f"Image {i+1} [Score: {score:.4f}]: {path}")
            output_lines.append(f" └─ Description: {img_description}\n")

            # Clean memory at the end of each image iteration step
            del inputs, generated_ids, generated_ids_trimmed
            torch.cuda.empty_cache()

        output_lines.append("-----------------------------------------------------------------\n")
        
        # Join all parts with newlines and return the single composite string
        return "\n".join(output_lines)
    
    