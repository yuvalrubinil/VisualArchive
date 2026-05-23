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

    @torch.no_grad()
    def extract_augmentation(self, image_as_text, query):
        """Extracts the augmentation asked by the user in the textual query if any"""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise image retrieval and editing analyzer.\n"
                    "Your job is to determine if a query requests a specific programmatic or generative "
                    "augmentation compared to the original image description.\n\n"
                    "RULES:\n"
                    "1. If the query asks for 'similar' images, alternative compositions, or exact matches with NO explicit modifications, output exactly: NONE\n"
                    "2. If an augmentation is requested (color shifts, orientation changes, additions, or omissions), "
                    "output a single, clear, direct instruction describing exactly what to change, add, or remove from the image description."
                )
            },
            # example 1: semantic similarity / asset retrieval -> NONE
            {
                "role": "user",
                "content": "Reference Image Description: A knife lies next to a mug on a table.\nUser Query: any similar images?"
            },
            {
                "role": "assistant",
                "content": "NONE"
            },
            # example 2: target-bound color change
            {
                "role": "user",
                "content": "Reference Image Description: A brown bear stands on grass.\nUser Query: any images of this animal but white?"
            },
            {
                "role": "assistant",
                "content": "change the color of the bear to white"
            },
            # example 3: scene-wide color change / grayscale
            {
                "role": "user",
                "content": "Reference Image Description: A yellow school bus on a city street.\nUser Query: are there similar images but in black and white?"
            },
            {
                "role": "assistant",
                "content": "make the color of the scene black and white"
            },
            # example 4: orientation shift
            {
                "role": "user",
                "content": "Reference Image Description: A stop sign on a street corner.\nUser Query: is there a photo like this but with a sign flipped horizontally?"
            },
            {
                "role": "assistant",
                "content": "flip the stop sign horizontally"
            },
            # example 5: adding elements
            {
                "role": "user",
                "content": "Reference Image Description: A clean view of a highway from a dashboard camera.\nUser Query: any simialr pictures but in heavy rain?"
            },
            {
                "role": "assistant",
                "content": "add heavy rain to the scene"
            },
            # example 6: omitting Elements
            {
                "role": "user",
                "content": "Reference Image Description: A person walking a dog on a sidewalk.\nUser Query: is there a similar image without the dog?"
            },
            {
                "role": "assistant",
                "content": "remove the dog"
            },
            # isolated runtime block to guarantee independent evaluation
            {
                "role": "user",
                "content": (
                    f"Evaluate the following data context independently and generate the edit instruction if needed.\n"
                    f"<data_context>\n"
                    f"Reference Image Description: {image_as_text}\n"
                    f"User Query: {query}\n"
                    f"</data_context>\n"
                    f"Result:"
                )
            }
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=25, 
            do_sample=False,
            temperature=0.0
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()


    @torch.no_grad()
    def apply_augmentation(self, image_as_text, augmentation):
        """Changes the image description according to the augmentation"""

        system_prompt = (
            "You are a deterministic text editing engine for an image retrieval database.\n"
            "Your job is to modify a base image description based on a specific, direct editing command.\n\n"
            "UNIVERSAL LOGIC:\n"
            "1. Apply the modification instruction precisely to the target subject mentioned in the command.\n"
            "2. Keep the overall structure and detail of the base description intact, only updating or adding the specific features requested.\n"
            "3. STRICT FORMAT: Output ONLY the final modified sentence. No labels, no quotes, no introductions, no explanations."
        )

        user_content = (
            f"Base Caption: \"{image_as_text}\"\n"
            f"Modification Command: \"{augmentation}\"\n\n"
            f"Modified Caption:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            
            # example 1: color shift
            {
                "role": "user", 
                "content": "Base Caption: \"A brown bear stands on grass.\"\nModification Command: \"change the color of the bear to white\"\n\nModified Caption:"
            },
            {"role": "assistant", "content": "A white bear stands on grass."},
            
            # example 2: scene-wide effect / environment addition
            {
                "role": "user", 
                "content": "Base Caption: \"A clean view of a highway from a dashboard camera.\"\nModification Command: \"add heavy rain to the scene\"\n\nModified Caption:"
            },
            {"role": "assistant", "content": "A view of a highway from a dashboard camera in heavy rain."},
            
            # example 3: omission / removal
            {
                "role": "user", 
                "content": "Base Caption: \"A person walking a dog on a sidewalk.\"\nModification Command: \"remove the dog\"\n\nModified Caption:"
            },
            {"role": "assistant", "content": "A person walking on a sidewalk."},
            
            # example 4: orientation / transformation
            {
                "role": "user", 
                "content": "Base Caption: \"A red stop sign stands prominently on a pole near a street corner.\"\nModification Command: \"flip the stop sign horizontally\"\n\nModified Caption:"
            },
            {"role": "assistant", "content": "A horizontally flipped red stop sign stands prominently on a pole near a street corner."},
            
            # runtime context
            {"role": "user", "content": user_content}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=45, 
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        
        augmented_query = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
        return augmented_query
    