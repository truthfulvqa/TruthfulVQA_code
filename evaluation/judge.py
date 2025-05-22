import argparse
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import json
import os
import re
from PIL import Image
from vllm.utils import device_count

SYSTEM_PROMPT = """
You are an expert in visual question answering. You need to critique and judge the two responses. Given an image, a question, two responses, you should output a critique and a label to indicate which response is better. You should also output a confidence score (a fractional number between 0 and 1) to indicate how sure you are about your judgement.

# Output Format
<critique>...</critique>
<label>...</label>
<confidence>...</confidence>
"""

def create_vllm_input(image: Image.Image, question: str, response_A: str, response_B: str, system_prompt: str = SYSTEM_PROMPT, processor: AutoProcessor = None) -> str:
    """Create a prompt using the exact template format from training."""
    
    prompt = [
            {'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]},
            {'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': f'[[Question]]\n{question}\n[[Response A]]\n{response_A}\n[[Response B]]\n{response_B}'},
            ]
            }
        ]
    
    processed_prompt = processor.apply_chat_template(prompt, add_generation_prompt=True)
    
    vllm_input = {
        "prompt": processed_prompt,
        "multi_modal_data": {"image": image}
    }

    return vllm_input

def load_image(image_path: str) -> Image.Image:
    image = Image.open(image_path)
    image = image.convert("RGB")
    return image

def extract_label(text):
    # Extract the label enclosed in <label> and </label>
    match = re.search(r'<label>([A-D])</label>', text)
    if match:
        return match.group(1)
    return None

def extract_confidence(text):
    # Extract the confidence enclosed in <confidence> and </confidence>
    match = re.search(r'<confidence>([0-9.]+)</confidence>', text)
    if match:
        try:
            return float(match.group(1))
        except:
            return match.group(1)
    return None

def process_batch(llm, batch_inputs, sampling_params):
    """Process a batch of inputs using the LLM."""
    outputs = llm.generate(prompts=batch_inputs, sampling_params=sampling_params)
    return outputs

def main():
    # add argument for pairwise_test_input_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise_test_input_path", type=str)
    parser.add_argument("--output_suffix", type=str, help="Suffix to add to output filename")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for processing")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    args = parser.parse_args()

    output_dir = args.output_dir

    pairwise_test_input_path = args.pairwise_test_input_path
    base_output_name = os.path.basename(pairwise_test_input_path).replace(".json", "-output")
    pairwise_test_output_path = os.path.join(output_dir, f"{base_output_name}{args.output_suffix}.json")

    # Load input data
    with open(pairwise_test_input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    
    model_name = "path/to/model"
    parallel_size = device_count()

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    vllm_inputs = [create_vllm_input(image=load_image(d['image_path']), question=d['question'], response_A=d['response_A']['text'], response_B=d['response_B']['text'], system_prompt=SYSTEM_PROMPT, processor=processor) for d in data]

    # Initialize model and tokenizer
    sampling_params = SamplingParams(
        temperature=0.1, 
        top_p=0.95,
        max_tokens=2048
    )

    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        tensor_parallel_size=parallel_size,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 1, "audio": 0, "video": 0},
        trust_remote_code=True,
    )
    
    # Process in batches
    batch_size = args.batch_size
    final_output = []
    
    for i in range(0, len(vllm_inputs), batch_size):
        batch = vllm_inputs[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(vllm_inputs) + batch_size - 1)//batch_size}")
        
        # Process the batch
        batch_outputs = process_batch(llm, batch, sampling_params)
        
        # Process results for this batch
        for j, output in enumerate(batch_outputs):
            idx = i + j
            item = data[idx]
            generated_text = output.outputs[0].text
            item['judge'] = generated_text.strip()
            item['label'] = extract_label(item['judge'])
            item['confidence'] = extract_confidence(item['judge'])
            final_output.append(item)
            
        # Save intermediate results after each batch
        with open(pairwise_test_output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4)
        
        print(f"Results saved to {pairwise_test_output_path}")

if __name__ == "__main__":
    main()
