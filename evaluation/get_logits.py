import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import json
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mulberry_prompt = """Question: {question}

Options:
{options_text}

Think step by step and provide the final answer in the following format: 'The final answer is: X' where X is the letter (A, B, C, or D). 

Format your response with the following sections, separated by ###:
### Image Description:
### Rationales:
### Let's think step by step.
### Step 1:
### Step 2:
...
### The final answer is:"""

class ModelDirectLogitsExtractor:
    def __init__(self, model_path: str = "moonshotai/Kimi-VL-A3B-Instruct", device: str = "cuda"):
        """
        Initialize the Qwen2.5-VL model for direct logits extraction at decision point
        
        Args:
            model_path: Path to the model or HuggingFace model name
            device: Device to run the model on
        """
        self.device = device
       
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Build multiple token variants for A, B, C, D (e.g., 'A', ' A', '\nA', ': A')
        # Only keep single-token encodings because we are selecting the next-token logits.
        self.option_token_variants = {}
        variant_prefixes_primary = ["", " ", "\n"]
        variant_prefixes_secondary = [": ", ") ", ". "]
        for option in ['A', 'B', 'C', 'D']:
            variant_token_ids = set()
            # Primary prefixes
            for prefix in variant_prefixes_primary:
                text = prefix + option
                ids = self.processor.tokenizer.encode(text, add_special_tokens=False)
                if len(ids) == 1:
                    variant_token_ids.add(ids[0])
            # Secondary prefixes that sometimes appear before the letter
            for prefix in variant_prefixes_secondary:
                text = prefix + option
                ids = self.processor.tokenizer.encode(text, add_special_tokens=False)
                if len(ids) == 1:
                    variant_token_ids.add(ids[0])
            # Fallback: ensure at least bare letter contributes
            if not variant_token_ids:
                ids = self.processor.tokenizer.encode(option, add_special_tokens=False)
                if len(ids) > 0:
                    variant_token_ids.add(ids[-1])
            self.option_token_variants[option] = sorted(variant_token_ids)

        logger.info(f"Option token variant IDs: {self.option_token_variants}")
        logger.info(f"Model vocabulary size: {self.processor.tokenizer.vocab_size}")
    
    def prepare_prompt(self, question: str, options: List[str], setting: int = 2) -> str:
        """
        Prepare the prompt based on the setting
        
        Args:
            question: The VQA question
            options: List of options (typically 4 options)
            setting: 1 for direct answer, 2 for reasoning then answer
            
        Returns:
            Formatted prompt string
        """
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        
        if setting == 1:
            # Direct answer setting
            prompt = f"""Question: {question}

Options:
{options_text}

Answer with just the letter (A, B, C, or D):"""
        
        elif setting == 2:
            # Reasoning then answer setting
            prompt = f"""Question: {question}

Options:
{options_text}
Think throughly and provide the final answer in the following format: 'The final answer is: X' where X is the letter (A, B, C, or D). 
DO NOT include content after the letter. 
Example output:
...
The final answer is A
"""
        
        else:
            raise ValueError("Setting must be 1 or 2")
        
        return prompt
    
    def extract_decision_point_logits(self, image: Image.Image, question: str, options: List[str], 
                                    setting: int = 1, return_top_k: int = 10, max_tokens: int = 512) -> Dict:
        """
        Extract logits for ALL options at the exact decision point
        
        Args:
            image: PIL Image
            question: VQA question
            options: List of answer options
            setting: 1 for direct, 2 for reasoning
            return_top_k: Number of top tokens to return for analysis
            max_tokens: Maximum tokens for generation (only used in setting=2)
            
        Returns:
            Dictionary with comprehensive logits analysis
        """
        prompt = self.prepare_prompt(question, options, setting=setting)
        
        # Prepare inputs
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process image and text
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        decision_point_logits = None
        generated_text = None
        original_response = None
        
        with torch.no_grad():
            if setting == 1:
                # Setting 1: Direct answer - get logits at decision point
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get logits for the next token position (where answer should be generated)
                decision_point_logits = logits[0, -1, :]  # Shape: [vocab_size]
                
                # For setting 1, also generate the actual response to save
                try:
                    response_outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,  # Short generation for direct answer
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                    response_ids = response_outputs[0][inputs.input_ids.shape[1]:]
                    original_response = self.processor.tokenizer.decode(response_ids, skip_special_tokens=True)
                except Exception as e:
                    logger.warning(f"Could not generate original response for setting 1: {e}")
                    original_response = None
                
            elif setting == 2:
                # Setting 2: Reasoning then answer - generate and extract from answer position
                generation_outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )
                
                # Decode generated text to find answer position
                generated_ids = generation_outputs.sequences[0][inputs.input_ids.shape[1]:]
                generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
                original_response = generated_text  # Save the complete original response
                #print(original_response)
                
                # Find answer in generated text using robust regexes
                answer_match = re.search(r'(?:The\s+final\s+answer\s+is|The\s+answer\s+is)\s*:?[\s\n]*([A-D])', generated_text, re.IGNORECASE)

                if not answer_match:
                   answer_match = re.search(r'◁/think▷\s*([A-D])', generated_text)
                
                if answer_match and len(generation_outputs.scores) > 0:
                    # Method 1: Try to find the exact position where the answer letter was generated
                    predicted_answer = answer_match.group(1)
                    answer_token_id_set = set(self.option_token_variants.get(predicted_answer, []))
                    
                    # Look for answer token in generated sequence
                    answer_positions = []
                    for i, token_id in enumerate(generated_ids):
                        
                        if token_id in answer_token_id_set:
                            answer_positions.append(i)
                    
                    # Use the last occurrence of the answer token (most likely the final answer)
                    if answer_positions and answer_positions[-1] < len(generation_outputs.scores):
                        answer_pos = answer_positions[-1]
                        decision_point_logits = generation_outputs.scores[answer_pos][0]
                        decision_point_pos = answer_pos
                        #logger.info(f"Found answer '{predicted_answer}' at position {answer_pos}")
                    else:
                        # Method 2: Look for position right before </answer>
                        # Find </answer> token positions
                        close_answer_tokens = self.processor.tokenizer.encode("</answer>", add_special_tokens=False)
                        close_token_positions = []
                        
                        for close_token in close_answer_tokens:
                            for i, token_id in enumerate(generated_ids):
                                if token_id == close_token:
                                    close_token_positions.append(i)
                        
                        if close_token_positions:
                            # Use position before the closing tag
                            pos_before_close = max(0, min(close_token_positions) - 1)
                            if pos_before_close < len(generation_outputs.scores):
                                decision_point_logits = generation_outputs.scores[pos_before_close][0]
                                decision_point_pos = pos_before_close
                                #logger.info(f"Using position {pos_before_close} before </answer> tag")
                        
                        if decision_point_logits is None:
                            # Method 3: Use the last few positions and find where answer token has highest logit
                            last_positions = min(5, len(generation_outputs.scores))
                            best_pos = -1
                            best_logit = float('-inf')
                            
                            for pos in range(len(generation_outputs.scores) - last_positions, len(generation_outputs.scores)):
                                if pos >= 0:
                                    pos_logits = generation_outputs.scores[pos][0]
                                    # Consider all variant token IDs for the predicted answer
                                    if answer_token_id_set:
                                        variant_idx = torch.tensor(list(answer_token_id_set), device=pos_logits.device)
                                        variant_logits = pos_logits[variant_idx]
                                        candidate_logit = torch.max(variant_logits).item()
                                    else:
                                        candidate_logit = float('-inf')
                                    if candidate_logit > best_logit:
                                        best_logit = candidate_logit
                                        best_pos = pos
                            
                            if best_pos >= 0:
                                decision_point_logits = generation_outputs.scores[best_pos][0]
                                decision_point_pos = best_pos
                                # Print best position, chosen token id and token text among variants
                                # try:
                                #     pos_logits = generation_outputs.scores[best_pos][0]
                                    # variant_idx = torch.tensor(list(answer_token_id_set), device=pos_logits.device)
                                    # variant_logits = pos_logits[variant_idx]
                                    # best_variant_local_idx = int(torch.argmax(variant_logits).item())
                                    # best_variant_token_id = int(variant_idx[best_variant_local_idx].item())
                                    # best_variant_token_text = self.processor.tokenizer.decode([best_variant_token_id])
                                    #logger.info(
                                    #    f"Best position {best_pos} | token_id={best_variant_token_id} | token='{best_variant_token_text}' | logit={best_logit:.3f}"
                                    #)
                                # except Exception as _e:
                                #     logger.info(f"Best position {best_pos} | logit={best_logit:.3f} (token decode failed: {_e})")
                
                # Final fallback: choose best position by scanning last K steps across all option variants
                if decision_point_logits is None and len(generation_outputs.scores) > 0:
                    k = min(8, len(generation_outputs.scores))
                    start_pos = len(generation_outputs.scores) - k
                    global_best_pos = -1
                    global_best_logit = float('-inf')
                    global_best_option = None
                    global_best_token_id = None
                    for pos in range(start_pos, len(generation_outputs.scores)):
                        pos_logits = generation_outputs.scores[pos][0]
                        for option, token_ids in self.option_token_variants.items():
                            if not token_ids:
                                continue
                            idx_tensor = torch.tensor(token_ids, device=pos_logits.device)
                            variant_logits = pos_logits[idx_tensor]
                            best_idx = int(torch.argmax(variant_logits).item())
                            best_logit_here = variant_logits[best_idx].item()
                            if best_logit_here > global_best_logit:
                                global_best_logit = best_logit_here
                                global_best_pos = pos
                                global_best_option = option
                                global_best_token_id = int(idx_tensor[best_idx].item())

                    if global_best_pos >= 0:
                        decision_point_logits = generation_outputs.scores[global_best_pos][0]
                        decision_point_pos = global_best_pos
                        try:
                            token_text = self.processor.tokenizer.decode([global_best_token_id]) if global_best_token_id is not None else "<N/A>"
                            logger.warning(
                                f"Fallback(best-of-window): pos={global_best_pos}, option={global_best_option}, token_id={global_best_token_id}, token='{token_text}', logit={global_best_logit:.3f}"
                            )
                        except Exception as _e:
                            logger.warning(
                                f"Fallback(best-of-window): pos={global_best_pos}, option={global_best_option}, token_id={global_best_token_id}, logit={global_best_logit:.3f} (token decode failed: {_e})"
                            )
                    else:
                        # As a last resort, keep previous last-token behavior and log last 5 decoded tokens
                        decision_point_logits = generation_outputs.scores[-1][0]
                        decision_point_pos = len(generation_outputs.scores) - 1
                        try:
                            k2 = min(5, len(generation_outputs.scores))
                            start2 = len(generation_outputs.scores) - k2
                            last_tokens_info = []
                            for pos in range(start2, len(generation_outputs.scores)):
                                tok_id = int(generated_ids[pos].item()) if pos < len(generated_ids) else None
                                tok_text = self.processor.tokenizer.decode([tok_id]) if tok_id is not None else "<N/A>"
                                last_tokens_info.append({
                                    'pos': pos,
                                    'token_id': tok_id,
                                    'token': tok_text
                                })
                            logger.warning("Could not find answer position, using last token logits. Last positions: %s", last_tokens_info)
                        except Exception as _e:
                            logger.warning("Could not find answer position, using last token logits (token decode failed: %s)", _e)
                
            else:
                raise ValueError("Setting must be 1 or 2")
        
        if decision_point_logits is None:
            raise ValueError("Could not extract decision point logits")
        
        # Extract logits for A, B, C, D using multiple token variants
        # Choose the single variant with the highest logit for each option.
        option_logits = {}
        option_chosen_token_id = {}
        option_variant_details = {}
        for option, token_ids in self.option_token_variants.items():
            if not token_ids:
                continue
            idx_tensor = torch.tensor(token_ids, device=decision_point_logits.device)
            variant_logits = decision_point_logits[idx_tensor]
            best_idx = int(torch.argmax(variant_logits).item())
            best_token_id = token_ids[best_idx]
            best_logit = variant_logits[best_idx].item()
            option_logits[option] = best_logit
            option_chosen_token_id[option] = best_token_id
            option_variant_details[option] = {
                'variant_token_ids': token_ids,
                'variant_logits': [decision_point_logits[i].item() for i in token_ids],
                'chosen_token_id': best_token_id,
                'chosen_logit': best_logit
            }
        
        # Calculate probabilities in different ways
        
        # 1. Probabilities among A, B, C, D only (constrained to these 4 options)
        option_logit_tensor = torch.tensor([option_logits[opt] for opt in ['A', 'B', 'C', 'D']])
        option_probs_constrained = F.softmax(option_logit_tensor, dim=0)
        option_probs_constrained_dict = {
            option: option_probs_constrained[i].item() 
            for i, option in enumerate(['A', 'B', 'C', 'D'])
        }
        
        # 2. Probabilities in full vocabulary context
        full_vocab_probs = F.softmax(decision_point_logits, dim=0)
        option_probs_full_vocab = {}
        for option, token_ids in self.option_token_variants.items():
            chosen_id = option_chosen_token_id.get(option)
            option_probs_full_vocab[option] = full_vocab_probs[chosen_id].item() if chosen_id is not None else 0.0
        
        # 3. Get top-k tokens for context
        top_k_values, top_k_indices = torch.topk(decision_point_logits, return_top_k)
        top_k_tokens = []
        for i in range(return_top_k):
            token_id = top_k_indices[i].item()
            token_text = self.processor.tokenizer.decode([token_id])
            logit_value = top_k_values[i].item()
            prob_value = full_vocab_probs[token_id].item()
            top_k_tokens.append({
                'token': token_text,
                'token_id': token_id,
                'logit': logit_value,
                'probability': prob_value,
                'rank': i + 1
            })
        
        # 4. Calculate confidence metrics
        max_option_prob = max(option_probs_constrained_dict.values())
        entropy_constrained = -sum(
            p * torch.log(torch.tensor(p + 1e-10)).item() 
            for p in option_probs_constrained_dict.values()
        )
        
        # 5. Check ranking of A, B, C, D in full vocabulary
        option_rankings = {}
        sorted_indices = torch.argsort(decision_point_logits, descending=True)
        for option, token_ids in self.option_token_variants.items():
            chosen_id = option_chosen_token_id.get(option)
            if chosen_id is None:
                option_rankings[option] = None
            else:
                pos = (sorted_indices == chosen_id).nonzero(as_tuple=True)
                option_rankings[option] = (pos[0].item() + 1) if len(pos[0]) > 0 else None
        
        result = {
            # Core results
            'option_logits': option_logits,
            'option_probs_constrained': option_probs_constrained_dict,  # Among A,B,C,D only
            'option_probs_full_vocab': option_probs_full_vocab,        # In full vocabulary
            'option_rankings_in_vocab': option_rankings,               # Rank among all tokens
            
            # Predictions
            'predicted_answer_constrained': max(option_probs_constrained_dict, key=option_probs_constrained_dict.get),
            'predicted_answer_full_vocab': max(option_probs_full_vocab, key=option_probs_full_vocab.get),
            
            # Confidence metrics
            'confidence_metrics': {
                'max_option_probability': max_option_prob,
                'entropy_among_options': entropy_constrained,
                'uncertainty': 1 - max_option_prob,
                'confidence_ratio': max_option_prob / (1 - max_option_prob + 1e-10)
            },
            
            # Context information
            'top_k_tokens': top_k_tokens,
            'vocab_size': decision_point_logits.shape[0],
            'total_probability_mass_in_options': sum(option_probs_full_vocab.values()),
            
            # Raw data for further analysis
            'raw_decision_logits_shape': decision_point_logits.shape,
            'prompt_used': prompt,
            'setting_used': setting,
            
            # Original model response (IMPORTANT: Always saved)
            'original_response': original_response
        }
        
        # Add generated text for setting 2
        if setting == 2:
            result['generated_text'] = generated_text  # Duplicate for backward compatibility
            result['max_tokens_used'] = max_tokens
            result['option_variant_details'] = option_variant_details
            
            # Extract reasoning (text before <answer>)
            if generated_text:
                reasoning_match = re.search(r'^(.*?)<answer>', generated_text, re.DOTALL)
                if reasoning_match:
                    result['reasoning'] = reasoning_match.group(1).strip()
                else:
                    result['reasoning'] = generated_text  # Fallback if no <answer> tag found
            else:
                result['reasoning'] = None
        
        elif setting == 1:
            # For setting 1, add generation info
            result['max_tokens_used'] = 50  # We used 50 tokens for direct answer generation
        return result
    
    def analyze_option_preferences(self, result: Dict) -> Dict:
        """
        Analyze the option preferences and provide interpretable insights
        
        Args:
            result: Output from extract_decision_point_logits
            
        Returns:
            Dictionary with analysis insights
        """
        option_logits = result['option_logits']
        option_probs = result['option_probs_constrained']
        rankings = result['option_rankings_in_vocab']
        
        # Find logit differences
        logit_values = [option_logits[opt] for opt in ['A', 'B', 'C', 'D']]
        max_logit = max(logit_values)
        min_logit = min(logit_values)
        logit_spread = max_logit - min_logit
        
        # Analyze confidence
        confidence = result['confidence_metrics']['max_option_probability']
        entropy = result['confidence_metrics']['entropy_among_options']
        
        # Categorize confidence level
        if confidence > 0.7:
            confidence_level = "High"
        elif confidence > 0.4:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Find option pairs with close logits
        close_pairs = []
        for i, opt1 in enumerate(['A', 'B', 'C', 'D']):
            for j, opt2 in enumerate(['A', 'B', 'C', 'D']):
                if i < j:
                    logit_diff = abs(option_logits[opt1] - option_logits[opt2])
                    if logit_diff < 1.0:  # Close logits
                        close_pairs.append({
                            'options': f"{opt1}-{opt2}",
                            'logit_difference': logit_diff,
                            'prob_difference': abs(option_probs[opt1] - option_probs[opt2])
                        })
        
        return {
            'summary': {
                'most_preferred': max(option_probs, key=option_probs.get),
                'least_preferred': min(option_probs, key=option_probs.get),
                'confidence_level': confidence_level,
                'is_uncertain': confidence < 0.5
            },
            'logit_analysis': {
                'logit_spread': logit_spread,
                'max_logit_value': max_logit,
                'min_logit_value': min_logit,
                'close_competitions': close_pairs
            },
            'ranking_analysis': {
                'best_vocab_rank': min(rankings.values()),
                'worst_vocab_rank': max(rankings.values()),
                'all_options_in_top_100': all(rank <= 100 for rank in rankings.values())
            },
            'detailed_breakdown': [
                {
                    'option': opt,
                    'logit': option_logits[opt],
                    'probability': option_probs[opt],
                    'vocab_rank': rankings[opt],
                    'confidence_pct': option_probs[opt] * 100
                }
                for opt in ['A', 'B', 'C', 'D']
            ]
        }
    
    def process_vqa_sample(self, image_path: str, question: str, options: List[str], 
                          setting: int = 1, analyze: bool = True, max_tokens: int = 512) -> Dict:
        """
        Process a single VQA sample and extract decision point logits
        
        Args:
            image_path: Path to the image file
            question: VQA question
            options: List of answer options
            setting: 1 for direct, 2 for reasoning
            analyze: Whether to include detailed analysis
            max_tokens: Maximum tokens for generation (only used in setting=2)
            
        Returns:
            Dictionary with logits and analysis
        """
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Extract logits at decision point
            result = self.extract_decision_point_logits(
                image, question, options, 
                setting=setting, max_tokens=max_tokens
            )
            
            if analyze:
                # Add detailed analysis
                analysis = self.analyze_option_preferences(result)
                result['analysis'] = analysis
            
            return result
                
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            return {'error': str(e)}


def main():
    """
    Example usage of the Direct Method (Improved) implementation
    """
    # Initialize the extractor
    extractor = ModelDirectLogitsExtractor()

    from tqdm import tqdm
    import json
    with open("./truthfulvqa.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # For testing, use a smaller subset
    #dataset = dataset[:12]
    results = []

    for item in tqdm(dataset, desc="Processing truthfulvqa dataset..."):
        image_path = item['image_path']
        question = item['question']
        options = item['options']
        
        # Process with setting=2 (reasoning) and max_tokens control
        result = extractor.process_vqa_sample(
            image_path, question, options, 
            setting=2,  # Reasoning mode
            analyze=False,  # Set to True if you want detailed analysis
            max_tokens=32768  # Control generation length
        )
        
        results.append({
            "case": item,
            "result": result
        })
        
        # Optional: Print progress for first few samples
        if len(results) <= 3 and 'error' not in result:
            print(f"\nSample {len(results)}:")
            print(f"Question: {question[:100]}...")
            if 'original_response' in result and result['original_response']:
                print(f"Original response: {result['original_response'][:200]}...")
            if 'reasoning' in result and result['reasoning']:
                print(f"Extracted reasoning: {result['reasoning'][:150]}...")
            print(f"Predicted answer: {result.get('predicted_answer_constrained', 'N/A')}")
            print(f"Confidence: {result.get('confidence_metrics', {}).get('max_option_probability', 0):.3f}")
            print("-" * 50)

    # Save results
    output_path = "./results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\nProcessed {len(results)} samples")
    print(f"Results saved to: {output_path}")
    
    # Print summary statistics
    successful_results = [r for r in results if 'error' not in r['result']]
    if successful_results:
        confidences = [r['result']['confidence_metrics']['max_option_probability'] 
                      for r in successful_results]
        avg_confidence = sum(confidences) / len(confidences)
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"High confidence samples (>0.7): {sum(1 for c in confidences if c > 0.7)}/{len(confidences)}")


if __name__ == "__main__":
    main()