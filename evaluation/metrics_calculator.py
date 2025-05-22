import json
import numpy as np
from collections import defaultdict
from typing import Dict

def compute_metrics(json_file: str) -> Dict:
    """
    Compute various accuracy metrics from the JSON data.
    
    Args:
        json_file: Path to the JSON file containing the results
        
    Returns:
        Dictionary containing all computed metrics
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Initialize counters
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    subcategory_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    level_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    total_correct = 0
    total_questions = 0
    
    # Process each question
    for item in data:
        category = item['category']
        subcategory = item['subcategory']
        level = item['level']
        is_correct = item['is_correct']
        
        # Update category stats
        category_stats[category]['total'] += 1
        if is_correct:
            category_stats[category]['correct'] += 1
        
        # Update subcategory stats
        subcategory_stats[subcategory]['total'] += 1
        if is_correct:
            subcategory_stats[subcategory]['correct'] += 1
        
        # Update level stats
        level_stats[level]['total'] += 1
        if is_correct:
            level_stats[level]['correct'] += 1
        
        # Update overall stats
        total_questions += 1
        if is_correct:
            total_correct += 1
    
    # Calculate accuracies
    results = {
        'overall_accuracy': total_correct / total_questions if total_questions > 0 else 0,
        'category_accuracies': {},
        'subcategory_accuracies': {},
        'level_accuracies': {},
        'level_variance': 0,
        'cai': 0  
    }
    
    # Calculate category accuracies
    for category, stats in category_stats.items():
        results['category_accuracies'][category] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    
    # Calculate subcategory accuracies
    for subcategory, stats in subcategory_stats.items():
        results['subcategory_accuracies'][subcategory] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    
    # Calculate level accuracies and prepare for variance calculation
    level_accuracies = []
    for level, stats in level_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        results['level_accuracies'][level] = accuracy
        level_accuracies.append(accuracy)
    
    # Calculate variance of level accuracies
    if len(level_accuracies) > 1:
        results['level_variance'] = np.var(level_accuracies)
    
    # Calculate CAI 
    if all(level in results['level_accuracies'] for level in [1, 2, 3]):
        level1 = results['level_accuracies'][1]
        level2 = results['level_accuracies'][2]
        level3 = results['level_accuracies'][3]
        
        if level1 > 0:
            cai_term_1 = (level1 - level2) / level1
        else:
            cai_term_1 = 0
        if level2 > 0:
            cai_term_2 = (level3 - level2) / level2
        else:
            cai_term_2 = 0

        cai = cai_term_1 + cai_term_2
        results['cai'] = cai
    
    return results

def save_metrics(metrics: Dict, output_file: str):
    """
    Save metrics to a JSON file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4) 