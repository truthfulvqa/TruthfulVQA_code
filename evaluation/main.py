import argparse
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from evaluation.answer_extractor import process_responses
from metrics_calculator import compute_metrics, save_metrics

def setup_directories(output_dir: str) -> tuple[str, str]:
    """
    Setup output directories for extracted results and metrics.
    Returns tuple of (extract_dir, metrics_dir)
    """
    # Create base output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    extract_dir = output_path / "extracted"
    metrics_dir = output_path / "metrics"
    extract_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    
    return str(extract_dir), str(metrics_dir)

def ensure_dir_exists(file_path: str):
    """Ensure the directory of the given file path exists."""
    directory = os.path.dirname(file_path)
    if directory:  # Only create if there is a directory path
        os.makedirs(directory, exist_ok=True)

def display_metrics(metrics: dict, console: Console):
    """Display metrics in a formatted way using rich."""
    # Create overall results panel with all metrics
    overall_panel = Panel(
        f"Overall Accuracy: {metrics['overall_accuracy']:.2%}\n"
        f"Level Variance: {metrics['level_variance']:.4f}\n"
        f"CAI: {metrics['cai']:.4f}",
        title="Overall Results",
        border_style="green"
    )
    console.print(overall_panel)
    
    # Category accuracies
    category_table = Table(title="Category Accuracies", show_header=True, header_style="bold magenta")
    category_table.add_column("Category", style="cyan")
    category_table.add_column("Accuracy", justify="right", style="green")
    
    for category, accuracy in metrics['category_accuracies'].items():
        category_table.add_row(category, f"{accuracy:.2%}")
    
    console.print(category_table)
    
    # Subcategory accuracies
    subcategory_table = Table(title="Subcategory Accuracies", show_header=True, header_style="bold magenta")
    subcategory_table.add_column("Subcategory", style="cyan")
    subcategory_table.add_column("Accuracy", justify="right", style="green")
    
    for subcategory, accuracy in metrics['subcategory_accuracies'].items():
        subcategory_table.add_row(subcategory, f"{accuracy:.2%}")
    
    console.print(subcategory_table)
    
    # Level accuracies
    level_table = Table(title="Level Accuracies", show_header=True, header_style="bold magenta")
    level_table.add_column("Level", style="cyan")
    level_table.add_column("Accuracy", justify="right", style="green")
    
    for level, accuracy in metrics['level_accuracies'].items():
        level_table.add_row(str(level), f"{accuracy:.2%}")
    
    console.print(level_table)

def main():
    parser = argparse.ArgumentParser(description="TruthfulVQA Evaluation Tool")
    parser.add_argument("mode", choices=["extract_and_eval", "eval"], 
                       help="Mode of operation: extract_and_eval or eval")
    parser.add_argument("input_file", type=str, 
                       help="Path to input JSON file")
    parser.add_argument("--output-dir", type=str, 
                       default="results",
                       help="Base directory for output files")
    parser.add_argument("--save-metrics", type=str,
                       help="Path to save metrics JSON file. If not specified, metrics will not be saved.")
    parser.add_argument("--save-extracted", type=str,
                       help="Path to save extracted results JSON file. Only used in extract_and_eval mode.")
    
    args = parser.parse_args()
    console = Console()
    
    # Setup output directories
    extract_dir, metrics_dir = setup_directories(args.output_dir)
    
    if args.mode == "extract_and_eval":
        # Extract answers from raw responses
        console.print("[bold blue]Extracting answers from responses...[/]")
        
        # Create a temporary file for extraction if not saving
        if args.save_extracted:
            extract_output = args.save_extracted
            ensure_dir_exists(extract_output)
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
                extract_output = temp_file.name
        
        # Process responses
        process_responses(args.input_file, extract_output)
        if args.save_extracted:
            console.print(f"[green]Extracted results saved to: {extract_output}[/]")
        
        # Compute metrics
        console.print("[bold blue]Computing metrics...[/]")
        metrics = compute_metrics(extract_output)
        
        # Save metrics if path is specified
        if args.save_metrics:
            ensure_dir_exists(args.save_metrics)
            save_metrics(metrics, args.save_metrics)
            console.print(f"[green]Metrics saved to: {args.save_metrics}[/]")
        
        # Clean up temporary file if it was created
        if not args.save_extracted:
            os.unlink(extract_output)
        
    else:  # eval mode
        # Compute metrics directly from processed file
        console.print("[bold blue]Computing metrics...[/]")
        metrics = compute_metrics(args.input_file)
        
        # Save metrics if path is specified
        if args.save_metrics:
            ensure_dir_exists(args.save_metrics)
            save_metrics(metrics, args.save_metrics)
            console.print(f"[green]Metrics saved to: {args.save_metrics}[/]")
    
    # Display results
    console.print("\n[bold blue]Evaluation Results:[/]")
    display_metrics(metrics, console)

if __name__ == "__main__":
    main() 