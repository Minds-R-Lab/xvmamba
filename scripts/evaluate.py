#!/usr/bin/env python
"""
Evaluation Script for VMamba Controllability Analysis

This script provides a simple interface to run comprehensive evaluation
on trained models.

Usage:
    python scripts/evaluate.py \
        --checkpoint ./checkpoints/bloodmnist/best_model.pth \
        --dataset bloodmnist \
        --num_samples 50
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the comprehensive evaluation
from evaluation.comprehensive_evaluation import main as comprehensive_main
from evaluation.run_evaluation import main as faithfulness_main


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VMamba Controllability",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name (bloodmnist, dermamnist, octmnist, pneumoniamnist)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=50,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--mode", type=str, default="comprehensive",
        choices=["comprehensive", "faithfulness", "both"],
        help="Evaluation mode"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = f"./results/{args.dataset}"
    
    print("=" * 70)
    print("VMAMBA CONTROLLABILITY EVALUATION")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Samples:    {args.num_samples}")
    print(f"  Mode:       {args.mode}")
    print(f"  Output:     {args.output_dir}")
    print("=" * 70)
    
    # Construct arguments for the evaluation scripts
    eval_args = [
        "--checkpoint", args.checkpoint,
        "--dataset", args.dataset,
        "--num_samples", str(args.num_samples),
        "--output_dir", args.output_dir,
        "--device", args.device,
    ]
    
    if args.mode in ["comprehensive", "both"]:
        print("\n>>> Running Comprehensive Evaluation...")
        sys.argv = ["comprehensive_evaluation.py"] + eval_args
        comprehensive_main()
    
    if args.mode in ["faithfulness", "both"]:
        print("\n>>> Running Faithfulness Evaluation...")
        sys.argv = ["run_evaluation.py"] + eval_args
        faithfulness_main()
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
