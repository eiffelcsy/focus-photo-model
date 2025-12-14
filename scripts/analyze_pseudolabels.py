#!/usr/bin/env python3
"""
Analyze generated pseudolabels to check quality and distribution.

Usage:
    python scripts/analyze_pseudolabels.py --input data/ava/pseudolabels/pseudolabels.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_pseudolabels(filepath: str) -> List[Dict]:
    """Load pseudolabels from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_statistics(data: List[Dict]) -> pd.DataFrame:
    """Compute statistics for each criterion."""
    criteria = ['impact', 'style', 'composition', 'lighting', 'color_balance', 'ava_score']
    
    stats = {}
    for criterion in criteria:
        values = [item.get(criterion, 0) for item in data if criterion in item]
        
        if values:
            stats[criterion] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75),
            }
    
    return pd.DataFrame(stats).T


def compute_correlations(data: List[Dict]) -> pd.DataFrame:
    """Compute correlations between criteria."""
    criteria = ['impact', 'style', 'composition', 'lighting', 'color_balance', 'ava_score']
    
    # Create dataframe
    df_data = {criterion: [item.get(criterion, np.nan) for item in data] 
               for criterion in criteria}
    df = pd.DataFrame(df_data)
    
    # Compute correlations
    return df.corr()


def check_missing_data(data: List[Dict]) -> Dict:
    """Check for missing data in pseudolabels."""
    criteria = ['impact', 'style', 'composition', 'lighting', 'color_balance']
    
    missing = {
        'total_records': len(data),
        'missing_by_criterion': {}
    }
    
    for criterion in criteria:
        missing_count = sum(1 for item in data if criterion not in item)
        missing['missing_by_criterion'][criterion] = {
            'count': missing_count,
            'percentage': (missing_count / len(data) * 100) if data else 0
        }
    
    return missing


def analyze_score_alignment(data: List[Dict]) -> Dict:
    """Analyze how well pseudolabel scores align with AVA scores."""
    # Compute average of individual scores
    aligned_data = []
    
    for item in data:
        if all(criterion in item for criterion in ['impact', 'style', 'composition', 'lighting', 'color_balance', 'ava_score']):
            avg_score = (
                item['impact'] + item['style'] + item['composition'] + 
                item['lighting'] + item['color_balance']
            ) / 5
            
            aligned_data.append({
                'ava_score': item['ava_score'],
                'avg_pseudolabel': avg_score,
                'diff': avg_score - item['ava_score']
            })
    
    if not aligned_data:
        return {}
    
    diffs = [item['diff'] for item in aligned_data]
    
    return {
        'num_samples': len(aligned_data),
        'mean_difference': np.mean(diffs),
        'std_difference': np.std(diffs),
        'mae': np.mean(np.abs(diffs)),
        'correlation': np.corrcoef(
            [item['ava_score'] for item in aligned_data],
            [item['avg_pseudolabel'] for item in aligned_data]
        )[0, 1]
    }


def print_analysis(data: List[Dict]):
    """Print comprehensive analysis of pseudolabels."""
    print("=" * 70)
    print("PSEUDOLABEL ANALYSIS")
    print("=" * 70)
    print()
    
    # Basic info
    print(f"Total pseudolabels: {len(data)}")
    print()
    
    # Statistics
    print("-" * 70)
    print("SCORE STATISTICS")
    print("-" * 70)
    stats = compute_statistics(data)
    print(stats.to_string())
    print()
    
    # Correlations
    print("-" * 70)
    print("CORRELATIONS BETWEEN CRITERIA")
    print("-" * 70)
    corr = compute_correlations(data)
    print(corr.to_string())
    print()
    
    # Missing data
    print("-" * 70)
    print("MISSING DATA CHECK")
    print("-" * 70)
    missing = check_missing_data(data)
    print(f"Total records: {missing['total_records']}")
    print("\nMissing by criterion:")
    for criterion, info in missing['missing_by_criterion'].items():
        print(f"  {criterion:15s}: {info['count']:5d} ({info['percentage']:5.2f}%)")
    print()
    
    # Score alignment
    print("-" * 70)
    print("AVA SCORE ALIGNMENT")
    print("-" * 70)
    alignment = analyze_score_alignment(data)
    if alignment:
        print(f"Number of samples:      {alignment['num_samples']}")
        print(f"Mean difference:        {alignment['mean_difference']:6.3f}")
        print(f"Std difference:         {alignment['std_difference']:6.3f}")
        print(f"MAE:                    {alignment['mae']:6.3f}")
        print(f"Correlation with AVA:   {alignment['correlation']:6.3f}")
    else:
        print("No complete records for alignment analysis")
    print()
    
    # Sample pseudolabels
    print("-" * 70)
    print("SAMPLE PSEUDOLABELS (first 3)")
    print("-" * 70)
    for i, item in enumerate(data[:3]):
        print(f"\nSample {i+1}:")
        print(f"  Image ID:      {item.get('image_id', 'N/A')}")
        print(f"  AVA Score:     {item.get('ava_score', 'N/A'):.2f}")
        print(f"  Impact:        {item.get('impact', 'N/A'):.2f}")
        print(f"  Style:         {item.get('style', 'N/A'):.2f}")
        print(f"  Composition:   {item.get('composition', 'N/A'):.2f}")
        print(f"  Lighting:      {item.get('lighting', 'N/A'):.2f}")
        print(f"  Color Balance: {item.get('color_balance', 'N/A'):.2f}")
        
        # Calculate average
        if all(k in item for k in ['impact', 'style', 'composition', 'lighting', 'color_balance']):
            avg = (item['impact'] + item['style'] + item['composition'] + 
                   item['lighting'] + item['color_balance']) / 5
            print(f"  Average:       {avg:.2f}")
        
        if 'reasoning' in item:
            reasoning = item['reasoning'][:100] + "..." if len(item['reasoning']) > 100 else item['reasoning']
            print(f"  Reasoning:     {reasoning}")
    
    print()
    print("=" * 70)


def save_analysis_report(data: List[Dict], output_file: str):
    """Save analysis report to file."""
    with open(output_file, 'w') as f:
        # Statistics
        f.write("SCORE STATISTICS\n")
        f.write("=" * 70 + "\n")
        stats = compute_statistics(data)
        f.write(stats.to_string())
        f.write("\n\n")
        
        # Correlations
        f.write("CORRELATIONS\n")
        f.write("=" * 70 + "\n")
        corr = compute_correlations(data)
        f.write(corr.to_string())
        f.write("\n\n")
        
        # Missing data
        f.write("MISSING DATA\n")
        f.write("=" * 70 + "\n")
        missing = check_missing_data(data)
        f.write(f"Total records: {missing['total_records']}\n\n")
        f.write("Missing by criterion:\n")
        for criterion, info in missing['missing_by_criterion'].items():
            f.write(f"  {criterion:15s}: {info['count']:5d} ({info['percentage']:5.2f}%)\n")
        f.write("\n")
        
        # Alignment
        f.write("AVA SCORE ALIGNMENT\n")
        f.write("=" * 70 + "\n")
        alignment = analyze_score_alignment(data)
        if alignment:
            f.write(f"Number of samples:      {alignment['num_samples']}\n")
            f.write(f"Mean difference:        {alignment['mean_difference']:6.3f}\n")
            f.write(f"Std difference:         {alignment['std_difference']:6.3f}\n")
            f.write(f"MAE:                    {alignment['mae']:6.3f}\n")
            f.write(f"Correlation with AVA:   {alignment['correlation']:6.3f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze generated pseudolabels"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to pseudolabels JSON file"
    )
    
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Save analysis report to file"
    )
    
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Export pseudolabels to CSV format"
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Load data
    print(f"Loading pseudolabels from {args.input}...")
    data = load_pseudolabels(args.input)
    
    if not data:
        print("Error: No data found in input file")
        sys.exit(1)
    
    # Print analysis
    print_analysis(data)
    
    # Save report if requested
    if args.output_report:
        save_analysis_report(data, args.output_report)
        print(f"Analysis report saved to: {args.output_report}")
    
    # Export to CSV if requested
    if args.export_csv:
        # Create dataframe
        df_data = []
        for item in data:
            row = {
                'image_id': item.get('image_id', ''),
                'ava_score': item.get('ava_score', np.nan),
                'impact': item.get('impact', np.nan),
                'style': item.get('style', np.nan),
                'composition': item.get('composition', np.nan),
                'lighting': item.get('lighting', np.nan),
                'color_balance': item.get('color_balance', np.nan),
                'reasoning': item.get('reasoning', ''),
            }
            
            # Calculate average
            if all(k in item for k in ['impact', 'style', 'composition', 'lighting', 'color_balance']):
                row['average_score'] = (
                    item['impact'] + item['style'] + item['composition'] + 
                    item['lighting'] + item['color_balance']
                ) / 5
            else:
                row['average_score'] = np.nan
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(args.export_csv, index=False)
        print(f"CSV exported to: {args.export_csv}")


if __name__ == "__main__":
    main()

