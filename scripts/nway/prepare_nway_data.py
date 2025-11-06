#!/usr/bin/env python3
"""
Simple Data Conversion Script for Multi-Candidate Follow-up Question Evaluation

This script takes the provided base CSV dataset (with all required columns already filled)
and converts it to evaluation-ready format. Users can choose which columns to use as candidates.

Usage:
    python convert_for_evaluation.py --input enhanced_data.csv --candidates final_fq,my_model1,my_model2
    
The base dataset already contains:
    set_id, question_id, question, complexity, answer, criteria_name, criteria_desc, fq_id, final_fq
    
Users add their columns and specify which ones to use as candidates for comparison.
"""

import pandas as pd
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
import sys

class SimpleDataConverter:
    def __init__(self, input_file, candidate_columns, **kwargs):
        self.input_file = input_file
        self.candidate_columns = candidate_columns if isinstance(candidate_columns, list) else candidate_columns.split(',')
        
        # Validate minimum candidates at initialization
        if len(self.candidate_columns) < 2:
            raise ValueError(f"Need at least 2 candidate columns for comparison, got {len(self.candidate_columns)}: {self.candidate_columns}")
        
        # Required base columns (provided in the base dataset)
        self.required_base_cols = [
            'question_id', 'question', 'answer', 'criteria_name', 
            'criteria_desc', 'fq_id'
        ]
        
        # Optional columns
        self.optional_cols = ['set_id', 'complexity', 'final_fq']
        
        # Options
        self.output_dir = Path(kwargs.get('output_dir', './prepped_data/nway'))
        self.seed = kwargs.get('seed', 42)
        self.fq_id_col = kwargs.get('fq_id_col', 'fq_id')
        
        # Data storage
        self.df = None
        self.records = []
        self.skipped_records = []
        
    def load_and_validate_data(self):
        """Load CSV and validate it has base structure + candidate columns"""
        print(f"=== LOADING DATA ===")
        print(f"Loading data from: {self.input_file}")
    
        try:
            self.df = pd.read_csv(self.input_file, encoding='utf-8')
            print(f"✅ Loaded data shape: {self.df.shape}")
        except Exception as e:
            print(f"❌ Failed to load CSV: {e}")
            return False
        
        print(f"Available columns: {list(self.df.columns)}")
    
        # Check required base columns
        missing_base_cols = [col for col in self.required_base_cols if col not in self.df.columns]
        if missing_base_cols:
            print(f"❌ Missing required base columns: {missing_base_cols}")
            print(f"Expected base columns: {self.required_base_cols}")
            print(f"Make sure you're using the provided base dataset!")
            return False
        
        # Check candidate columns
        missing_candidates = [col for col in self.candidate_columns if col not in self.df.columns]
        if missing_candidates:
            print(f"❌ Candidate columns not found: {missing_candidates}")
            print(f"Available columns: {list(self.df.columns)}")
            print(f"Did you specify the correct column names?")
            return False
        
        print(f"✅ All required base columns found")
        print(f"✅ Candidate columns found: {self.candidate_columns}")
    
        return True
        
    def analyze_data_completeness(self):
        """Analyze data completeness and show statistics"""
        print(f"\n=== DATA ANALYSIS ===")
        
        # Basic statistics
        total_rows = len(self.df)
        unique_questions = self.df['question_id'].nunique()
        unique_criteria = self.df['criteria_name'].nunique()
        
        print(f"Dataset statistics:")
        print(f"  - Total rows: {total_rows}")
        print(f"  - Unique questions: {unique_questions}")
        print(f"  - Unique criteria: {unique_criteria}")
        print(f"  - Expected rows: {unique_questions} × {unique_criteria} = {unique_questions * unique_criteria}")
        
        # Check criteria distribution
        criteria_counts = self.df['criteria_name'].value_counts()
        print(f"\nCriteria distribution:")
        for criteria, count in criteria_counts.items():
            print(f"  - {criteria}: {count} questions")
            
        # Check data completeness
        print(f"\n=== DATA COMPLETENESS CHECK ===")
        
        # Check base data (should be 100% complete)
        essential_cols = ['question_id', 'question', 'answer']
        base_missing = self.df[essential_cols].isnull().sum()
        
        if base_missing.sum() > 0:
            print(f"⚠️ Missing base data (should be complete in provided dataset):")
            for col, count in base_missing[base_missing > 0].items():
                print(f"  - {col}: {count} missing values")
        else:
            print(f"✅ Base dataset is complete (no missing essential data)")
            
        # Check candidate data completeness
        print(f"\nCandidate data completeness:")
        for candidate_col in self.candidate_columns:
            candidate_missing = self.df[candidate_col].isnull().sum()
            candidate_empty = (self.df[candidate_col].fillna('').str.strip() == '').sum()
            candidate_available = total_rows - candidate_missing - candidate_empty
            print(f"  - {candidate_col}: {candidate_available}/{total_rows} ({candidate_available/total_rows*100:.1f}%)")
        
        min_available = min([
            total_rows - self.df[col].isnull().sum() - (self.df[col].fillna('').str.strip() == '').sum()
            for col in self.candidate_columns
        ])
        
        if min_available < total_rows:
            print(f"⚠️ Some evaluations will be skipped due to missing candidate data")
            
    def create_evaluation_records(self):
        """Create multi-candidate evaluation records"""
        print(f"\n=== CREATING EVALUATION RECORDS ===")
        
        self.records = []
        self.skipped_records = []
        
        # Process each row (each row = one question + one criteria combination)
        for idx, row in self.df.iterrows():
            question_id = row['question_id']
            criteria_name = row['criteria_name']
            
            # Check if we have at least minimum required candidates
            available_candidates = {}
            missing_candidates = []
            
            # Check all specified candidate columns
            for candidate_col in self.candidate_columns:
                candidate_response = row[candidate_col]
                if not (pd.isna(candidate_response) or str(candidate_response).strip() == ''):
                    available_candidates[candidate_col] = str(candidate_response).strip()
                else:
                    missing_candidates.append(candidate_col)
                    
            # Skip if insufficient candidates available (need at least 2)
            total_candidates = len(available_candidates)
            if total_candidates < 2:
                self.skipped_records.append({
                    'question_id': question_id,
                    'criteria_name': criteria_name,
                    'reason': 'insufficient_candidates',
                    'available_count': total_candidates,
                    'minimum_required': 2,
                    'missing_candidates': missing_candidates
                })
                continue
                
            # Create candidates list from available columns
            candidates = []
            
            for candidate_col in self.candidate_columns:
                if candidate_col in available_candidates:
                    # Generate candidate ID
                    candidate_id = f"{row.get(self.fq_id_col, f'fq_{question_id}')}_{candidate_col.replace('.', '_').replace('-', '_')}"
                    
                    # Determine candidate type (try to infer from column name)
                    if candidate_col == 'final_fq':
                        candidate_type = 'bench'
                    else:
                        candidate_type = candidate_col
                    
                    candidates.append({
                        'id': candidate_id,
                        'text': available_candidates[candidate_col],
                        'type': candidate_type
                    })
            
            # Create evaluation record
            record = {
                'question_id': question_id,
                'user_question': row['question'],
                'answer': row['answer'],
                'criteria_name': criteria_name,
                'criteria_desc': row['criteria_desc'],
                'complexity': row.get('complexity', ''),
                'original_fq_id': row[self.fq_id_col],
                'candidates': candidates
            }
            
            self.records.append(record)
            
        print(f"✅ Created {len(self.records)} evaluation records")
        print(f"⚠️ Skipped {len(self.skipped_records)} records")
        
        # Show skip reasons
        if self.skipped_records:
            skip_reasons = defaultdict(int)
            for skip in self.skipped_records:
                skip_reasons[skip['reason']] += 1
            print(f"Skip reasons:")
            for reason, count in skip_reasons.items():
                reason_desc = {
                    'insufficient_candidates': f'Insufficient candidates available (need at least 2)'
                }
                print(f"  - {reason_desc.get(reason, reason)}: {count}")
                
    def anonymize_and_randomize(self, record):
        """Anonymize IDs and randomize candidate order"""
        # Generate candidate labels (A, B, C, etc. for multiple candidates)
        num_candidates = len(record['candidates'])
        candidate_labels = [f"candidate_{chr(65 + i)}" for i in range(num_candidates)]
        
        # Create pairs and shuffle
        candidate_pairs = list(zip(record['candidates'], candidate_labels))
        random.shuffle(candidate_pairs)
        
        # Build anonymized data
        anonymized_candidates = []
        mapping = {}
        
        for original_candidate, anonymous_label in candidate_pairs:
            anonymized_candidates.append({
                "id": anonymous_label,
                "text": original_candidate["text"]
            })
            
            mapping[anonymous_label] = {
                "original_id": original_candidate["id"],
                "original_type": original_candidate["type"]
            }
            
        # Create anonymized record
        anonymized_record = {
            "question_id": record["question_id"],
            "user_question": record["user_question"],
            "answer": record["answer"],
            "criteria_name": record["criteria_name"],
            "complexity": record.get("complexity", ""),
            "original_fq_id": record.get("original_fq_id", ""),
            "candidates": anonymized_candidates,
            "mapping": mapping
        }
        
        return anonymized_record
        
    def save_evaluation_files(self):
        """Save anonymized data split by criteria"""
        print(f"\n=== SAVING EVALUATION-READY FILES ===")
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set random seed for reproducible anonymization
        random.seed(self.seed)
        
        # Group records by criteria
        records_by_criteria = defaultdict(list)
        for record in self.records:
            criteria_name = record['criteria_name']
            
            # Anonymize and randomize
            anonymized_record = self.anonymize_and_randomize(record)
            records_by_criteria[criteria_name].append(anonymized_record)
            
        # Save each criteria separately
        saved_files = {}
        for criteria_name, records in records_by_criteria.items():
            # Create safe filename
            safe_name = criteria_name.replace(' ', '_').replace('-', '_').lower()
            
            # Save as JSON
            candidate_suffix = "_".join(sorted(self.candidate_columns)).replace(' ', '_').replace('.', '_').lower()
            json_filename = f'{safe_name}_{candidate_suffix}_evaluation_ready.json'
            json_path = self.output_dir / json_filename
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
                
            saved_files[criteria_name] = {
                'filename': json_filename,
                'path': str(json_path),
                'record_count': len(records)
            }
            
            print(f"✅ Saved {len(records)} records for '{criteria_name}' to {json_filename}")
            
        # Save conversion summary
        summary = {
            'input_file': str(self.input_file),
            'candidate_columns': self.candidate_columns,
            'total_input_rows': len(self.df),
            'total_output_records': len(self.records),
            'records_by_criteria': {name: info['record_count'] for name, info in saved_files.items()},
            'skipped_records': len(self.skipped_records),
            'output_files': saved_files,
            'conversion_settings': {
                'seed': self.seed,
                'candidates_per_evaluation': f"{len(self.candidate_columns)} specified candidates (variable based on data availability)"
            }
        }
        
        summary_path = self.output_dir / 'conversion_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            
        print(f"📊 Conversion summary saved to: conversion_summary.json")
        
        # Save skipped records if any
        if self.skipped_records:
            skipped_path = self.output_dir / 'skipped_records.json'
            with open(skipped_path, 'w', encoding='utf-8') as f:
                json.dump(self.skipped_records, f, ensure_ascii=False, indent=2)
            print(f"📋 Skipped records saved to: skipped_records.json")
            
        # Save a sample record for inspection
        if records_by_criteria:
            first_criteria = list(records_by_criteria.keys())[0]
            sample_record = records_by_criteria[first_criteria][0]
            
            sample_path = self.output_dir / 'sample_evaluation_record.json'
            with open(sample_path, 'w', encoding='utf-8') as f:
                json.dump(sample_record, f, ensure_ascii=False, indent=2)
            print(f"📝 Sample record saved to: sample_evaluation_record.json")
            
        return saved_files
        
    def generate_final_report(self, saved_files):
        """Generate final conversion report"""
        print(f"\n" + "="*60)
        print(f"🎉 CONVERSION COMPLETED SUCCESSFULLY!")
        print(f"="*60)
        
        total_evaluations = sum(info['record_count'] for info in saved_files.values())
        
        print(f"\n📊 Conversion Summary:")
        print(f"  Input CSV: {self.input_file}")
        print(f"  Candidate columns: {', '.join(self.candidate_columns)}")
        print(f"  Total input rows: {len(self.df)}")
        print(f"  Questions processed: {self.df['question_id'].nunique()}")
        print(f"  Criteria processed: {len(saved_files)}")
        print(f"  Total evaluations created: {total_evaluations}")
        print(f"  Evaluations per criteria:")
        for criteria_name, info in saved_files.items():
            print(f"    - {criteria_name}: {info['record_count']}")
        print(f"  Skipped records: {len(self.skipped_records)}")
        
        print(f"\n📁 Output Files (in {self.output_dir}):")
        for criteria_name, info in saved_files.items():
            print(f"  ✅ {info['filename']}")
        print(f"  📊 conversion_summary.json")
        print(f"  📝 sample_evaluation_record.json")
        if self.skipped_records:
            print(f"  📋 skipped_records.json")
            
        print(f"\n🚀 Next Steps:")
        print(f"  1. Review the conversion summary and sample record")
        print(f"  2. Run evaluation using your framework")
            
        print(f"\n💡 Tips:")
        print(f"  - Each evaluation compares your selected candidates ({len(self.candidate_columns)} columns)")
        print(f"  - Candidates are anonymized and randomized for fair evaluation")
        print(f"  - Use the mapping in each record to trace results back to models")
        
    def convert(self):
        """Main conversion process"""
        if not self.load_and_validate_data():
            return False
            
        self.analyze_data_completeness()
        self.create_evaluation_records()
        
        if not self.records:
            print(f"❌ No valid evaluation records created.")
            print(f"Check that your candidate columns have sufficient data.")
            return False
            
        saved_files = self.save_evaluation_files()
        self.generate_final_report(saved_files)
        
        return True

def main():
    parser = argparse.ArgumentParser(
        description="Convert enhanced base dataset to evaluation-ready format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Include human benchmark + 2 user models
  python convert_for_evaluation.py --input data.csv --candidates final_fq,my_model1,my_model2
  
  # Only user models (no human benchmark)
  python convert_for_evaluation.py --input data.csv --candidates my_model1,my_model2,my_model3
  
  # 2-way comparison (human vs single model)
  python convert_for_evaluation.py --input data.csv --candidates final_fq,my_model
  
  # Custom output directory
  python convert_for_evaluation.py --input data.csv --candidates final_fq,my_model --output_dir ./my_evaluation_data
  
The input CSV should be the provided base dataset with your model's column added:
  Base columns: set_id, question_id, question, complexity, answer, criteria_name, criteria_desc, fq_id, final_fq
  Your addition: [your_model_column_name]
        """
    )
    
    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Path to enhanced CSV file (base dataset + your model column)')
    parser.add_argument('--candidates', type=str, required=True,
                       help='Comma-separated list of candidate columns (e.g., final_fq,my_model1,my_model2)')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./evaluation_ready_data',
                       help='Output directory for evaluation files (default: ./evaluation_ready_data)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible anonymization (default: 42)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"❌ Error: Input file '{args.input}' not found")
        return 1
    
    # Validate candidate columns
    candidate_list = args.candidates.split(',')
    if len(candidate_list) < 2:
        print(f"❌ Error: Need at least 2 candidate columns for comparison, got {len(candidate_list)}: {candidate_list}")
        return 1
        
    # Create converter and run
    try:
        converter = SimpleDataConverter(
            input_file=args.input,
            candidate_columns=args.candidates,
            output_dir=args.output_dir,
            seed=args.seed
        )
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    
    success = converter.convert()
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())