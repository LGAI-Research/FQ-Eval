#!/usr/bin/env python3
"""
Scoring-Based Data Preparation Script

Converts user's enhanced CSV (base dataset + their model columns) into 
scoring-evaluation-ready JSON files. Creates separate files for each 
user model and each criteria combination.

Usage:
    python prepare_scoring_data.py --input enhanced_data.csv --user_models my_model1_fq,my_model2_fq

Output: For each user model and criteria combination, creates JSON files like:
    my_model1_contextual_relevance_scoring.json
    my_model1_exploratory_scope_scoring.json
    etc.
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from collections import defaultdict
import sys

class ScoringDataPrep:
    def __init__(self, input_file, user_model_columns, **kwargs):
        self.input_file = input_file
        self.user_model_columns = user_model_columns if isinstance(user_model_columns, list) else user_model_columns.split(',')
        
        # Required base columns
        self.required_base_cols = [
            'question_id', 'question', 'answer', 'criteria_name', 
            'criteria_desc', 'fq_id'
        ]
        
        # Optional columns
        self.optional_cols = ['set_id', 'complexity']
        
        # Options
        self.output_dir = Path(kwargs.get('output_dir', './scoring_ready_data'))
        
        # Data storage
        self.df = None
        self.skipped_records = []
        
    def load_and_validate_data(self):
        """Load CSV and validate structure"""
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
            return False
            
        # Check user model columns
        missing_models = [col for col in self.user_model_columns if col not in self.df.columns]
        if missing_models:
            print(f"❌ User model columns not found: {missing_models}")
            print(f"Available columns: {list(self.df.columns)}")
            return False
            
        print(f"✅ All required base columns found")
        print(f"✅ User model columns found: {self.user_model_columns}")
        
        return True
        
    def analyze_data_structure(self):
        """Analyze data structure and completeness"""
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
            
        # Check user model data completeness
        print(f"\n=== USER MODEL DATA COMPLETENESS ===")
        for model_col in self.user_model_columns:
            total_missing = self.df[model_col].isnull().sum()
            total_empty = (self.df[model_col].fillna('').str.strip() == '').sum()
            total_available = total_rows - total_missing - total_empty
            
            print(f"{model_col}:")
            print(f"  - Available: {total_available}/{total_rows} ({total_available/total_rows*100:.1f}%)")
            print(f"  - Missing/Empty: {total_missing + total_empty}/{total_rows}")
            
            if total_available < total_rows:
                print(f"  ⚠️ Some records will be skipped for this model")
                
    def create_scoring_files(self):
        """Create scoring-ready JSON files for each model and criteria combination"""
        print(f"\n=== CREATING SCORING-READY FILES ===")
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        created_files = {}
        total_files_created = 0
        
        # Process each user model column
        for model_col in self.user_model_columns:
            print(f"\n🔄 Processing model: {model_col}")
            
            model_files = {}
            model_skipped = []
            
            # Group by criteria for this model
            for criteria_name, group in self.df.groupby('criteria_name'):
                records = []
                
                for idx, row in group.iterrows():
                    # Check if this model has data for this row
                    model_response = row[model_col]
                    
                    if pd.isna(model_response) or str(model_response).strip() == '':
                        model_skipped.append({
                            'question_id': row['question_id'],
                            'criteria_name': criteria_name,
                            'model_column': model_col,
                            'reason': 'missing_model_response'
                        })
                        continue
                        
                    # Create scoring record
                    record = {
                        'question_id': row['question_id'],
                        'question': row['question'],
                        'answer': row['answer'],
                        'criteria_name': criteria_name,
                        'criteria_desc': row['criteria_desc'],
                        'fq_id': row['fq_id'],
                        'follow_up_question': str(model_response).strip()  # User's model response
                    }
                    
                    # Add optional fields if available
                    if 'complexity' in self.df.columns:
                        record['complexity'] = row.get('complexity', '')
                    if 'set_id' in self.df.columns:
                        record['set_id'] = row.get('set_id', '')
                        
                    records.append(record)
                    
                # Save this criteria file for this model
                if records:
                    safe_criteria_name = criteria_name.replace(' ', '_').lower()
                    safe_model_name = model_col.replace(' ', '_').replace('.', '_').lower()
                    
                    filename = f'{safe_model_name}_{safe_criteria_name}_scoring.json'
                    filepath = self.output_dir / filename
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(records, f, ensure_ascii=False, indent=2)
                        
                    model_files[criteria_name] = {
                        'filename': filename,
                        'path': str(filepath),
                        'record_count': len(records)
                    }
                    
                    print(f"  ✅ {criteria_name}: {len(records)} records → {filename}")
                    total_files_created += 1
                else:
                    print(f"  ⚠️ {criteria_name}: No valid records (all missing data)")
                    
            created_files[model_col] = {
                'files': model_files,
                'skipped_count': len(model_skipped)
            }
            
            # Store skipped records for this model
            if model_skipped:
                self.skipped_records.extend(model_skipped)
                
        print(f"\n📊 Summary: Created {total_files_created} scoring-ready JSON files")
        
        return created_files
        
    def save_processing_summary(self, created_files):
        """Save processing summary and skipped records"""
        print(f"\n=== SAVING SUMMARY INFORMATION ===")
        
        # Create processing summary
        summary = {
            'input_file': str(self.input_file),
            'user_model_columns': self.user_model_columns,
            'total_input_rows': len(self.df),
            'total_files_created': sum(len(info['files']) for info in created_files.values()),
            'files_by_model': {},
            'skipped_records_total': len(self.skipped_records),
            'processing_type': 'scoring_evaluation'
        }
        
        # Add file details for each model
        for model_col, info in created_files.items():
            summary['files_by_model'][model_col] = {
                'files_created': len(info['files']),
                'records_by_criteria': {criteria: file_info['record_count'] 
                                      for criteria, file_info in info['files'].items()},
                'skipped_records': info['skipped_count']
            }
            
        # Save summary
        summary_path = self.output_dir / 'scoring_preparation_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"📊 Processing summary saved to: scoring_preparation_summary.json")
        
        # Save skipped records if any
        if self.skipped_records:
            skipped_path = self.output_dir / 'skipped_records.json'
            with open(skipped_path, 'w', encoding='utf-8') as f:
                json.dump(self.skipped_records, f, ensure_ascii=False, indent=2)
            print(f"📋 Skipped records saved to: skipped_records.json")
            
        # Save a sample record for each model
        for model_col, info in created_files.items():
            if info['files']:
                first_criteria = list(info['files'].keys())[0]
                first_file = info['files'][first_criteria]
                
                # Load and save first record as sample
                with open(first_file['path'], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data:
                        safe_model_name = model_col.replace(' ', '_').replace('.', '_').lower()
                        sample_path = self.output_dir / f'sample_{safe_model_name}_record.json'
                        with open(sample_path, 'w', encoding='utf-8') as sf:
                            json.dump(data[0], sf, ensure_ascii=False, indent=2)
                        print(f"📝 Sample record for {model_col} saved to: sample_{safe_model_name}_record.json")
                        
    def generate_final_report(self, created_files):
        """Generate final preparation report"""
        print(f"\n" + "="*60)
        print(f"🎉 SCORING DATA PREPARATION COMPLETE!")
        print(f"="*60)
        
        total_files = sum(len(info['files']) for info in created_files.values())
        total_records = sum(
            sum(file_info['record_count'] for file_info in info['files'].values())
            for info in created_files.values()
        )
        
        print(f"\n📊 Preparation Summary:")
        print(f"  Input CSV: {self.input_file}")
        print(f"  User models: {len(self.user_model_columns)} ({', '.join(self.user_model_columns)})")
        print(f"  Total input rows: {len(self.df)}")
        print(f"  Total files created: {total_files}")
        print(f"  Total scoring records: {total_records}")
        print(f"  Skipped records: {len(self.skipped_records)}")
        
        print(f"\n📁 Files Created by Model:")
        for model_col, info in created_files.items():
            print(f"  {model_col}: {len(info['files'])} files")
            for criteria, file_info in info['files'].items():
                print(f"    - {file_info['filename']} ({file_info['record_count']} records)")
                
        print(f"\n🚀 Next Steps:")
        print(f"  1. Review the scoring_preparation_summary.json for details")
        print(f"  2. Run scoring evaluation for each file:")
        print(f"     python score_evaluation.py \\")
        print(f"       --data_fp {self.output_dir}/[model]_[criteria]_scoring.json \\")
        print(f"       --prompt_fp prompts/[criteria]_scoring_prompt.txt")
        
        print(f"\n💡 Tips:")
        print(f"  - Each file contains individual follow-up questions for scoring (1-5 scale)")
        print(f"  - Run evaluation separately for each model and criteria combination")
        print(f"  - Compare scores across models to see relative performance")
        
    def prepare(self):
        """Main preparation process"""
        if not self.load_and_validate_data():
            return False
            
        self.analyze_data_structure()
        created_files = self.create_scoring_files()
        
        if not any(info['files'] for info in created_files.values()):
            print(f"❌ No valid files created. Check your data and model columns.")
            return False
            
        self.save_processing_summary(created_files)
        self.generate_final_report(created_files)
        
        return True

def main():
    parser = argparse.ArgumentParser(
        description="Prepare CSV data for scoring-based evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Single model
  python prepare_scoring_data.py --input enhanced_data.csv --user_models my_model_fq
  
  # Multiple models
  python prepare_scoring_data.py --input enhanced_data.csv --user_models my_model1_fq,my_model2_fq,my_model3_fq
  
  # Custom output directory
  python prepare_scoring_data.py --input data.csv --user_models my_gpt,my_claude --output_dir ./my_scoring_data

The input CSV should be the provided base dataset with your model columns added:
  Base columns: set_id, question_id, question, complexity, answer, criteria_name, criteria_desc, fq_id
  Your addition: [your_model_columns]
        """
    )
    
    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Path to enhanced CSV file (base dataset + your model columns)')
    parser.add_argument('--user_models', type=str, required=True,
                       help='Comma-separated list of your model column names')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./scoring_ready_data',
                       help='Output directory for scoring files (default: ./scoring_ready_data)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"❌ Error: Input file '{args.input}' not found")
        return 1
        
    # Create preparer and run
    preparer = ScoringDataPrep(
        input_file=args.input,
        user_model_columns=args.user_models,
        output_dir=args.output_dir
    )
    
    success = preparer.prepare()
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())