#!/bin/bash

# Scoring Auto Evaluation Script
# Automatically prepares data and runs scoring evaluations for all user models and criteria

# Note: We don't use 'set -e' here because it can interfere with the evaluation loop

# Default values
DEFAULT_MODEL="gpt-4.1-2025-04-14"
DEFAULT_PARALLEL=5
DEFAULT_TEMPERATURE=0.0
DEFAULT_N=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Scoring Auto Evaluation Script

USAGE:
    $0 --input <csv_file> --user_models <columns> --key <api_key> [OPTIONS]

REQUIRED ARGUMENTS:
    --input         Path to input CSV file (in raw_data/ directory)
    --user_models   Comma-separated list of user model columns (e.g., my_fq,my_fq1,my_fq2)
    --key           OpenAI API key

OPTIONAL ARGUMENTS:
    --model         OpenAI model to use (default: $DEFAULT_MODEL)
    --parallel      Number of parallel API calls (default: $DEFAULT_PARALLEL)
    --sample_size   Process only first N items for testing (default: all)
    --help          Show this help message

EXAMPLES:
    # Basic usage
    $0 --input raw_data/my_data.csv --user_models my_fq,my_fq1,my_fq2 --key sk-your-key

    # With custom model and parallel processing
    $0 --input raw_data/my_data.csv --user_models my_model1,my_model2 --key sk-your-key --model gpt-4 --parallel 3

    # Testing with limited data
    $0 --input raw_data/my_data.csv --user_models my_fq --key sk-your-key --sample_size 20

WORKFLOW:
    For M user models and 5 criteria, this creates M×5 files and runs M×5 evaluations.
    Each evaluation uses the criteria-specific prompt for that data file.

OUTPUT STRUCTURE:
    prepped_data/scoring/             - Prepared evaluation data (M×5 files)
    results/scoring/individual/       - Individual evaluation results (M×5 files)
    results/scoring/summary/          - Comprehensive per-model summaries (M files)

EOF
}

# Parse arguments
INPUT_FILE=""
USER_MODELS=""
API_KEY=""
MODEL="$DEFAULT_MODEL"
PARALLEL="$DEFAULT_PARALLEL"
SAMPLE_SIZE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            if [[ -z "$2" || "$2" == --* ]]; then
                print_error "--input requires a value"
                exit 1
            fi
            INPUT_FILE="$2"
            shift 2
            ;;
        --user_models)
            if [[ -z "$2" || "$2" == --* ]]; then
                print_error "--user_models requires a value"
                exit 1
            fi
            USER_MODELS="$2"
            shift 2
            ;;
        --key)
            if [[ -z "$2" || "$2" == --* ]]; then
                print_error "--key requires a value"
                exit 1
            fi
            API_KEY="$2"
            shift 2
            ;;
        --model)
            if [[ -z "$2" || "$2" == --* ]]; then
                print_error "--model requires a value"
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        --parallel)
            if [[ -z "$2" || "$2" == --* ]]; then
                print_error "--parallel requires a value"
                exit 1
            fi
            PARALLEL="$2"
            shift 2
            ;;
        --sample_size)
            if [[ -z "$2" || "$2" == --* ]]; then
                print_error "--sample_size requires a value"
                exit 1
            fi
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        -*)
            print_error "Unknown argument: $1"
            show_usage
            exit 1
            ;;
        *)
            print_error "Unexpected argument: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_FILE" ]]; then
    print_error "Missing required argument: --input"
    show_usage
    exit 1
fi

if [[ -z "$USER_MODELS" ]]; then
    print_error "Missing required argument: --user_models"
    show_usage
    exit 1
fi

if [[ -z "$API_KEY" ]]; then
    print_error "Missing required argument: --key"
    show_usage
    exit 1
fi

# Validate input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    print_error "Input file not found: $INPUT_FILE"
    exit 1
fi

# Validate user models format (at least 1 model)
IFS=',' read -ra USER_MODEL_ARRAY <<< "$USER_MODELS"
if [[ ${#USER_MODEL_ARRAY[@]} -lt 1 ]]; then
    print_error "Need at least 1 user model, got ${#USER_MODEL_ARRAY[@]}: $USER_MODELS"
    exit 1
fi

# Define criteria list (should match the 5 criteria in your CSV)
CRITERIA_LIST=("contextual_relevance" "creative_leap" "exploratory_scope" "guided_onboarding" "llm_enablement")

print_status "Starting Scoring Auto Evaluation"
print_status "================================="
print_status "Input file: $INPUT_FILE"
print_status "User models: $USER_MODELS (${#USER_MODEL_ARRAY[@]} total)"
print_status "Expected evaluations: ${#USER_MODEL_ARRAY[@]} models × ${#CRITERIA_LIST[@]} criteria = $((${#USER_MODEL_ARRAY[@]} * ${#CRITERIA_LIST[@]}))"
print_status "Model: $MODEL"
print_status "Parallel workers: $PARALLEL"
if [[ -n "$SAMPLE_SIZE" ]]; then
    print_status "Sample size: $SAMPLE_SIZE (testing mode)"
fi

# Step 1: Data Preparation
print_status ""
print_status "Step 1: Preparing scoring data..."
print_status "=================================="

PREP_OUTPUT_DIR="prepped_data/scoring"
mkdir -p "$PREP_OUTPUT_DIR"

PREP_CMD="python scripts/scoring/prepare_scoring_data.py \
    --input \"$INPUT_FILE\" \
    --user_models \"$USER_MODELS\" \
    --output_dir \"$PREP_OUTPUT_DIR\""

print_status "Running: $PREP_CMD"

if ! eval $PREP_CMD; then
    print_error "Data preparation failed!"
    exit 1
fi

print_success "Data preparation completed!"

# Build list of files that SHOULD exist based on current user models and criteria
SCORING_FILES=()
for model in "${USER_MODEL_ARRAY[@]}"; do
    # Convert model name to safe format (same as data prep script)
    safe_model_name=$(echo "$model" | sed 's/\./_/g' | tr '[:upper:]' '[:lower:]')
    
    for criteria in "${CRITERIA_LIST[@]}"; do
        expected_file="$PREP_OUTPUT_DIR/${safe_model_name}_${criteria}_scoring.json"
        if [[ -f "$expected_file" ]]; then
            SCORING_FILES+=("$expected_file")
        else
            print_warning "  - Missing: $(basename "$expected_file") (model may have no data for this criteria)"
        fi
    done
done

# Check if we found any files
if [[ ${#SCORING_FILES[@]} -eq 0 ]]; then
    print_error "No scoring files were generated. Check your input data and user model columns."
    exit 1
fi

print_status "Found ${#SCORING_FILES[@]} scoring files for specified models:"
for file in "${SCORING_FILES[@]}"; do
    filename=$(basename "$file")
    print_status "  - $filename"
done

# Step 2: Run Scoring Evaluations
print_status ""
print_status "Step 2: Running scoring evaluations..."
print_status "====================================="

# Create results directories
RESULTS_INDIVIDUAL_DIR="results/scoring/individual"
RESULTS_COMPREHENSIVE_DIR="results/scoring/summary"
mkdir -p "$RESULTS_INDIVIDUAL_DIR"
mkdir -p "$RESULTS_COMPREHENSIVE_DIR"

# Define criteria-to-prompt mapping (lowercase underscore format)
declare -A CRITERIA_PROMPTS=(
    ["contextual_relevance"]="prompts/scoring/scoringprompt_contextual_relevance.txt"
    ["creative_leap"]="prompts/scoring/scoringprompt_creative_leap.txt"
    ["exploratory_scope"]="prompts/scoring/scoringprompt_exploratory_scope.txt"
    ["guided_onboarding"]="prompts/scoring/scoringprompt_guided_onboarding.txt"
    ["llm_enablement"]="prompts/scoring/scoringprompt_llm_enablement.txt"
)

SUCCESSFUL_EVALUATIONS=0
FAILED_EVALUATIONS=0

# Track results by model for comprehensive summaries
declare -A MODEL_RESULTS

# Process each scoring file (each represents one model+criteria combination)
for scoring_file in "${SCORING_FILES[@]}"; do
    # Extract model and criteria from filename
    # Expected format from data prep: {safe_model_name}_{safe_criteria_name}_scoring.json
    filename=$(basename "$scoring_file")
    filename_base=$(echo "$filename" | sed 's/_scoring\.json$//')
    
    # Extract criteria by checking against known criteria list
    criteria=""
    model_name=""
    
    for known_criteria in "${CRITERIA_LIST[@]}"; do
        if [[ "$filename_base" == *"_${known_criteria}" ]]; then
            criteria="$known_criteria"
            model_name=$(echo "$filename_base" | sed "s/_${known_criteria}$//")
            break
        fi
    done
    
    # Fallback if no match found
    if [[ -z "$criteria" ]]; then
        print_error "Could not extract criteria from filename: $filename"
        print_error "Expected one of: ${CRITERIA_LIST[*]}"
        exit 1
    fi    

    print_status ""
    print_status "Processing: $model_name → $criteria"
    print_status "File: $(basename "$scoring_file")"
    print_status "--------------------------------"
    
    if [[ ! -f "$scoring_file" ]]; then
        print_error "Scoring file not found: $scoring_file"
        print_error "Critical error in file processing"
        exit 1
    fi
    
    # Validate criteria and get corresponding prompt
    prompt_file="${CRITERIA_PROMPTS[$criteria]}"
    if [[ -z "$prompt_file" ]]; then
        print_error "No prompt mapping found for criteria: $criteria"
        print_error "Available criteria: ${!CRITERIA_PROMPTS[@]}"
        print_error "Check if criteria extraction from filename is correct"
        exit 1
    fi
    
    if [[ ! -f "$prompt_file" ]]; then
        print_error "Prompt file not found: $prompt_file"
        print_error "Critical error: Cannot proceed without criteria-specific prompt"
        exit 1
    fi
    
    print_status "Using prompt: $prompt_file"
    
    # Setup output paths
    result_file="$RESULTS_INDIVIDUAL_DIR/${model_name}_${criteria}_scores.json"
    
    # Build evaluation command
    EVAL_CMD="python scripts/scoring/score_evaluation.py \
        --prompt_fp \"$prompt_file\" \
        --data_fp \"$scoring_file\" \
        --save_fp \"$result_file\" \
        --key \"$API_KEY\" \
        --model \"$MODEL\" \
        --temperature $DEFAULT_TEMPERATURE \
        --n $DEFAULT_N \
        --parallel $PARALLEL"
    
    # Add sample_size if specified
    if [[ -n "$SAMPLE_SIZE" ]]; then
        EVAL_CMD="$EVAL_CMD --sample_size $SAMPLE_SIZE"
    fi
    
    print_status "Running scoring evaluation..."
    
    # Run evaluation with error handling
    if eval $EVAL_CMD 2>&1; then
        print_success "✅ $model_name - $criteria completed"
        ((SUCCESSFUL_EVALUATIONS++))
        
        # Track results for comprehensive summary
        if [[ -z "${MODEL_RESULTS[$model_name]}" ]]; then
            MODEL_RESULTS[$model_name]=""
        fi
        MODEL_RESULTS[$model_name]="${MODEL_RESULTS[$model_name]} $result_file"
        
    else
        print_error "❌ $model_name - $criteria evaluation failed"
        print_error "Command that failed: $EVAL_CMD"
        print_error "Stopping evaluation due to failure"
        ((FAILED_EVALUATIONS++))
        exit 1
    fi
    
    # Progress indicator
    print_status "Progress: $SUCCESSFUL_EVALUATIONS/${#SCORING_FILES[@]} evaluations completed"
done

# Step 3: Generate Comprehensive Per-Model Summaries
print_status ""
print_status "Step 3: Creating comprehensive per-model summaries..."
print_status "===================================================="

for model_name in "${!MODEL_RESULTS[@]}"; do
    print_status "Generating comprehensive summary for: $model_name"
    
    result_files=(${MODEL_RESULTS[$model_name]})
    comprehensive_summary_file="$RESULTS_COMPREHENSIVE_DIR/${model_name}_comprehensive_summary_scoring.json"
    
    print_status "Processing ${#result_files[@]} result files for $model_name"
    
    # Create comprehensive summary using Python
    python3 << EOF
import json
import sys
from collections import defaultdict
from pathlib import Path
import datetime

def create_comprehensive_summary(result_files, output_file, model_name):
    """Create comprehensive summary across all criteria for a model"""
    
    all_scores = []
    criteria_scores = defaultdict(list)
    criteria_stats = {}
    total_evaluations = 0
    total_valid_scores = 0
    parse_errors = 0
    api_errors = 0
    
    for result_file in result_files:
        result_file = result_file.strip()
        if not result_file:
            continue
            
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract criteria name from filename
            # Format: {model}_{criteria}_scores.json
            filename = Path(result_file).stem
            # Remove model name prefix and "_scores" suffix to get criteria
            criteria = filename.replace(model_name + "_", "").replace("_scores", "")
            
            file_scores = []
            valid_scores_count = 0
            file_parse_errors = 0
            file_api_errors = 0
            file_total_evaluations = len(data)
            
            for record in data:
                total_evaluations += 1
                if 'evaluation' in record:
                    eval_data = record['evaluation']
                    
                    # Check for errors
                    if eval_data.get('parse_error'):
                        file_parse_errors += 1
                        parse_errors += 1
                    elif eval_data.get('error'):
                        file_api_errors += 1
                        api_errors += 1
                    elif eval_data.get('score') is not None:
                        score = eval_data['score']
                        if isinstance(score, (int, float)) and 1 <= score <= 5:
                            all_scores.append(score)
                            criteria_scores[criteria].append(score)
                            file_scores.append(score)
                            valid_scores_count += 1
                            total_valid_scores += 1
            
            # Calculate criteria-specific stats
            if file_scores:
                criteria_stats[criteria] = {
                    'count': len(file_scores),
                    'average': sum(file_scores) / len(file_scores),
                    'min': min(file_scores),
                    'max': max(file_scores),
                    'distribution': {str(i): file_scores.count(i) for i in range(1, 6)},
                    'quality_breakdown': {
                        'high_quality_4_5': len([s for s in file_scores if s >= 4]),
                        'average_quality_3': len([s for s in file_scores if s == 3]),
                        'low_quality_1_2': len([s for s in file_scores if s <= 2])
                    },
                    'parse_errors': file_parse_errors,
                    'api_errors': file_api_errors,
                    # FIXED: Correct success rate calculation
                    'success_rate': (valid_scores_count / file_total_evaluations * 100) if file_total_evaluations > 0 else 0
                }
            else:
                criteria_stats[criteria] = {
                    'count': 0,
                    'average': 0,
                    'min': 0,
                    'max': 0,
                    'distribution': {str(i): 0 for i in range(1, 6)},
                    'quality_breakdown': {
                        'high_quality_4_5': 0,
                        'average_quality_3': 0,
                        'low_quality_1_2': 0
                    },
                    'parse_errors': file_parse_errors,
                    'api_errors': file_api_errors,
                    'success_rate': 0
                }
                
        except Exception as e:
            print(f"Error processing {result_file}: {e}", file=sys.stderr)
            continue
    
    # Overall statistics
    overall_stats = {}
    if all_scores:
        overall_stats = {
            'total_evaluations': total_evaluations,
            'valid_scores': total_valid_scores,
            # FIXED: Correct success rate calculation
            'success_rate': (total_valid_scores / total_evaluations * 100) if total_evaluations > 0 else 0,
            'average_score': sum(all_scores) / len(all_scores),
            'min_score': min(all_scores),
            'max_score': max(all_scores),
            'score_distribution': {str(i): all_scores.count(i) for i in range(1, 6)},
            'quality_breakdown': {
                'high_quality_4_5': len([s for s in all_scores if s >= 4]),
                'average_quality_3': len([s for s in all_scores if s == 3]),
                'low_quality_1_2': len([s for s in all_scores if s <= 2])
            },
            'quality_percentages': {
                'high_quality_4_5_pct': len([s for s in all_scores if s >= 4]) / len(all_scores) * 100,
                'average_quality_3_pct': all_scores.count(3) / len(all_scores) * 100,
                'low_quality_1_2_pct': len([s for s in all_scores if s <= 2]) / len(all_scores) * 100
            },
            'error_summary': {
                'parse_errors': parse_errors,
                'api_errors': api_errors,
                'total_errors': parse_errors + api_errors
            }
        }
    else:
        overall_stats = {
            'total_evaluations': total_evaluations,
            'valid_scores': 0,
            'success_rate': 0,
            'average_score': 0,
            'min_score': 0,
            'max_score': 0,
            'score_distribution': {str(i): 0 for i in range(1, 6)},
            'quality_breakdown': {
                'high_quality_4_5': 0,
                'average_quality_3': 0,
                'low_quality_1_2': 0
            },
            'quality_percentages': {
                'high_quality_4_5_pct': 0,
                'average_quality_3_pct': 0,
                'low_quality_1_2_pct': 0
            },
            'error_summary': {
                'parse_errors': parse_errors,
                'api_errors': api_errors,
                'total_errors': parse_errors + api_errors
            }
        }
    
    # Performance insights
    performance_insights = {}
    if criteria_stats:
        # Best and worst performing criteria
        criteria_with_scores = {k: v for k, v in criteria_stats.items() if v['count'] > 0}
        
        if criteria_with_scores:
            best_criteria = max(criteria_with_scores.keys(), key=lambda c: criteria_with_scores[c]['average'])
            worst_criteria = min(criteria_with_scores.keys(), key=lambda c: criteria_with_scores[c]['average'])
            
            # Calculate consistency (how close are criteria averages to overall average)
            criteria_averages = [v['average'] for v in criteria_with_scores.values()]
            if len(criteria_averages) > 1 and overall_stats['average_score'] > 0:
                avg_variance = sum((avg - overall_stats['average_score'])**2 for avg in criteria_averages) / len(criteria_averages)
                # FIXED: Better consistency calculation to avoid division by zero
                consistency_score = max(0, 1 - (avg_variance / (overall_stats['average_score']**2 + 0.001)))
            else:
                consistency_score = 1.0 if len(criteria_averages) == 1 else 0
        else:
            best_criteria = None
            worst_criteria = None
            consistency_score = 0
            
        performance_insights = {
            'best_criteria': best_criteria,
            'worst_criteria': worst_criteria,
            'criteria_evaluated': len(criteria_stats),
            'criteria_with_valid_scores': len(criteria_with_scores),
            'consistency_score': consistency_score,
            'performance_range': {
                'highest_avg': max(criteria_with_scores.values(), key=lambda x: x['average'])['average'] if criteria_with_scores else 0,
                'lowest_avg': min(criteria_with_scores.values(), key=lambda x: x['average'])['average'] if criteria_with_scores else 0,
                'range_spread': (max(criteria_with_scores.values(), key=lambda x: x['average'])['average'] - 
                               min(criteria_with_scores.values(), key=lambda x: x['average'])['average']) if len(criteria_with_scores) > 1 else 0
            }
        }
    else:
        performance_insights = {
            'best_criteria': None,
            'worst_criteria': None,
            'criteria_evaluated': 0,
            'criteria_with_valid_scores': 0,
            'consistency_score': 0,
            'performance_range': {
                'highest_avg': 0,
                'lowest_avg': 0,
                'range_spread': 0
            }
        }
    
    # Create comprehensive summary
    summary = {
        'model_name': model_name,
        'evaluation_metadata': {
            'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'total_files_processed': len([f for f in result_files if f.strip()]),
            'criteria_list': sorted(list(criteria_stats.keys())),
            'evaluation_type': 'scoring',
            'score_scale': '1-5'
        },
        'overall_performance': overall_stats,
        'criteria_breakdown': criteria_stats,
        'performance_insights': performance_insights
    }
    
    # Save comprehensive summary
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return summary

# Create the comprehensive summary
result_files = """${MODEL_RESULTS[$model_name]}""".split()
summary = create_comprehensive_summary(result_files, "$comprehensive_summary_file", "$model_name")

print(f"✅ Created comprehensive summary for $model_name")
print(f"   - Total evaluations: {summary['overall_performance']['total_evaluations']}")
print(f"   - Valid scores: {summary['overall_performance']['valid_scores']}")
print(f"   - Success rate: {summary['overall_performance']['success_rate']:.1f}%")
if summary['overall_performance']['valid_scores'] > 0:
    print(f"   - Average score: {summary['overall_performance']['average_score']:.2f}")
    print(f"   - Best criteria: {summary['performance_insights']['best_criteria']}")
    print(f"   - Worst criteria: {summary['performance_insights']['worst_criteria']}")

EOF

    if [[ $? -eq 0 ]]; then
        print_success "✅ Comprehensive summary created for $model_name"
    else
        print_error "❌ Failed to create comprehensive summary for $model_name"
        exit 1
    fi
done

# Final Summary
print_status ""
print_status "Evaluation Summary"
print_status "=================="
print_success "Successful evaluations: $SUCCESSFUL_EVALUATIONS"
if [[ $FAILED_EVALUATIONS -gt 0 ]]; then
    print_warning "Failed evaluations: $FAILED_EVALUATIONS"
fi

if [[ $SUCCESSFUL_EVALUATIONS -eq 0 ]]; then
    print_error "No evaluations completed successfully!"
    exit 1
fi

# Show output locations
print_status ""
print_status "Output Files:"
print_status "============="
print_status "📁 Prepared data: $PREP_OUTPUT_DIR/ (${#SCORING_FILES[@]} files)"
print_status "📁 Individual results: $RESULTS_INDIVIDUAL_DIR/"
print_status "📁 Comprehensive summaries: $RESULTS_COMPREHENSIVE_DIR/"

# List generated result files by model
print_status ""
print_status "Generated result files by model:"
for model_name in "${!MODEL_RESULTS[@]}"; do
    result_files=(${MODEL_RESULTS[$model_name]})
    model_criteria_count=${#result_files[@]}
    print_status "  📊 $model_name ($model_criteria_count criteria):"
    
    for result_file in "${result_files[@]}"; do
        if [[ -n "$result_file" ]]; then
            filename=$(basename "$result_file")
            # Extract criteria from filename
            criteria=$(echo "$filename" | sed "s/${model_name}_//" | sed 's/_scores\.json$//')
            print_status "    ✅ $criteria"
        fi
    done
    
    comprehensive_file="${model_name}_comprehensive_summary_scoring.json"
    if [[ -f "$RESULTS_COMPREHENSIVE_DIR/$comprehensive_file" ]]; then
        print_status "    📋 Comprehensive summary: $comprehensive_file"
    fi
done

print_status ""
print_success "🎉 Scoring auto evaluation completed!"
print_status "Review the comprehensive summaries to compare model performance across criteria."
print_status ""
print_status "💡 Next Steps:"
print_status "  1. Check comprehensive summaries for overall model performance"
print_status "  2. Compare average scores across models and criteria"
print_status "  3. Identify best/worst performing criteria for each model"
print_status "  4. Review individual results for detailed score analysis"
print_status "  5. Look for patterns in quality breakdowns (high/average/low scores)"