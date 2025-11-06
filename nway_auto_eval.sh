#!/bin/bash

# N-way Auto Evaluation Script
# Automatically prepares data and runs n-way evaluations for all criteria

# Note: We don't use 'set -e' here because it can interfere with the evaluation loop

# Default values
DEFAULT_MODEL="gpt-4.1-2025-04-14"
DEFAULT_PARALLEL=5
DEFAULT_TEMPERATURE=0.0
DEFAULT_N=1
DEFAULT_SEED=42

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
N-way Auto Evaluation Script

USAGE:
    $0 --input <csv_file> --candidates <columns> --key <api_key> [OPTIONS]

REQUIRED ARGUMENTS:
    --input       Path to input CSV file (in raw_data/ directory)
    --candidates  Comma-separated list of candidate columns (e.g., final_fq,my_model1,my_model2)
    --key         OpenAI API key

OPTIONAL ARGUMENTS:
    --model       OpenAI model to use (default: $DEFAULT_MODEL)
    --parallel    Number of parallel API calls (default: $DEFAULT_PARALLEL)
    --sample_size Process only first N items for testing (default: all)
    --help        Show this help message

EXAMPLES:
    # Basic usage
    $0 --input raw_data/my_data.csv --candidates final_fq,my_model1,my_model2 --key sk-your-key

    # With custom model and parallel processing
    $0 --input raw_data/my_data.csv --candidates my_model1,my_model2 --key sk-your-key --model gpt-4 --parallel 3

    # Testing with limited data
    $0 --input raw_data/my_data.csv --candidates final_fq,my_model --key sk-your-key --sample_size 20

OUTPUT STRUCTURE:
    prepped_data/nway/           - Prepared evaluation data
    results/nway/individual/     - Individual evaluation results
    results/nway/summary/        - Summary statistics

EOF
}

# Parse arguments
INPUT_FILE=""
CANDIDATES=""
API_KEY=""
MODEL="$DEFAULT_MODEL"
PARALLEL="$DEFAULT_PARALLEL"
SAMPLE_SIZE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --candidates)
            CANDIDATES="$2"
            shift 2
            ;;
        --key)
            API_KEY="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --sample_size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown argument: $1"
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

if [[ -z "$CANDIDATES" ]]; then
    print_error "Missing required argument: --candidates"
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

# Validate candidate format (at least 2 candidates)
IFS=',' read -ra CANDIDATE_ARRAY <<< "$CANDIDATES"
if [[ ${#CANDIDATE_ARRAY[@]} -lt 2 ]]; then
    print_error "Need at least 2 candidates for comparison, got ${#CANDIDATE_ARRAY[@]}: $CANDIDATES"
    exit 1
fi

# Define all criteria and their corresponding prompt files (moved up here)
declare -A CRITERIA_PROMPTS=(
    ["contextual_relevance"]="prompts/nway/nwayprompt_contextual_relevance.txt"
    ["creative_leap"]="prompts/nway/nwayprompt_creative_leap.txt"
    ["exploratory_scope"]="prompts/nway/nwayprompt_exploratory_scope.txt"
    ["guided_onboarding"]="prompts/nway/nwayprompt_guided_onboarding.txt"
    ["llm_enablement"]="prompts/nway/nwayprompt_llm_enablement.txt"
)

print_status "Starting N-way Auto Evaluation"
print_status "================================"
print_status "Input file: $INPUT_FILE"
print_status "Candidates: $CANDIDATES (${#CANDIDATE_ARRAY[@]} total)"
print_status "Model: $MODEL"
print_status "Parallel workers: $PARALLEL"
if [[ -n "$SAMPLE_SIZE" ]]; then
    print_status "Sample size: $SAMPLE_SIZE (testing mode)"
fi

# Step 1: Data Preparation
print_status ""
print_status "Step 1: Preparing evaluation data..."
print_status "====================================="

PREP_OUTPUT_DIR="prepped_data/nway"
mkdir -p "$PREP_OUTPUT_DIR"

PREP_CMD="python scripts/nway/prepare_nway_data.py \
    --input \"$INPUT_FILE\" \
    --candidates \"$CANDIDATES\" \
    --output_dir \"$PREP_OUTPUT_DIR\" \
    --seed $DEFAULT_SEED"

print_status "Running: $PREP_CMD"

if ! eval $PREP_CMD; then
    print_error "Data preparation failed!"
    exit 1
fi

print_success "Data preparation completed!"

# Build list of expected files based on current candidates
candidate_suffix=$(echo "$CANDIDATES" | tr ',' '_' | tr '[:upper:]' '[:lower:]')
EVAL_FILES=()
for criteria in "${!CRITERIA_PROMPTS[@]}"; do
    # Use wildcard to match any candidate suffix pattern
    matching_files=("$PREP_OUTPUT_DIR"/${criteria}_*_evaluation_ready.json)
    
    if [[ -f "${matching_files[0]}" ]]; then
        EVAL_FILES+=("${matching_files[0]}")
    else
        print_warning "Missing evaluation file for criteria: $criteria"
    fi
done

if [[ ${#EVAL_FILES[@]} -eq 0 ]]; then
    print_error "No evaluation files were generated. Check your input data and candidate columns."
    exit 1
fi

print_status "Found ${#EVAL_FILES[@]} evaluation files for specified candidates:"
for file in "${EVAL_FILES[@]}"; do
    filename=$(basename "$file")
    # Extract criteria by checking against known criteria list
    criteria=""
    for known_criteria in "${!CRITERIA_PROMPTS[@]}"; do
        if [[ "$filename" == "${known_criteria}_"*"_evaluation_ready.json" ]]; then
            criteria="$known_criteria"
            break
        fi
    done
    if [[ -n "$criteria" ]]; then
        print_status "  - $criteria"
    else
        print_status "  - $(basename "$file")"
    fi
done

# Step 2: Run Evaluations
print_status ""
print_status "Step 2: Running evaluations for all criteria..."
print_status "=============================================="

# Create results directories
RESULTS_INDIVIDUAL_DIR="results/nway/individual"
RESULTS_SUMMARY_DIR="results/nway/summary"
mkdir -p "$RESULTS_INDIVIDUAL_DIR"
mkdir -p "$RESULTS_SUMMARY_DIR"

SUCCESSFUL_EVALUATIONS=0
FAILED_EVALUATIONS=0

# Process each evaluation file
for eval_file in "${EVAL_FILES[@]}"; do
    if [[ ! -f "$eval_file" ]]; then
        print_warning "Evaluation file not found: $eval_file"
        ((FAILED_EVALUATIONS++))
        continue
    fi
    
    # Extract criteria name from filename using the known criteria list
    filename=$(basename "$eval_file")
    criteria=""
    for known_criteria in "${!CRITERIA_PROMPTS[@]}"; do
        if [[ "$filename" == "${known_criteria}_"*"_evaluation_ready.json" ]]; then
            criteria="$known_criteria"
            break
        fi
    done
    
    # print_status "DEBUG: filename=$filename, extracted criteria=$criteria"

    if [[ -z "$criteria" ]]; then
        print_warning "Could not extract criteria from filename: $filename. Skipping..."
        ((FAILED_EVALUATIONS++))
        continue
    fi
    
    print_status ""
    print_status "Processing: $criteria"
    print_status "Input file: $(basename "$eval_file")"
    print_status "------------------------"
    
    # Check if prompt file exists
    prompt_file="${CRITERIA_PROMPTS[$criteria]}"
    if [[ -z "$prompt_file" ]]; then
        print_warning "No prompt mapping found for criteria: $criteria. Skipping..."
        ((FAILED_EVALUATIONS++))
        continue
    fi
    
    if [[ ! -f "$prompt_file" ]]; then
        print_warning "Prompt file not found: $prompt_file. Skipping $criteria..."
        ((FAILED_EVALUATIONS++))
        continue
    fi
    
    print_status "Using prompt: $prompt_file"

    # Setup output paths
    result_file="$RESULTS_INDIVIDUAL_DIR/${criteria}_results.json"
    
    # Build evaluation command
    EVAL_CMD="python scripts/nway/nway_evaluation.py \
        --prompt_fp \"$prompt_file\" \
        --data_fp \"$eval_file\" \
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
    
    print_status "Running evaluation for $criteria..."
    
    # Run evaluation with error handling
    if eval $EVAL_CMD 2>&1; then
        print_success "✅ $criteria evaluation completed"
        ((SUCCESSFUL_EVALUATIONS++))
        
        # Move summary file to summary directory if it exists
        summary_file="${result_file%.*}_summary.json"
        if [[ -f "$summary_file" ]]; then
            mv "$summary_file" "$RESULTS_SUMMARY_DIR/${criteria}_summary.json"
        fi
    else
        print_error "❌ $criteria evaluation failed"
        print_error "Command that failed: $EVAL_CMD"
        ((FAILED_EVALUATIONS++))
    fi
    
    # Progress indicator
    print_status "Progress: $SUCCESSFUL_EVALUATIONS/${#EVAL_FILES[@]} evaluations completed"
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
print_status "📁 Prepared data: $PREP_OUTPUT_DIR/ (${#EVAL_FILES[@]} files)"
print_status "📁 Individual results: $RESULTS_INDIVIDUAL_DIR/"
print_status "📁 Summary statistics: $RESULTS_SUMMARY_DIR/"

# List generated result files
print_status ""
print_status "Generated result files:"
for result_file in "$RESULTS_INDIVIDUAL_DIR"/*.json; do
    if [[ -f "$result_file" ]]; then
        filename=$(basename "$result_file")
        criteria=$(echo "$filename" | sed 's/_results\.json$//')
        print_status "  ✅ $criteria"
    fi
done

print_status ""
print_success "🎉 N-way auto evaluation completed!"
print_status "Review the results in the output directories above."