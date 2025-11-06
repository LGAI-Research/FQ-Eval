# FQ-Eval
This is an official repository for **FQ-Eval Evaluation Dataset** built by **LG AI Research**, a user-centered evaluation dataset designed for assessing follow-up question generation in chat-LLM services, as described in *["FQ-Eval: Building Evaluation Dataset for User-centered Follow-up Question Generation (Seo et al., EMNLP 2025)"](https://aclanthology.org/2025.emnlp-industry.188/)*. FQ-Eval is developed by incorporating realistic chat-LLM usage scenarios and five distinct human-aligned criteria, each reflecting user expectations of effective follow-up questions. For the details regarding dataset construction and various experimental results, please refer to the official paper.

In this repository, we provide:
1) The official FQ-Eval dataset (`dataset/FQ-Eval.json`) and the evaluation-ready version, processed CSV file (`raw_data/fq_eval.csv`)
2) A comprehensive evaluation framework for assessing the quality of follow-up questions across multiple criteria using both scoring and n-way selection methods
3) Sample results and usage examples for real-case adoption

## FQ-Eval Dataset

### Original Dataset
The `FQ-Eval.json` file (`dataset/FQ-Eval.json`) contains the complete, original FQ-Eval dataset with all metadata and annotations. This is the authoritative version of our dataset used in the research paper.

### FQ-Eval Example 
Here's an example from FQ-Eval:

| **Field** | **Content** |
|-----------|-------------|
| **Category** | Creativity & Recreation |
| **Use Case** | Entertain Kids |
| **Question** | Can you make up a short bedtime story for my 5-year-old who loves dinosaurs? |
| **Answer** | Of course! Here is a short, gentle bedtime story for your dinosaur-loving 5-year-old.<br>*Daisy the Dinosaur's Sleepy Adventure*<br>... *(truncated)* |

**Criteria and Representative Follow-up Questions:**

1. **Exploratory Scope**: How can I make the bedtime story more interactive so my child can participate as we read?

2. **Contextual Relevance**: How can I use ideas from the story, like counting stars or listening to crickets, to help my child fall asleep?

3. **Creative Leap**: Can you tell a bedtime story where a dinosaur and a dragon become best friends and share an adventure?

4. **LLM Enablement**: How can I adapt AI-generated stories to be appropriate for both younger and slightly older kids?

5. **Guided Onboarding**: What are some age-appropriate adventure themes for bedtime stories for 5-year-olds?


**Complete JSON Structure:**
```json
{
  "set_id": "s_0001",
  "category": "Personal & Professional Support",
  "use_case": "Therapy & companionship",
  "definition": "Therapy provides emotional support ... ",
        "user_quotes": "['Too many people are lonely nowadays ...] ",
        "qa_pairs": [
            {
                "question_id": "q_0001",
                "question": "Can you give me a couple of reminders about things I've accomplished this week?",
                "complexity": "low",
                "intent": "generate positive affirmation",
                "topic": "self-esteem boost",
                "answer": "Of course! Here are ... ",
                "follow_up_questions": [
                    {
                        "fq_id": "fq_0001",
                        "criteria_name": "Exploratory Scope",
                        "criteria_description": "Definition: This criteria evaluates ... ",
                        "follow_up_question": "Can you help me identify some achievements from this week that I might have overlooked or not given myself credit for?"
                    },
                    {
                        "fq_id": "fq_0002",
                        "criteria_name": "Contextual Relevance",
                        "criteria_description": "Definition: Evaluates whether ... ",
                        "follow_up_question": "Can you help me list the specific things I accomplished each day this week?"
                    },
                    {
                        "fq_id": "fq_0003",
                        "criteria_name": "Creative Leap",
                        "criteria_description": "Definition: Measures whether ... ",
                        "follow_up_question": "If my week’s achievements had a movie soundtrack, what songs would be on it and why?"
                    },
                    {
                        "fq_id": "fq_0004",
                        "criteria_name": "LLM Enablement",
                        "criteria_description": "Definition: Evaluates how ... ",
                        "follow_up_question": "Can you help me create a weekly progress report that I can update with you each week?"
                    },
                    {
                        "fq_id": "fq_0005",
                        "criteria_name": "Guided Onboarding",
                        "criteria_description": "Definition: Assesses whether ... ",
                        "follow_up_question": "What are the key concepts or terms I should know to better recognize and track my weekly accomplishments?"
                    }
                ]
            },
        ]
        // ... other questions
}
```

## Evaluation Framework Overview

This framework provides two complementary evaluation approaches:
- **Scoring Evaluation**: Absolute quality assessment (1-5 scale) for individual follow-up questions across each criterion
- **N-way Evaluation**: Relative comparison to determine the best follow-up question among multiple candidates per each criterion

Both evaluations assess follow-up questions across 5 key criteria:
- **Contextual Relevance**
- **Creative Leap**
- **Exploratory Scope**
- **Guided Onboarding**
- **LLM Enablement**

(For detailed explanations of each criterion, please refer to our paper.)

The evaluation framework works with 3 main phases:
**1. Preparing the raw CSV file**
**2. Running evaluations**
**3. Analyzing results**

## Data Preparation

To run the evaluation, you must prepare a CSV file based on the following:

### Base CSV File (FQ-Eval) Format

We provide the base FQ-Eval CSV file to work as a base template for evaluation at `raw_data/fq_eval.csv`.
The `raw_data/fq_eval.csv` file is a flattened, processed version of `FQ-Eval.json`, specifically formatted for evaluation. This CSV structure allows users to easily add their own follow-up question columns.

**Key Differences from JSON File:**
- Flattened structure: Each question-criteria combination becomes a separate row
- Simplified columns: Essential fields only (question, answer, criteria, benchmark follow-up)
- Evaluation-optimized: Ready for direct use with our evaluation scripts
- User-extensible: Easy to add new columns for your own follow-up questions

**Note:** The processed CSV file includes only the essential columns needed for evaluation, so it excludes various metadata (such as category, intent, etc.) present in the original dataset JSON file. For extended analysis using such metadata, refer to the original dataset JSON file.

```csv
set_id,question_id,question,complexity,answer,criteria_name,criteria_desc,fq_id,final_fq
s_0001,q_0001,"Can you give me reminders about my accomplishments?",low,"Of course! Here are reminders...",Contextual Relevance,"Definition: Evaluates whether...",fq_0001,"Can you help me list specific accomplishments?"
```

The evaluation framework expects a CSV file with the following base columns (as in `fq_eval.csv`):

**Base Columns:**
- `set_id`: Unique identifier for question sets
- `question_id`: Unique identifier for each question
- `question`: The original user question
- `complexity`: Question difficulty level (low/high)
- `answer`: AI model's response to the question (gpt-4.1)
- `criteria_name`: Evaluation criterion (one of the 5 criteria)
- `criteria_desc`: Detailed description of the evaluation criterion
- `fq_id`: Unique identifier for the follow-up question
- `final_fq`: Benchmark follow-up question (human-written baseline from FQ-Eval)

### Data Preparation: Adding Your Follow-up Questions

To evaluate your own follow-up questions, you must prepare your own CSV file, by adding additional columns to the base CSV (next to the 'final_fq' column):

```csv
set_id,question_id,question,complexity,answer,criteria_name,criteria_desc,fq_id,final_fq,my_fq1,my_fq2,gpt-4,claude-opus
s_0001,q_0001,"Can you give me reminders?",low,"Of course!...",Contextual Relevance,"Definition...",fq_0001,"Can you help me list...","What did you do?","How was your week?","Can you specify which accomplishments?","What achievements matter most?"
```

**Required Columns:**
- `set_id` through `final_fq`: Same as the base file
- `my_fq1`: Your own follow-up questions or follow-up questions generated from other models that you wish to run the evaluation on
- `my_fq2`: Your own follow-up questions or follow-up questions generated from other models that you wish to run the evaluation on
- Add more columns that suit your needs.

**Important Notes:**
- Each row represents one question evaluated against one criterion (1,000 total, 200 per criterion)
- There should be no missing values in your CSV file
- Added column names must be exact (used as `--user_models` or `--candidates` parameters in the evaluation scripts)
- We provide a base CSV file in `raw_data/` directory as a template

Simply add new columns after the 'final_fq' column, using any column names you choose, and fill them with your follow-up questions for evaluation.

**💡 Quick Reference**: See `raw_data/sample_data_raw.csv` for a complete example of properly formatted evaluation-ready raw data with added columns.

## Setup & Installation

### Prerequisites
- Python 3.7+
- OpenAI API key

### Installation

```bash
git clone https://github.com/LG-AI-EXAONE/FQ-Eval
cd FQ-Eval
bash setup.sh
```

## Quick Start

### 1. Prepare Your Data
1. Start with the provided base CSV file: `raw_data/fq_eval.csv`
2. Add columns with follow-up questions from your models or frameworks (see `raw_data/sample_data_raw.csv` for example format)
3. Save the new, updated CSV file to use for your evaluation (e.g., `raw_data/my_prepared_data.csv`)
4. Ensure exact column names (you'll use these in commands)

### 2. Set Your API Key
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run Scoring Evaluation
```bash
./scoring_auto_eval.sh \
  --input raw_data/my_prepared_data.csv \
  --user_models final_fq,my_fq1,my_fq2,my_fq3 \
  --key $OPENAI_API_KEY
```

### 4. Run N-way Evaluation
```bash
./nway_auto_eval.sh \
  --input raw_data/my_prepared_data.csv \
  --candidates final_fq,my_fq1,my_fq2,my_fq3 \
  --key $OPENAI_API_KEY
```
### Alternative: Test with Sample Data
To test the framework immediately:
```bash
./scoring_auto_eval.sh \
  --input raw_data/sample_data_raw.csv \
  --user_models my_fq1,my_fq2,my_fq3,my_fq4 \
  --key $OPENAI_API_KEY \
  --sample_size 10
```

## Evaluation Scripts

### Scoring Evaluation (`scoring_auto_eval.sh`)

**Purpose**: Evaluates each follow-up question individually on a 1-5 scale across all criteria.

**What it does**:
1. Prepares data by creating separate JSON files for each column-criteria combination
2. Runs GPT-based evaluation using criteria-specific prompts
3. Generates individual scores with detailed reasoning
4. Creates comprehensive summaries with performance analytics

**Sample Command**:
```bash
./scoring_auto_eval.sh \
  --input raw_data/my_prepared_data.csv \
  --user_models my_fq1,my_fq2 \
  --key $OPENAI_API_KEY \
  --parallel 5 \
  --sample_size 50
```

**Key Options**:
- `--user_models`: Exact column names from your CSV (comma-separated). The columns you've added, or the columns you wish to run the evaluation on
- `--parallel`: Number of parallel API calls (default: 5)
- `--sample_size`: Limit evaluation to first N questions (for testing). For full evaluation, omit this parameter
- `--model`: OpenAI model to use for evaluation (default: gpt-4.1-2025-04-14)

### N-way Evaluation (`nway_auto_eval.sh`)

**Purpose**: Compares multiple follow-up questions head-to-head to determine the single best option.

**What it does**:
1. Prepares data with anonymized, randomized candidate comparisons
2. Runs comparative evaluation where GPT selects the best follow-up question
3. Tracks win/loss rates and performance rankings
4. Provides detailed performance breakdowns by model

**Sample Command**:
```bash
./nway_auto_eval.sh \
  --input raw_data/my_prepared_data.csv \
  --candidates final_fq,my_fq1,my_fq2 \
  --key $OPENAI_API_KEY \
  --parallel 3 \
  --sample_size 20
```

**Key Options**:
- `--candidates`: Exact column names from your CSV (minimum 2 required). The columns you wish to use as the evaluation candidates
- `--parallel`: Number of parallel API calls (default: 5)
- `--sample_size`: Limit evaluation to first N questions (for testing). For full evaluation, omit this parameter

## Example Results

### Scoring Evaluation Output

**Individual Results** (`results/scoring/individual/`):
```json
{
    "question_id": "q_0001",
    "question": "Can you give me a couple of reminders about things I've accomplished this week?",
    "answer": "Of course! Here are a couple of reminders about things you might have accomplished this week ... ",
    "criteria_name": "Contextual Relevance",
    "criteria_desc": "Definition: Evaluates whether the follow-up question maintains consistency with ... ",
    "fq_id": "fq_0002",
    "follow_up_question": "Can you help me list the specific things I accomplished each day this week?",
    "complexity": "low",
    "set_id": "s_0001",
    "evaluation": {
      "score": 4,
      "reason": "The follow-up question demonstrates strong contextual relevance. It directly builds on ... ",
      "raw_response": "```json\n{\n  \"fq_id\": \"fq_0002\",\n  \"follow_up_question\": \"Can you help me list the specific things ..."
    }
}
```

**Comprehensive Summary** (`results/scoring/summary/`):
```json
{
  "model_name": "final_fq",
  "evaluation_metadata": {
    "generated_at": "2025-08-12T09:56:44.016834+00:00",
    "total_files_processed": 5,
    "criteria_list": [
      "contextual_relevance",
      "creative_leap",
      "exploratory_scope",
      "guided_onboarding",
      "llm_enablement"
    ],
    "evaluation_type": "scoring",
    "score_scale": "1-5"
  },
  "overall_performance": {
    "total_evaluations": 1000,
    "valid_scores": 1000,
    "success_rate": 100.0,
    "average_score": 4.367,
    "min_score": 2,
    "max_score": 5,
    "score_distribution": {
      "1": 0,
      "2": 2,
      "3": 60,
      "4": 507,
      "5": 431
    },
  }
  // ...
}
```

### N-way Evaluation Output

**Individual Results** (`results/nway/individual/`):
```json
{
    "question_id": "q_0025",
    "user_question": "I’m blanking on the name of this woodworking joint. ... ",
    "criteria_name": "Creative Leap",
    "complexity": "low",
    "original_fq_id": "fq_0123",
    "candidates": [
      {
        "id": "candidate_C",
        "text": "How do I make a box joint for a drawer?"
      },
      {
        "id": "candidate_E",
        "text": "How can I create a box joint for my woodworking project?"
      },
      {
        "id": "candidate_D",
        "text": "What tools do I need to make box joints, and is there a simple method for beginners?"
      },
      {
        "id": "candidate_B",
        "text": "How can I make accurate box joints at home? What tools do I need and do you have any tips?"
      },
      {
        "id": "candidate_A",
        "text": "Could you design a completely new type of woodworking joint, taking inspiration from the box joint, but with a unique twist that hasn’t been done before?"
      }
    ],
    "mapping": {
      "candidate_C": {
        "original_id": "fq_0123_my_fq3",
        "original_type": "my_fq3"
      },
      "candidate_E": {
        "original_id": "fq_0123_my_fq4",
        "original_type": "my_fq4"
      },
      "candidate_D": {
        "original_id": "fq_0123_my_fq2",
        "original_type": "my_fq2"
      },
      "candidate_B": {
        "original_id": "fq_0123_my_fq1",
        "original_type": "my_fq1"
      },
      "candidate_A": {
        "original_id": "fq_0123_final_fq",
        "original_type": "bench"
      }
    },
    "evaluation": {
      "winner": "candidate_A",
      "winner_valid": true,
      "winner_type": "bench",
      "winner_details": {
        "original_id": "fq_0123_final_fq",
        "original_type": "bench"
      },
      "reason": "Candidate_A stands out as the clear winner for Creative Leap because ... ",
      "raw_response": "```json ... ",
      "parse_method": "json",
      "num_candidates": 5
    }
}
...
```

**Comprehensive Summary** (`results/nway/summary/`):
```json
{
  "total_evaluations": 200,
  "valid_results": 200,
  "invalid_results": 0,
  "winner_counts": {
    "bench": 156,
    "my_fq2": 15,
    "my_fq3": 6,
    "my_fq4": 17,
    "my_fq1": 6
  },
  "entity_performance": {
    "my_fq4": {
      "total_appearances": 200,
      "wins": 17,
      "losses": 183,
      "invalid": 0
    },
    "my_fq1": {
      "total_appearances": 200,
      "wins": 6,
      "losses": 194,
      "invalid": 0
    },
    "my_fq3": {
      "total_appearances": 200,
      "wins": 6,
      "losses": 194,
      "invalid": 0
    },
    "my_fq2": {
      "total_appearances": 200,
      "wins": 15,
      "losses": 185,
      "invalid": 0
    },
    "bench": {
      "total_appearances": 200,
      "wins": 156,
      "losses": 44,
      "invalid": 0
    }
  },
  "candidate_count_distribution": {
    "5": 200
  }
}
```

## Advanced Usage

### Individual Python Scripts

For detailed customization, you can run the Python scripts directly:

#### Data Preparation Scripts

**Scoring Data Preparation**:
```bash
python scripts/scoring/prepare_scoring_data.py \
  --input raw_data/my_prepared_data.csv \
  --user_models my_fq1,my_fq2 \
  --output_dir custom_scoring_data/
```

**N-way Data Preparation**:
```bash
python scripts/nway/prepare_nway_data.py \
  --input raw_data/my_prepared_data.csv \
  --candidates final_fq,my_fq1,my_fq2 \
  --output_dir custom_nway_data/ \
  --seed 42
```

#### Evaluation Scripts

**Scoring Evaluation**:
```bash
python scripts/scoring/score_evaluation.py \
  --prompt_fp prompts/scoring/scoringprompt_contextual_relevance.txt \
  --data_fp custom_scoring_data/my_fq1_contextual_relevance_scoring.json \
  --save_fp results/my_fq1_contextual_relevance_scores.json \
  --key $OPENAI_API_KEY \
  --model gpt-4.1-2025-04-14 \
  --parallel 5
```

**N-way Evaluation**:
```bash
python scripts/nway/nway_evaluation.py \
  --prompt_fp prompts/nway/nwayprompt_creative_leap.txt \
  --data_fp prepared_data/creative_leap_evaluation_ready.json \
  --save_fp custom_nway_data/creative_leap_results.json \
  --key $OPENAI_API_KEY \
  --parallel 3
```

## Output Structure

```
├── prepped_data/
│   ├── scoring/          # Prepared data for scoring evaluation
│   └── nway/             # Prepared data for n-way evaluation
├── results/
│   ├── scoring/
│   │   ├── individual/   # Individual score files per model-criteria
│   │   └── summary/      # Comprehensive per-model summaries
│   └── nway/
│       ├── individual/   # Individual comparison results per criteria
│       └── summary/      # Win/loss statistics and summaries
├── prompts/
│   ├── scoring/          # Criteria-specific scoring prompts
│   └── nway/             # Criteria-specific comparison prompts
└── raw_data/             # Your input CSV files
```

## Notes

- **Column Names**: Must match exactly between CSV and command arguments
- **API Costs**: Evaluate costs before running large datasets (each question uses 1-2 API calls)
- **Parallel Processing**: Higher parallelism = faster but more API rate limit risk
- **Sample Testing**: Always test with `--sample_size` before full runs
- **Model Names**: Any valid OpenAI model name (default: gpt-4.1-2025-04-14)


## Citation

If you use FQ-Eval in your research, please cite our paper:
- [FQ-Eval: Building Evaluation Dataset for User-centered Follow-up Question Generation](https://aclanthology.org/2025.emnlp-industry.188/)
- Sanghyun Seo, Bumsoo Kang, Dahm Lee, Jaeheon Kim, Joongbo Shin, Eui Soon Kim, and Kijeong Jeon. 2025. FQ-Eval: Building Evaluation Dataset for User-centered Follow-up Question Generation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 2811–2827, Suzhou (China). Association for Computational Linguistics.

```bibtex
@inproceedings{seo-etal-2025-fq,
    title = "{FQ}-Eval: Building Evaluation Dataset for User-centered Follow-up Question Generation",
    author = "Seo, Sanghyun  and
      Kang, Bumsoo  and
      Lee, Dahm  and
      Kim, Jaeheon  and
      Shin, Joongbo  and
      Kim, Eui Soon  and
      Jeon, Kijeong",
    editor = "Potdar, Saloni  and
      Rojas-Barahona, Lina  and
      Montella, Sebastien",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track",
    month = nov,
    year = "2025",
    address = "Suzhou (China)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-industry.188/",
    pages = "2811--2827",
    ISBN = "979-8-89176-333-3",
    abstract = "To effectively support users' goal achievement in chat-LLM services, providing user-centered follow-up questions is essential. Existing studies primarily focus on enhancing information-seeking or topical relevance, often missing how follow-up questions could satisfy users' intrinsic needs and conversational goals. To bridge this gap, we introduce FQ-Eval, a user-centered evaluation dataset designed for assessing follow-up question generation in chat-LLM services. FQ-Eval incorporates realistic chat-LLM usage scenarios and five distinct human-aligned criteria, each reflecting user expectations of effective follow-up questions. Experimental results show that FQ-Eval constructed through our approach clearly capture human-aligned criteria, enabling robust, human-aligned follow-up question generation evaluation of various models and services."
}
```

## License

- **Code**: Licensed under [BSD-3-Clause-LG AI Research License](./License.md)
- **Dataset**: Licensed under [CC-BY-NC-4.0](./License-Dataset.md)

