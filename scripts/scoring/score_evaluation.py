#!/usr/bin/env python3
from openai import OpenAI
import json
import argparse
import time
import re
from tqdm import tqdm
from collections import defaultdict
import concurrent.futures
import os

def process_single_evaluation(instance, client, model, template, temperature, n):
    """Process a single scoring evaluation and return the result."""
    # Build prompt with standard field mapping
    prompt = (
        template
          .replace('{user_question}',      str(instance.get('question', '')))
          .replace('{answer}',             str(instance.get('answer', '')))
          .replace('{fq_id}',              str(instance.get('fq_id', '')))
          .replace('{follow_up_question}', str(instance.get('follow_up_question', '')))
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            temperature=temperature,
            max_tokens=1000,
            n=n
        )
        text = resp.choices[0].message.content.strip()

        # Try to parse JSON response
        try:
            result_json = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: extract JSON object
            json_match = re.search(r'\{[^{}]*"fq_id"[^{}]*\}', text, re.DOTALL)
            if json_match:
                try:
                    result_json = json.loads(json_match.group())
                except json.JSONDecodeError:
                    print(f"  ⚠️ Could not parse JSON from response for fq_id: {instance.get('fq_id', 'unknown')}")
                    result_json = None
            else:
                print(f"  ⚠️ No JSON found in response for fq_id: {instance.get('fq_id', 'unknown')}")
                result_json = None

        if result_json:
            # Validate score is in 1-5 range
            score = result_json.get('score')
            if score is not None and (score < 1 or score > 5):
                print(f"  ⚠️ Warning: Score {score} out of range (1-5) for fq_id: {instance.get('fq_id', 'unknown')}")
                
            instance['evaluation'] = {
                'score':        score,
                'reason':       result_json.get('reason'),
                'raw_response': text
            }
        else:
            # Manual fallback parsing
            score_match = re.search(r'"score"\s*:\s*(\d+\.?\d*)', text)
            reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text, re.DOTALL)
            
            score = float(score_match.group(1)) if score_match else None
            if score is not None and (score < 1 or score > 5):
                print(f"  ⚠️ Warning: Parsed score {score} out of range (1-5) for fq_id: {instance.get('fq_id', 'unknown')}")
            
            instance['evaluation'] = {
                'score':        score,
                'reason':       reason_match.group(1) if reason_match else text,
                'raw_response': text,
                'parse_error':  True
            }

    except Exception as e:
        print(f"  ⚠️ Error for fq_id {instance.get('fq_id', 'unknown')}: {e}")
        instance['evaluation'] = {
            'score':        None,
            'reason':       None,
            'error':        str(e),
            'raw_response': None
        }

    return instance

def main():
    parser = argparse.ArgumentParser(
        description="Run G-Eval for follow-up question scoring evaluation"
    )
    parser.add_argument('--prompt_fp',   type=str, required=True, help='Path to your prompt template')
    parser.add_argument('--data_fp',     type=str, required=True, help='JSON array with evaluation data')
    parser.add_argument('--save_fp',     type=str, required=True, help='Where to write the scored output')
    parser.add_argument('--key',         type=str, required=True, help='Your OpenAI API key')
    parser.add_argument('--model',       type=str, default='gpt-4.1-2025-04-14', help='Which OpenAI model to use')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature for ChatCompletion')
    parser.add_argument('--n',           type=int,   default=1,   help='Number of completions per prompt')
    parser.add_argument('--sample_size', type=int,   default=None, help='Process only first N items (for testing)')
    parser.add_argument('--parallel',    type=int,   default=1,   help='Number of parallel API calls')

    args = parser.parse_args()

    client = OpenAI(api_key=args.key)

    # Load data and template
    with open(args.data_fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(args.prompt_fp, 'r', encoding='utf-8') as f:
        template = f.read()

    # Limit data if batch_size is specified
    if args.sample_size:
        data = data[:args.sample_size]
        print(f"Processing first {args.sample_size} items only")

    # Validate data structure
    print(f"Loaded {len(data)} instances for scoring evaluation")
    if data:
        sample = data[0]
        print(f"Sample keys: {list(sample.keys())}")
        
        # Check for required fields
        required_fields = ['question', 'answer', 'fq_id', 'follow_up_question']
        missing_fields = [field for field in required_fields if field not in sample]
        if missing_fields:
            print(f"❌ Error: Missing required fields: {missing_fields}")
            print(f"Expected fields: {required_fields}")
            return 1
        else:
            print(f"✅ All required fields found")
            
        # Show sample follow-up question
        print(f"Sample follow-up question: {sample['follow_up_question'][:100]}...")
        
        # Check criteria information
        criteria_name = sample.get('criteria_name', 'Unknown')
        print(f"Evaluation criteria: {criteria_name}")

    print(f"Total evaluations to process: {len(data)}")
    
    # Process evaluations
    results = []
    
    if args.parallel > 1:
        # Parallel processing
        print(f"Using {args.parallel} parallel workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = []
            for instance in data:
                future = executor.submit(
                    process_single_evaluation,
                    instance, client, args.model, template, args.temperature, args.n
                )
                futures.append(future)
                time.sleep(0.1)  # Small delay to avoid rate limits
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), 
                             desc="Scoring follow-up questions"):
                results.append(future.result())
    else:
        # Sequential processing
        for instance in tqdm(data, desc="Scoring follow-up questions"):
            result = process_single_evaluation(
                instance, client, args.model, template, args.temperature, args.n
            )
            results.append(result)
            time.sleep(0.5)  # Rate-limit friendly

    # Calculate detailed statistics
    scores = [r['evaluation']['score'] for r in results if r['evaluation'].get('score') is not None]
    valid_scores = [s for s in scores if 1 <= s <= 5]
    
    print(f"\n📊 Scoring Statistics:")
    print(f"  - Total instances: {len(results)}")
    print(f"  - Successfully scored: {len(scores)}")
    print(f"  - Valid scores (1-5): {len(valid_scores)}")
    print(f"  - Parse errors: {sum(1 for r in results if r['evaluation'].get('parse_error', False))}")
    print(f"  - API errors: {sum(1 for r in results if r['evaluation'].get('error'))}")
    
    if valid_scores:
        print(f"\n📈 Score Analysis:")
        print(f"  - Average score: {sum(valid_scores)/len(valid_scores):.2f}")
        print(f"  - Score range: {min(valid_scores):.1f} - {max(valid_scores):.1f}")
        print(f"  - Score distribution:")
        for score in range(1, 6):
            count = valid_scores.count(score)
            percentage = (count/len(valid_scores)*100) if valid_scores else 0
            print(f"    Score {score}: {count:>3} ({percentage:>5.1f}%)")
            
        # Quality indicators
        high_scores = [s for s in valid_scores if s >= 4]
        low_scores = [s for s in valid_scores if s <= 2]
        
        print(f"\n💡 Quality Indicators:")
        print(f"  - High quality (4-5): {len(high_scores)} ({len(high_scores)/len(valid_scores)*100:.1f}%)")
        print(f"  - Low quality (1-2): {len(low_scores)} ({len(low_scores)/len(valid_scores)*100:.1f}%)")
        print(f"  - Average quality (3): {valid_scores.count(3)} ({valid_scores.count(3)/len(valid_scores)*100:.1f}%)")
    else:
        print(f"\n❌ No valid scores found!")

    # Write output
    output_dir = os.path.dirname(args.save_fp)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(args.save_fp, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)

    print(f"\n✅ Wrote {len(results)} scored instances to {args.save_fp}")
    
    # Show sample result
    if results and results[0]['evaluation'].get('score') is not None:
        sample_result = results[0]
        print(f"\n📝 Sample Result:")
        print(f"  Question ID: {sample_result.get('question_id', 'N/A')}")
        print(f"  Follow-up Question: {sample_result['follow_up_question'][:80]}...")
        print(f"  Score: {sample_result['evaluation']['score']}")
        print(f"  Reason: {sample_result['evaluation']['reason'][:100]}...")

if __name__ == '__main__':
    main()