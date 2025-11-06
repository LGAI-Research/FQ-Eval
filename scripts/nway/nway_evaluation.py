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

def validate_winner_id(winner, candidates, mapping=None):
    """Validate that winner matches one of the candidate IDs exactly and return original type."""
    candidate_ids = [c['id'] for c in candidates]
    if winner in candidate_ids:
        if mapping and winner in mapping:
            # Get original type from mapping (anonymized data)
            original_type = mapping[winner]['original_type']
            return True, original_type
        else:
            # Fallback for non-anonymized data
            for candidate in candidates:
                if candidate['id'] == winner:
                    return True, candidate.get('type', 'unknown')
            return True, 'unknown'
    return False, None

def extract_json_from_text(text):
    """Extract JSON from text with multiple fallback strategies."""
    # Strategy 1: Direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Find JSON object with "winner" field
    json_patterns = [
        r'\{[^{}]*"winner"[^{}]*\}',  # Simple object
        r'\{(?:[^{}]|\{[^{}]*\})*"winner"(?:[^{}]|\{[^{}]*\})*\}',  # Nested objects
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match)
                if 'winner' in result:
                    return result
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Manual extraction of winner field
    winner_patterns = [
        r'"winner"\s*:\s*"([^"]+)"',
        r'"winner"\s*:\s*\'([^\']+)\'',
        r'winner\s*:\s*"([^"]+)"',
    ]
    
    for pattern in winner_patterns:
        match = re.search(pattern, text)
        if match:
            # Also try to get reason
            reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text, re.DOTALL)
            return {
                'winner': match.group(1),
                'reason': reason_match.group(1) if reason_match else "Could not extract reason",
                'parse_method': 'regex'
            }
    
    return None

def format_candidates_for_prompt(candidates):
    """Format candidates list for prompt substitution."""
    candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)
    return candidates_json

def process_single_evaluation(instance, client, model, template, temperature):
    """Process a single multi-candidate evaluation and return the result."""
    # Validate input has candidates
    if not instance.get('candidates') or len(instance['candidates']) < 2:
        print(f"\n  ⚠️ Invalid input: Need at least 2 candidates, got {len(instance.get('candidates', []))}")
        instance['evaluation'] = {
            'winner': None,
            'winner_valid': False,
            'winner_type': None,
            'winner_details': None,
            'reason': None,
            'error': 'Insufficient candidates (need at least 2)',
            'raw_response': None
        }
        return instance
    
    # Format candidates for prompt
    candidates_str = format_candidates_for_prompt(instance['candidates'])
    num_candidates = len(instance['candidates'])
    
    # Build prompt with required fields
    prompt = (
        template
          .replace('{user_question}', instance.get('user_question', ''))
          .replace('{answer}', instance.get('answer', ''))
          .replace('{candidates}', candidates_str)
          .replace('{num_candidates}', str(num_candidates))
    )
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            temperature=temperature,
            max_tokens=1000,
            n=1
        )
        text = resp.choices[0].message.content.strip()
        
        # Parse response
        result_json = extract_json_from_text(text)
        
        if result_json:
            winner = result_json.get('winner')
            
            # Validate winner ID using mapping (if available)
            mapping = instance.get('mapping', {})
            is_valid, winner_type = validate_winner_id(winner, instance['candidates'], mapping)
            
            if not is_valid:
                candidate_ids = [c['id'] for c in instance['candidates']]
                print(f"\n  ⚠️ Invalid winner ID: {winner}")
                print(f"     Expected one of: {candidate_ids}")
            
            # Get winner details
            winner_details = None
            if is_valid:
                if mapping and winner in mapping:
                    # Anonymized data
                    winner_details = mapping[winner]
                else:
                    # Non-anonymized data
                    winner_details = next((c for c in instance['candidates'] if c['id'] == winner), None)
            
            instance['evaluation'] = {
                'winner': winner,
                'winner_valid': is_valid,
                'winner_type': winner_type,  # 'bench' or user model names
                'winner_details': winner_details,
                'reason': result_json.get('reason', ''),
                'raw_response': text,
                'parse_method': result_json.get('parse_method', 'json'),
                'num_candidates': num_candidates
            }
        else:
            print(f"\n  ⚠️ Could not extract winner from response")
            instance['evaluation'] = {
                'winner': None,
                'winner_valid': False,
                'winner_type': None,
                'winner_details': None,
                'reason': None,
                'raw_response': text,
                'parse_error': True,
                'num_candidates': num_candidates
            }
            
    except Exception as e:
        print(f"\n  ⚠️ Error: {str(e)}")
        instance['evaluation'] = {
            'winner': None,
            'winner_valid': False,
            'winner_type': None,
            'winner_details': None,
            'reason': None,
            'error': str(e),
            'raw_response': None,
            'num_candidates': num_candidates
        }
    
    return instance

def main():
    parser = argparse.ArgumentParser(
        description="Run Multi-Candidate G-Eval for follow-up question evaluation"
    )
    parser.add_argument('--prompt_fp',   type=str, required=True, help='Path to your prompt template')
    parser.add_argument('--data_fp',     type=str, required=True, help='JSON array with multi-candidate data')
    parser.add_argument('--save_fp',     type=str, required=True, help='Where to write the evaluation output')
    parser.add_argument('--key',         type=str, required=True, help='Your OpenAI API key')
    parser.add_argument('--model',       type=str, default='gpt-4.1-2025-04-14', help='Which OpenAI model to use')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature for ChatCompletion')
    parser.add_argument('--n',           type=int,   default=1,   help='Number of completions per prompt')
    parser.add_argument('--sample_size',  type=int,   default=None, help='Process only first N items (for testing)')
    parser.add_argument('--parallel',    type=int,   default=1,   help='Number of parallel API calls')
    args = parser.parse_args()

    client = OpenAI(api_key=args.key)

    # Load data and template
    with open(args.data_fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(args.prompt_fp, 'r', encoding='utf-8') as f:
        template = f.read()

    # Validate data has candidates
    if data:
        candidate_counts = [len(instance.get('candidates', [])) for instance in data]
        min_candidates = min(candidate_counts) if candidate_counts else 0
        max_candidates = max(candidate_counts) if candidate_counts else 0
        
        print(f"📊 Candidate count distribution:")
        print(f"  - Min candidates per instance: {min_candidates}")
        print(f"  - Max candidates per instance: {max_candidates}")
        
        if min_candidates < 2:
            print(f"  ⚠️ Warning: Some instances have fewer than 2 candidates")

    # Limit data if batch_size is specified
    if args.sample_size:
        data = data[:args.sample_size]
        print(f"Processing first {args.sample_size} items only")

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
                    instance, client, args.model, template, args.temperature
                )
                futures.append(future)
                time.sleep(0.1)  # Small delay to avoid rate limits
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), 
                             desc="Evaluating candidates"):
                results.append(future.result())
    else:
        # Sequential processing
        for instance in tqdm(data, desc="Evaluating candidates"):
            result = process_single_evaluation(
                instance, client, args.model, template, args.temperature
            )
            results.append(result)
            time.sleep(0.5)  # Rate-limit friendly

    # Calculate detailed statistics
    print("\n" + "="*60)
    print("📊 MULTI-CANDIDATE EVALUATION RESULTS")
    print("="*60)
    
    # Data quality stats
    total_evaluations = len(results)
    valid_results = sum(1 for r in results if r['evaluation'].get('winner_valid', False))
    invalid_results = total_evaluations - valid_results
    
    print(f"\n📋 Data Quality:")
    print(f"  - Total evaluations: {total_evaluations}")
    print(f"  - Valid results: {valid_results} ({valid_results/total_evaluations*100:.1f}%)")
    print(f"  - Invalid results: {invalid_results} ({invalid_results/total_evaluations*100:.1f}%)")
    
    # Overall win stats (only valid results)
    winner_counts = defaultdict(int)
    for r in results:
        if r['evaluation'].get('winner_valid', False):
            winner_type = r['evaluation'].get('winner_type')
            if winner_type:
                winner_counts[winner_type] += 1
    
    print(f"\n🏆 Win Counts by Entity (valid results only):")
    if valid_results > 0:
        print(f"  Total valid evaluations: {valid_results}")
        for entity, count in sorted(winner_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / valid_results) * 100
            print(f"  - {entity}: {count} wins ({percentage:.1f}%)")
    else:
        print("  - No valid results to analyze")
    
    # Per-entity detailed statistics
    entity_stats = defaultdict(lambda: {'total_appearances': 0, 'wins': 0, 'losses': 0, 'invalid': 0})
    
    # Count appearances and wins for each entity
    for r in results:
        mapping = r.get('mapping', {})
        
        if mapping:
            # Anonymized data - use mapping
            for candidate_id, mapping_info in mapping.items():
                entity_type = mapping_info['original_type']
                if r['evaluation'].get('winner_valid', False):
                    entity_stats[entity_type]['total_appearances'] += 1
                    if r['evaluation'].get('winner') == candidate_id:
                        entity_stats[entity_type]['wins'] += 1
                    else:
                        entity_stats[entity_type]['losses'] += 1
                else:
                    entity_stats[entity_type]['invalid'] += 1
        else:
            # Non-anonymized data - use candidate type
            for candidate in r['candidates']:
                entity_type = candidate.get('type', 'unknown')
                if r['evaluation'].get('winner_valid', False):
                    entity_stats[entity_type]['total_appearances'] += 1
                    if r['evaluation'].get('winner') == candidate['id']:
                        entity_stats[entity_type]['wins'] += 1
                    else:
                        entity_stats[entity_type]['losses'] += 1
                else:
                    entity_stats[entity_type]['invalid'] += 1
    
    print(f"\n📈 Detailed Performance by Entity:")
    print(f"{'Entity':<40} {'Wins':>6} {'Losses':>6} {'Win%':>6} {'Total':>7} {'Invalid':>7}")
    print("-" * 80)
    
    for entity, stats in sorted(entity_stats.items(), 
                               key=lambda x: x[1]['wins']/x[1]['total_appearances'] if x[1]['total_appearances'] > 0 else 0, 
                               reverse=True):
        if stats['total_appearances'] > 0:
            win_rate = stats['wins'] / stats['total_appearances'] * 100
            print(f"{entity:<40} {stats['wins']:>6} {stats['losses']:>6} {win_rate:>5.1f}% {stats['total_appearances']:>7} {stats['invalid']:>7}")
        else:
            print(f"{entity:<40} {stats['wins']:>6} {stats['losses']:>6} {'N/A':>6} {0:>7} {stats['invalid']:>7}")
    
    # Candidate count distribution in results
    candidate_count_dist = defaultdict(int)
    for r in results:
        num_candidates = r['evaluation'].get('num_candidates', 0)
        candidate_count_dist[num_candidates] += 1
    
    if candidate_count_dist:
        print(f"\n📊 Candidate Count Distribution:")
        for count, frequency in sorted(candidate_count_dist.items()):
            print(f"  - {count} candidates: {frequency} instances")
    
    # Error analysis
    parse_errors = sum(1 for r in results if r['evaluation'].get('parse_error', False))
    api_errors = sum(1 for r in results if r['evaluation'].get('error'))
    
    if parse_errors > 0 or api_errors > 0:
        print(f"\n⚠️  Error Summary:")
        if parse_errors > 0:
            print(f"  - Parse errors: {parse_errors}")
        if api_errors > 0:
            print(f"  - API errors: {api_errors}")
    
    # Save results
    output_dir = os.path.dirname(args.save_fp)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(args.save_fp, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to: {args.save_fp}")
    
    # Save summary statistics
    summary_fp = args.save_fp.replace('.json', '_summary.json')
    summary = {
        'total_evaluations': total_evaluations,
        'valid_results': valid_results,
        'invalid_results': invalid_results,
        'winner_counts': dict(winner_counts),
        'entity_performance': dict(entity_stats),
        'candidate_count_distribution': dict(candidate_count_dist)
    }
    
    with open(summary_fp, 'w', encoding='utf-8') as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
    
    print(f"📊 Summary saved to: {summary_fp}")

if __name__ == '__main__':
    main()