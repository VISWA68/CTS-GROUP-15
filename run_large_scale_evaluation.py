#!/usr/bin/env python3
"""
Large Scale RAGAS Evaluation Launcher
=====================================

This script provides a command-line interface to run large-scale RAGAS evaluations
with the 50-question test dataset and configurable delays.

Usage:
    python run_large_scale_evaluation.py [--delay-minutes MINUTES] [--questions NUM]

Examples:
    # Run full 50-question evaluation with 10-minute delays
    python run_large_scale_evaluation.py
    
    # Run with 5-minute delays
    python run_large_scale_evaluation.py --delay-minutes 5
    
    # Run only first 10 questions with 8-minute delays
    python run_large_scale_evaluation.py --delay-minutes 8 --questions 10
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

from test import (
    DrugChatbotOrchestrator, 
    process_large_batch_with_delays,
    run_ragas_evaluation,
    save_ragas_results,
    generate_research_report
)

def load_test_dataset(limit: int = None) -> List[Dict[str, Any]]:
    """Load the test dataset from JSON file."""
    try:
        dataset_path = Path("test_dataset.json")
        if not dataset_path.exists():
            print("❌ Error: test_dataset.json not found!")
            print("Please ensure the test dataset file exists in the current directory.")
            return []
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            dataset = data.get("test_dataset", [])
            
        if limit and limit < len(dataset):
            dataset = dataset[:limit]
            print(f"📝 Using first {limit} questions from dataset")
        
        return dataset
    except Exception as e:
        print(f"❌ Error loading test dataset: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(
        description="Run large-scale RAGAS evaluation with configurable parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--delay-minutes",
        type=int,
        default=10,
        help="Minutes to wait between each question (default: 10)"
    )
    
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Number of questions to process (default: all 50)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results (default: ./results)"
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating the research report"
    )
    
    args = parser.parse_args()
    
    print("🔬 Large Scale RAGAS Evaluation")
    print("=" * 50)
    print(f"⏱️  Delay between questions: {args.delay_minutes} minutes")
    print(f"📊 Questions to process: {args.questions or 'All (50)'}")
    print(f"📁 Output directory: {args.output_dir}")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load test dataset
    print("📂 Loading test dataset...")
    test_dataset = load_test_dataset(args.questions)
    
    if not test_dataset:
        print("❌ No test data available. Exiting.")
        return 1
    
    print(f"✅ Loaded {len(test_dataset)} questions")
    
    # Count categories
    categories = {}
    for item in test_dataset:
        cat = item.get('category', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n📊 Question Categories:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {cat}: {count} questions")
    
    # Calculate time estimates
    total_time_hours = len(test_dataset) * args.delay_minutes / 60
    print(f"\n⏰ Estimated completion time: {total_time_hours:.1f} hours")
    
    # Confirm before starting
    if len(test_dataset) > 10:
        response = input(f"\n⚠️  This will take {total_time_hours:.1f} hours. Continue? (y/N): ")
        if response.lower() != 'y':
            print("❌ Evaluation cancelled.")
            return 0
    
    try:
        # Initialize orchestrator
        print("\n🤖 Initializing RAG system...")
        orchestrator = DrugChatbotOrchestrator()
        print("✅ RAG system initialized")
        
        # Start evaluation
        print(f"\n🚀 Starting large-scale evaluation...")
        print(f"📝 Processing {len(test_dataset)} questions with {args.delay_minutes}-minute delays")
        
        start_time = time.time()
        
        # This would normally use Streamlit, but for CLI we'll adapt
        print("⚠️  Note: This CLI version is simplified. For full progress tracking, use the Streamlit interface.")
        
        # Process questions with delays
        questions = []
        contexts = []
        answers = []
        ground_truths = []
        references = []
        
        successful_questions = 0
        failed_questions = 0
        
        for i, sample in enumerate(test_dataset):
            q = sample.get("question", "").strip()
            category = sample.get("category", "Unknown")
            gt_list = sample.get("ground_truths", [])
            if isinstance(sample.get("ground_truth"), str):
                gt_list = [sample.get("ground_truth")]
            
            if not q:
                continue
            
            print(f"\n📋 Question {i+1}/{len(test_dataset)} [{category}]")
            print(f"❓ {q}")
            
            try:
                # Extract entities and search
                entities = orchestrator.retrieval_agent.extract_entities(q, "cli_session")
                
                # Construct filter metadata
                filter_metadata = None
                if entities.get("drugs"):
                    filter_metadata = {"drug": {"$in": entities["drugs"]}}
                
                # Perform retrieval
                retrievals = orchestrator.retrieval_agent.vector_search(
                    q, top_k=8, filter_metadata=filter_metadata
                )
                
                # Filter relevant chunks
                filtered_retrievals = orchestrator.reasoning_agent.assess_chunk_relevance(q, retrievals)
                
                # Generate answer
                response = orchestrator.answer_agent.generate_final_response(
                    q, "cli_session", filtered_retrievals[:4], ""
                )
                
                # Store results
                questions.append(q)
                contexts.append([chunk["text"] for chunk in filtered_retrievals[:4]])
                answers.append(response["short_answer"])
                ground_truths.append(gt_list)
                references.append(gt_list[0] if gt_list else "")
                
                successful_questions += 1
                print(f"✅ Completed successfully")
                
                # Save intermediate results every 10 questions
                if (i + 1) % 10 == 0:
                    intermediate_file = output_dir / f"intermediate_results_{i+1}_questions.json"
                    intermediate_data = {
                        "partial_results": {
                            "questions": questions,
                            "contexts": contexts,
                            "answers": answers,
                            "ground_truths": ground_truths,
                            "references": references
                        },
                        "progress": {
                            "completed": i + 1,
                            "total": len(test_dataset),
                            "successful": successful_questions,
                            "failed": failed_questions,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    }
                    
                    with open(intermediate_file, 'w', encoding='utf-8') as f:
                        json.dump(intermediate_data, f, indent=2, ensure_ascii=False)
                    print(f"💾 Intermediate results saved: {intermediate_file}")
                
            except Exception as e:
                failed_questions += 1
                print(f"❌ Failed: {e}")
                
                # Add empty entries for consistency
                questions.append(q)
                contexts.append([""])
                answers.append(f"Error: {str(e)}")
                ground_truths.append(gt_list)
                references.append(gt_list[0] if gt_list else "")
            
            # Delay before next question (except for the last one)
            if i < len(test_dataset) - 1:
                print(f"⏳ Waiting {args.delay_minutes} minutes before next question...")
                for remaining in range(args.delay_minutes * 60, 0, -60):
                    mins = remaining // 60
                    print(f"   ⏱️  {mins} minutes remaining...", end='\r')
                    time.sleep(60)
                print()  # New line after countdown
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 Evaluation Complete!")
        print(f"📊 Statistics: ✅ {successful_questions} successful, ❌ {failed_questions} failed")
        print(f"⏱️ Total time: {total_time/3600:.1f} hours")
        
        # Create dataset for RAGAS evaluation
        print("\n🔬 Running RAGAS evaluation...")
        
        # Note: We can't use the full RAGAS evaluation without Streamlit context
        # Save the raw results instead
        final_results = {
            "evaluation_metadata": {
                "total_questions": len(test_dataset),
                "successful_questions": successful_questions,
                "failed_questions": failed_questions,
                "delay_minutes": args.delay_minutes,
                "total_time_hours": total_time / 3600,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "categories": categories
            },
            "raw_results": {
                "questions": questions,
                "contexts": contexts,
                "answers": answers,
                "ground_truths": ground_truths,
                "references": references
            }
        }
        
        # Save final results
        results_file = output_dir / f"large_scale_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Final results saved: {results_file}")
        
        # Create summary report
        summary_file = output_dir / f"evaluation_summary_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"""Large Scale RAG Evaluation Summary
===================================

Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total Questions: {len(test_dataset)}
Successful: {successful_questions}
Failed: {failed_questions}
Success Rate: {(successful_questions/len(test_dataset))*100:.1f}%

Processing Details:
- Delay between questions: {args.delay_minutes} minutes
- Total processing time: {total_time/3600:.1f} hours
- Average time per question: {total_time/len(test_dataset):.1f} seconds

Question Categories:
{chr(10).join([f"- {cat}: {count}" for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)])}

Results saved to: {results_file}

Next Steps:
1. Use the Streamlit interface for full RAGAS metric evaluation
2. Load the results file in the web interface for detailed analysis
3. Generate comprehensive research reports with metric scores
""")
        
        print(f"📋 Summary report saved: {summary_file}")
        print("\n✅ Large-scale evaluation completed successfully!")
        print("\n💡 Next Steps:")
        print("   1. Use 'streamlit run test.py' for full RAGAS metric evaluation")
        print("   2. Load these results in the web interface for detailed analysis")
        print("   3. Generate comprehensive research reports with metric scores")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
