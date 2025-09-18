#!/usr/bin/env python3
"""
Real model performance test for batch filter processing.

This script uses real Lotus models to test batch processing functionality
and measure performance improvements with large-scale test data.
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import statistics

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

import lotus
from lotus.models import LM
from lotus.sem_ops.sem_filter import sem_filter
from lotus.types import ReasoningStrategy


def generate_test_data(num_docs: int = 100) -> List[Dict[str, Any]]:
    """Generate large-scale test data for performance testing."""
    print(f"Generating {num_docs} test documents...")
    
    # Sample texts with different sentiment patterns
    positive_texts = [
        "This product is absolutely amazing and exceeded my expectations!",
        "I love this service, it's fantastic and works perfectly.",
        "Excellent quality, highly recommend to everyone!",
        "Outstanding performance, couldn't be happier with the results.",
        "Brilliant design and functionality, worth every penny!",
        "Perfect solution for my needs, very satisfied!",
        "Great value for money, excellent customer service!",
        "Superb quality and fast delivery, will buy again!",
        "Wonderful experience, top-notch product!",
        "Exceptional service, highly impressed!",
    ]
    
    negative_texts = [
        "Terrible product, complete waste of money and time.",
        "Poor quality, disappointed with this purchase.",
        "Awful service, would not recommend to anyone.",
        "Bad experience, product broke after one day.",
        "Horrible customer support, very frustrating.",
        "Worst purchase ever, regret buying this item.",
        "Disappointing quality, not worth the price.",
        "Faulty product, customer service was unhelpful.",
        "Low quality materials, fell apart quickly.",
        "Unsatisfactory experience, would avoid this brand.",
    ]
    
    neutral_texts = [
        "The product arrived on time and works as described.",
        "Standard quality, nothing special but functional.",
        "Average performance, meets basic requirements.",
        "Product is okay, neither great nor terrible.",
        "Decent quality for the price point offered.",
        "Functional design, serves its intended purpose.",
        "Regular product with typical performance.",
        "Standard features, works adequately for needs.",
        "Basic functionality, no major issues encountered.",
        "Ordinary product with expected performance.",
    ]
    
    # Generate documents with mixed sentiment
    docs = []
    for i in range(num_docs):
        if i % 3 == 0:
            text = positive_texts[i % len(positive_texts)]
        elif i % 3 == 1:
            text = negative_texts[i % len(negative_texts)]
        else:
            text = neutral_texts[i % len(neutral_texts)]
        
        docs.append({
            "text": f"Document {i+1}: {text}",
            "id": i+1,
            "category": "review"
        })
    
    print(f"Generated {len(docs)} test documents")
    return docs


def test_individual_processing(
    docs: List[Dict[str, Any]], 
    model: LM, 
    user_instruction: str
) -> Dict[str, Any]:
    """Test individual document processing."""
    print("Testing individual processing...")
    
    start_time = time.time()
    
    result = sem_filter(
        docs=docs,
        model=model,
        user_instruction=user_instruction,
        use_batch_processing=False,
        show_progress_bar=True,
        progress_bar_desc="Individual Processing"
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return {
        "method": "individual",
        "processing_time": processing_time,
        "num_docs": len(docs),
        "outputs": result.outputs,
        "raw_outputs": result.raw_outputs,
        "explanations": result.explanations,
        "model_stats": {
            "virtual_cost": model.stats.virtual_usage.total_cost,
            "virtual_tokens": model.stats.virtual_usage.total_tokens,
            "physical_cost": model.stats.physical_usage.total_cost,
            "physical_tokens": model.stats.physical_usage.total_tokens,
            "cache_hits": model.stats.cache_hits,
        }
    }


def test_batch_processing(
    docs: List[Dict[str, Any]], 
    model: LM, 
    user_instruction: str,
    batch_size: int = 10
) -> Dict[str, Any]:
    """Test batch document processing."""
    print(f"Testing batch processing with batch_size={batch_size}...")
    
    # Reset model stats
    model.reset_stats()
    
    start_time = time.time()
    
    result = sem_filter(
        docs=docs,
        model=model,
        user_instruction=user_instruction,
        use_batch_processing=True,
        batch_size=batch_size,
        show_progress_bar=True,
        progress_bar_desc=f"Batch Processing (size={batch_size})"
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return {
        "method": "batch",
        "batch_size": batch_size,
        "processing_time": processing_time,
        "num_docs": len(docs),
        "outputs": result.outputs,
        "raw_outputs": result.raw_outputs,
        "explanations": result.explanations,
        "model_stats": {
            "virtual_cost": model.stats.virtual_usage.total_cost,
            "virtual_tokens": model.stats.virtual_usage.total_tokens,
            "physical_cost": model.stats.physical_usage.total_cost,
            "physical_tokens": model.stats.physical_usage.total_tokens,
            "cache_hits": model.stats.cache_hits,
        }
    }


def compare_results(individual_result: Dict[str, Any], batch_result: Dict[str, Any]) -> Dict[str, Any]:
    """Compare individual vs batch processing results."""
    print("Comparing results...")
    
    # Check if outputs match
    outputs_match = individual_result["outputs"] == batch_result["outputs"]
    
    # Calculate performance improvements
    time_improvement = (individual_result["processing_time"] - batch_result["processing_time"]) / individual_result["processing_time"] * 100
    cost_improvement = (individual_result["model_stats"]["virtual_cost"] - batch_result["model_stats"]["virtual_cost"]) / individual_result["model_stats"]["virtual_cost"] * 100 if individual_result["model_stats"]["virtual_cost"] > 0 else 0
    token_improvement = (individual_result["model_stats"]["virtual_tokens"] - batch_result["model_stats"]["virtual_tokens"]) / individual_result["model_stats"]["virtual_tokens"] * 100 if individual_result["model_stats"]["virtual_tokens"] > 0 else 0
    
    return {
        "outputs_match": outputs_match,
        "time_improvement_percent": time_improvement,
        "cost_improvement_percent": cost_improvement,
        "token_improvement_percent": token_improvement,
        "individual_time": individual_result["processing_time"],
        "batch_time": batch_result["processing_time"],
        "individual_cost": individual_result["model_stats"]["virtual_cost"],
        "batch_cost": batch_result["model_stats"]["virtual_cost"],
        "individual_tokens": individual_result["model_stats"]["virtual_tokens"],
        "batch_tokens": batch_result["model_stats"]["virtual_tokens"],
    }


def test_different_batch_sizes(
    docs: List[Dict[str, Any]], 
    model: LM, 
    user_instruction: str,
    batch_sizes: List[int] = [5, 10, 20, 50]
) -> Dict[str, Any]:
    """Test different batch sizes to find optimal performance."""
    print("Testing different batch sizes...")
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Reset model stats
        model.reset_stats()
        
        start_time = time.time()
        
        result = sem_filter(
            docs=docs,
            model=model,
            user_instruction=user_instruction,
            use_batch_processing=True,
            batch_size=batch_size,
            show_progress_bar=True,
            progress_bar_desc=f"Batch Size {batch_size}"
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        results[f"batch_{batch_size}"] = {
            "batch_size": batch_size,
            "processing_time": processing_time,
            "cost": model.stats.virtual_usage.total_cost,
            "tokens": model.stats.virtual_usage.total_tokens,
            "outputs": result.outputs,
        }
    
    return results


def test_with_examples(
    docs: List[Dict[str, Any]], 
    model: LM, 
    user_instruction: str
) -> Dict[str, Any]:
    """Test batch processing with few-shot examples."""
    print("Testing batch processing with examples...")
    
    # Create examples
    examples_multimodal_data = [
        {"text": "This product is amazing and I love it!"},
        {"text": "Terrible quality, very disappointed."},
        {"text": "Average product, nothing special."},
    ]
    examples_answers = [True, False, False]  # positive, negative, neutral
    
    # Reset model stats
    model.reset_stats()
    
    start_time = time.time()
    
    result = sem_filter(
        docs=docs,
        model=model,
        user_instruction=user_instruction,
        examples_multimodal_data=examples_multimodal_data,
        examples_answers=examples_answers,
        use_batch_processing=True,
        batch_size=10,
        show_progress_bar=True,
        progress_bar_desc="Batch with Examples"
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return {
        "method": "batch_with_examples",
        "processing_time": processing_time,
        "num_docs": len(docs),
        "outputs": result.outputs,
        "explanations": result.explanations,
        "model_stats": {
            "virtual_cost": model.stats.virtual_usage.total_cost,
            "virtual_tokens": model.stats.virtual_usage.total_tokens,
            "physical_cost": model.stats.physical_usage.total_cost,
            "physical_tokens": model.stats.physical_usage.total_tokens,
            "cache_hits": model.stats.cache_hits,
        }
    }


def test_with_cot_reasoning(
    docs: List[Dict[str, Any]], 
    model: LM, 
    user_instruction: str
) -> Dict[str, Any]:
    """Test batch processing with chain-of-thought reasoning."""
    print("Testing batch processing with CoT reasoning...")
    
    # Reset model stats
    model.reset_stats()
    
    start_time = time.time()
    
    result = sem_filter(
        docs=docs,
        model=model,
        user_instruction=user_instruction,
        strategy=ReasoningStrategy.ZS_COT,
        use_batch_processing=True,
        batch_size=10,
        show_progress_bar=True,
        progress_bar_desc="Batch with CoT"
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return {
        "method": "batch_with_cot",
        "processing_time": processing_time,
        "num_docs": len(docs),
        "outputs": result.outputs,
        "explanations": result.explanations,
        "model_stats": {
            "virtual_cost": model.stats.virtual_usage.total_cost,
            "virtual_tokens": model.stats.virtual_usage.total_tokens,
            "physical_cost": model.stats.physical_usage.total_cost,
            "physical_tokens": model.stats.physical_usage.total_tokens,
            "cache_hits": model.stats.cache_hits,
        }
    }


def print_results_summary(results: Dict[str, Any]) -> None:
    """Print a summary of test results."""
    print("\n" + "="*80)
    print("PERFORMANCE TEST RESULTS SUMMARY")
    print("="*80)
    
    if "comparison" in results:
        comp = results["comparison"]
        print(f"\n📊 INDIVIDUAL vs BATCH COMPARISON:")
        print(f"   Outputs Match: {'✅ Yes' if comp['outputs_match'] else '❌ No'}")
        print(f"   Time Improvement: {comp['time_improvement_percent']:.1f}%")
        print(f"   Cost Improvement: {comp['cost_improvement_percent']:.1f}%")
        print(f"   Token Improvement: {comp['token_improvement_percent']:.1f}%")
        print(f"   Individual Time: {comp['individual_time']:.2f}s")
        print(f"   Batch Time: {comp['batch_time']:.2f}s")
        print(f"   Individual Cost: ${comp['individual_cost']:.6f}")
        print(f"   Batch Cost: ${comp['batch_cost']:.6f}")
    
    if "batch_sizes" in results:
        print(f"\n📈 BATCH SIZE OPTIMIZATION:")
        for batch_key, batch_result in results["batch_sizes"].items():
            print(f"   Batch Size {batch_result['batch_size']:2d}: {batch_result['processing_time']:.2f}s, ${batch_result['cost']:.6f}, {batch_result['tokens']:,} tokens")
    
    if "examples" in results:
        ex = results["examples"]
        print(f"\n🎯 BATCH WITH EXAMPLES:")
        print(f"   Processing Time: {ex['processing_time']:.2f}s")
        print(f"   Cost: ${ex['model_stats']['virtual_cost']:.6f}")
        print(f"   Tokens: {ex['model_stats']['virtual_tokens']:,}")
    
    if "cot" in results:
        cot = results["cot"]
        print(f"\n🧠 BATCH WITH COT REASONING:")
        print(f"   Processing Time: {cot['processing_time']:.2f}s")
        print(f"   Cost: ${cot['model_stats']['virtual_cost']:.6f}")
        print(f"   Tokens: {cot['model_stats']['virtual_tokens']:,}")
    
    print("\n" + "="*80)


def save_results(results: Dict[str, Any], filename: str = "batch_performance_results.json") -> None:
    """Save test results to JSON file."""
    # Convert non-serializable objects to strings
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {}
            for sub_key, sub_value in value.items():
                if hasattr(sub_value, '__dict__'):
                    serializable_results[key][sub_key] = str(sub_value)
                else:
                    serializable_results[key][sub_key] = sub_value
        else:
            serializable_results[key] = value
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filename}")


def main():
    """Run comprehensive batch processing performance tests."""
    print("🚀 Starting Real Model Batch Processing Performance Tests")
    print("="*80)
    
    # Configure model
    model = LM(
        model="openrouter/google/gemini-2.5-flash",
        max_batch_size=4,
        temperature=0.0,
        max_tokens=256,
        api_key="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",
        base_url="https://openrouter.ai/api/v1"
    )
    lotus.settings.configure(
        lm=model,
        enable_cache=False  # Disable cache for accurate performance measurement
    )
    
    # Test parameters
    num_docs = 50  # Start with 50 documents for initial testing
    user_instruction = "Is this a positive sentiment? Answer True for positive, False for negative or neutral."
    
    print(f"Model: {model.model}")
    print(f"Number of documents: {num_docs}")
    print(f"User instruction: {user_instruction}")
    
    # Generate test data
    docs = generate_test_data(num_docs)
    
    results = {}
    
    try:
        # Test 1: Individual vs Batch processing comparison
        print(f"\n{'='*50}")
        print("TEST 1: Individual vs Batch Processing Comparison")
        print(f"{'='*50}")
        
        individual_result = test_individual_processing(docs, model, user_instruction)
        batch_result = test_batch_processing(docs, model, user_instruction, batch_size=10)
        
        comparison = compare_results(individual_result, batch_result)
        results["individual"] = individual_result
        results["batch"] = batch_result
        results["comparison"] = comparison
        
        # Test 2: Different batch sizes
        print(f"\n{'='*50}")
        print("TEST 2: Different Batch Sizes")
        print(f"{'='*50}")
        
        batch_sizes = [5, 10, 20]  # Smaller sizes for faster testing
        batch_size_results = test_different_batch_sizes(docs, model, user_instruction, batch_sizes)
        results["batch_sizes"] = batch_size_results
        
        # Test 3: Batch processing with examples
        print(f"\n{'='*50}")
        print("TEST 3: Batch Processing with Examples")
        print(f"{'='*50}")
        
        examples_result = test_with_examples(docs, model, user_instruction)
        results["examples"] = examples_result
        
        # Test 4: Batch processing with CoT reasoning
        print(f"\n{'='*50}")
        print("TEST 4: Batch Processing with CoT Reasoning")
        print(f"{'='*50}")
        
        cot_result = test_with_cot_reasoning(docs, model, user_instruction)
        results["cot"] = cot_result
        
        # Print summary
        print_results_summary(results)
        
        # Save results
        save_results(results)
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
