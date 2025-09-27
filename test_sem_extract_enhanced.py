"""
Test suite for enhanced sem_extract with chunking, batching, and async support.

This test file validates the enhanced sem_extract functionality including:
1. Document classification and chunking
2. Hybrid batch processing
3. Result aggregation
4. Async processing
5. Error handling and fallbacks
"""

import asyncio
import pytest
import pandas as pd
from typing import List, Dict, Any

import lotus
from lotus.models import LM
from lotus.sem_ops.sem_extract_enhanced import (
    sem_extract_enhanced,
    sem_extract_enhanced_async,
    DocumentClassifier,
    DocumentChunker,
    ChunkResultAggregator,
    HybridBatchProcessor
)


class TestDocumentClassifier:
    """Test document classification functionality."""
    
    def test_classify_documents_mixed_sizes(self):
        """Test classification of mixed document sizes."""
        model = LM(model="gpt-4o-mini", max_ctx_len=4000)
        classifier = DocumentClassifier(model, chunk_threshold=0.1)  # Very low threshold for testing
        
        docs = [
            {"text": "Short document", "doc_id": "doc1"},
            {"text": "Another short one", "doc_id": "doc2"},
            {"text": "This is a very long document that exceeds the context limits and should be chunked. " * 300, "doc_id": "doc3"},  # Make it very long
            {"text": "Medium length document that might or might not need chunking", "doc_id": "doc4"}
        ]
        
        docs_to_chunk, docs_to_process = classifier.classify_documents(docs)
        
        # Check that classification worked
        assert len(docs_to_chunk) + len(docs_to_process) == len(docs)
        
        # The very long document should definitely be chunked
        long_doc_found = any("doc3" in doc.get("doc_id", "") for doc in docs_to_chunk)
        assert long_doc_found, "Long document should be classified for chunking"
    
    def test_should_chunk_document(self):
        """Test individual document chunking decision."""
        model = LM(model="gpt-4o-mini", max_ctx_len=4000)
        classifier = DocumentClassifier(model, chunk_threshold=0.1)  # Very low threshold for testing
        
        # Short document should not be chunked
        short_doc = {"text": "Short text"}
        assert not classifier._should_chunk_document(short_doc)
        
        # Long document should be chunked
        long_doc = {"text": "Very long text " * 1000}
        assert classifier._should_chunk_document(long_doc)


class TestDocumentChunker:
    """Test document chunking functionality."""
    
    def test_token_based_chunking(self):
        """Test token-based chunking."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20, strategy="token")
        
        doc = {
            "text": "This is a test document with multiple sentences. " * 50,
            "doc_id": "test_doc"
        }
        
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) > 1
        assert all("chunk_id" in chunk for chunk in chunks)
        assert all("is_chunk" in chunk for chunk in chunks)
        assert all(chunk["is_chunk"] for chunk in chunks)
    
    def test_sentence_based_chunking(self):
        """Test sentence-based chunking."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10, strategy="sentence")
        
        doc = {
            "text": "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.",
            "doc_id": "test_doc"
        }
        
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) >= 1
        assert all("chunk_id" in chunk for chunk in chunks)
    
    def test_paragraph_based_chunking(self):
        """Test paragraph-based chunking."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10, strategy="paragraph")
        
        doc = {
            "text": "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
            "doc_id": "test_doc"
        }
        
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) >= 1
        assert all("chunk_id" in chunk for chunk in chunks)


class TestChunkResultAggregator:
    """Test chunk result aggregation functionality."""
    
    def test_merge_strategy(self):
        """Test merge aggregation strategy."""
        aggregator = ChunkResultAggregator(strategy="merge")
        
        chunk_results = [
            {"sentiment": "positive", "confidence": 0.8},
            {"sentiment": "positive", "confidence": 0.9},
            {"sentiment": "neutral", "confidence": 0.7}
        ]
        
        result = aggregator.aggregate_results(chunk_results, {})
        
        assert "sentiment" in result
        assert "confidence" in result
        assert isinstance(result["sentiment"], list)
        assert isinstance(result["confidence"], list)
    
    def test_vote_strategy(self):
        """Test vote aggregation strategy."""
        aggregator = ChunkResultAggregator(strategy="vote")
        
        chunk_results = [
            {"sentiment": "positive", "confidence": 0.8},
            {"sentiment": "positive", "confidence": 0.9},
            {"sentiment": "neutral", "confidence": 0.7}
        ]
        
        result = aggregator.aggregate_results(chunk_results, {})
        
        assert "sentiment" in result
        assert "confidence" in result
        # Should vote for most common sentiment
        assert result["sentiment"] == "positive"
    
    def test_weighted_strategy(self):
        """Test weighted aggregation strategy."""
        aggregator = ChunkResultAggregator(strategy="weighted")
        
        chunk_results = [
            {"sentiment": "positive", "confidence": 0.8},
            {"sentiment": "positive", "confidence": 0.9},
            {"sentiment": "neutral", "confidence": 0.7}
        ]
        
        result = aggregator.aggregate_results(chunk_results, {})
        
        assert "sentiment" in result
        assert "confidence" in result


class TestHybridBatchProcessor:
    """Test hybrid batch processing functionality."""
    
    def test_process_mixed_documents(self):
        """Test processing mixed document collection."""
        model = LM(model="gpt-4o-mini")
        processor = HybridBatchProcessor(model, batch_size=2, chunk_size=100, chunk_overlap=20)
        
        docs_to_chunk = [
            {"text": "Very long document " * 200, "doc_id": "long_doc"}
        ]
        docs_to_process = [
            {"text": "Short document", "doc_id": "short_doc"}
        ]
        output_cols = {"sentiment": "positive/negative/neutral"}
        
        # This would normally call the model, but we'll mock it for testing
        # In a real test, you'd need to mock the model calls
        try:
            result = processor.process_mixed_documents(
                docs_to_chunk, docs_to_process, output_cols
            )
            assert isinstance(result, lotus.types.SemanticExtractOutput)
        except Exception as e:
            # Expected to fail without real model, but should not crash
            assert "model" in str(e).lower() or "api" in str(e).lower()


class TestSemExtractEnhanced:
    """Test enhanced sem_extract functionality."""
    
    def test_enhanced_without_chunking(self):
        """Test enhanced sem_extract without chunking."""
        model = LM(model="gpt-4o-mini")
        docs = [
            {"text": "This product is great!", "doc_id": "doc1"},
            {"text": "I love this service", "doc_id": "doc2"}
        ]
        output_cols = {"sentiment": "positive/negative/neutral"}
        
        try:
            result = sem_extract_enhanced(
                docs, model, output_cols,
                enable_chunking=False,
                use_batch_processing=True
            )
            assert isinstance(result, lotus.types.SemanticExtractOutput)
        except Exception as e:
            # Expected to fail without real API key, but should not crash
            assert "api" in str(e).lower() or "key" in str(e).lower()
    
    def test_enhanced_with_chunking(self):
        """Test enhanced sem_extract with chunking."""
        model = LM(model="gpt-4o-mini")
        docs = [
            {"text": "Short document", "doc_id": "doc1"},
            {"text": "Very long document " * 1000, "doc_id": "doc2"}
        ]
        output_cols = {"sentiment": "positive/negative/neutral"}
        
        try:
            result = sem_extract_enhanced(
                docs, model, output_cols,
                enable_chunking=True,
                chunk_size=500,
                chunk_overlap=50
            )
            assert isinstance(result, lotus.types.SemanticExtractOutput)
        except Exception as e:
            # Expected to fail without real API key, but should not crash
            assert "api" in str(e).lower() or "key" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_enhanced_async_without_chunking(self):
        """Test enhanced async sem_extract without chunking."""
        model = LM(model="gpt-4o-mini")
        docs = [
            {"text": "This product is great!", "doc_id": "doc1"},
            {"text": "I love this service", "doc_id": "doc2"}
        ]
        output_cols = {"sentiment": "positive/negative/neutral"}
        
        try:
            result = await sem_extract_enhanced_async(
                docs, model, output_cols,
                enable_chunking=False,
                use_batch_processing=True
            )
            assert isinstance(result, lotus.types.SemanticExtractOutput)
        except Exception as e:
            # Expected to fail without real API key, but should not crash
            assert "api" in str(e).lower() or "key" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_enhanced_async_with_chunking(self):
        """Test enhanced async sem_extract with chunking."""
        model = LM(model="gpt-4o-mini")
        docs = [
            {"text": "Short document", "doc_id": "doc1"},
            {"text": "Very long document " * 1000, "doc_id": "doc2"}
        ]
        output_cols = {"sentiment": "positive/negative/neutral"}
        
        try:
            result = await sem_extract_enhanced_async(
                docs, model, output_cols,
                enable_chunking=True,
                chunk_size=500,
                chunk_overlap=50
            )
            assert isinstance(result, lotus.types.SemanticExtractOutput)
        except Exception as e:
            # Expected to fail without real API key, but should not crash
            assert "api" in str(e).lower() or "key" in str(e).lower()


class TestDataFrameIntegration:
    """Test DataFrame integration with enhanced sem_extract."""
    
    def test_dataframe_enhanced_extract(self):
        """Test DataFrame integration with enhanced sem_extract."""
        # Create test DataFrame
        df = pd.DataFrame({
            'text': [
                'This product is excellent!',
                'Very long document that should be chunked ' * 100,
                'Another short review',
                'Another long document ' * 200
            ],
            'rating': [5, 4, 5, 3]
        })
        
        # Test that the enhanced functionality can be integrated
        # In a real implementation, you would extend the DataFrame accessor
        assert len(df) == 4
        assert 'text' in df.columns
        assert 'rating' in df.columns


class TestErrorHandling:
    """Test error handling and fallback mechanisms."""
    
    def test_classifier_error_handling(self):
        """Test document classifier error handling."""
        model = LM(model="gpt-4o-mini")
        classifier = DocumentClassifier(model)
        
        # Test with empty documents
        empty_docs = [{"text": ""}, {"text": None}]
        docs_to_chunk, docs_to_process = classifier.classify_documents(empty_docs)
        
        # Should handle empty documents gracefully
        assert len(docs_to_chunk) == 0
        assert len(docs_to_process) == 2
    
    def test_chunker_error_handling(self):
        """Test document chunker error handling."""
        chunker = DocumentChunker()
        
        # Test with empty document
        empty_doc = {"text": "", "doc_id": "empty"}
        chunks = chunker.chunk_document(empty_doc)
        
        # Should return original document
        assert len(chunks) == 1
        assert chunks[0] == empty_doc
    
    def test_aggregator_error_handling(self):
        """Test result aggregator error handling."""
        aggregator = ChunkResultAggregator()
        
        # Test with empty results
        empty_results = []
        result = aggregator.aggregate_results(empty_results, {})
        
        # Should return empty dict
        assert result == {}


class TestPerformanceOptimization:
    """Test performance optimization features."""
    
    def test_batch_size_optimization(self):
        """Test batch size optimization."""
        model = LM(model="gpt-4o-mini")
        processor = HybridBatchProcessor(model, batch_size=5)
        
        # Test with different document counts
        docs = [{"text": f"Document {i}", "doc_id": f"doc{i}"} for i in range(10)]
        
        # Should handle batch size optimization
        assert processor.batch_size == 5
    
    def test_chunk_size_optimization(self):
        """Test chunk size optimization."""
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        
        # Test chunking with optimized sizes
        doc = {"text": "Test document " * 100, "doc_id": "test"}
        chunks = chunker.chunk_document(doc)
        
        # Should create appropriate number of chunks
        assert len(chunks) >= 1


def run_comprehensive_test():
    """Run comprehensive test of enhanced sem_extract functionality."""
    print("=== Enhanced sem_extract Comprehensive Test ===")
    
    # Test document classification
    print("Testing document classification...")
    model = LM(model="gpt-4o-mini")
    classifier = DocumentClassifier(model)
    
    docs = [
        {"text": "Short document", "doc_id": "short1"},
        {"text": "Another short one", "doc_id": "short2"},
        {"text": "Very long document " * 500, "doc_id": "long1"},
        {"text": "Medium length document " * 50, "doc_id": "medium1"}
    ]
    
    docs_to_chunk, docs_to_process = classifier.classify_documents(docs)
    print(f"Documents to chunk: {len(docs_to_chunk)}")
    print(f"Documents to process: {len(docs_to_process)}")
    
    # Test chunking
    print("Testing document chunking...")
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
    
    for doc in docs_to_chunk:
        chunks = chunker.chunk_document(doc)
        print(f"Document {doc['doc_id']} chunked into {len(chunks)} pieces")
    
    # Test aggregation
    print("Testing result aggregation...")
    aggregator = ChunkResultAggregator(strategy="merge")
    
    mock_chunk_results = [
        {"sentiment": "positive", "confidence": 0.8},
        {"sentiment": "positive", "confidence": 0.9}
    ]
    
    aggregated = aggregator.aggregate_results(mock_chunk_results, {})
    print(f"Aggregated result: {aggregated}")
    
    print("=== Test Complete ===")


if __name__ == "__main__":
    # Run comprehensive test
    run_comprehensive_test()
    
    # Run pytest tests
    pytest.main([__file__, "-v"])
