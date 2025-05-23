#!/usr/bin/env python3
"""
Updated evaluation script for your rebuilt FAISS index
"""

import os
import yaml
import json
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_config():
    with open(os.path.join('config', 'config.yaml'), 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_qa_pairs(qa_path):
    pairs = []
    with open(qa_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pairs.append(record)
    return pairs

def main():
    cfg = load_config()
    
    # Paths
    faiss_path = cfg['paths']['faiss_index'] + "_dir"  # Use rebuilt directory
    processed_dir = cfg['paths']['data_processed']
    qa_path = os.path.join(processed_dir, 'qa_pairs.jsonl')
    model_name = cfg['retrieval']['embedding_model']
    top_k = cfg['retrieval']['top_k']

    print(f'Loading FAISS from: {faiss_path}')
    
    # Load embeddings and vector store
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.load_local(
        faiss_path,
        embedder,
        allow_dangerous_deserialization=True
    )

    # Load QA pairs
    qa_pairs = load_qa_pairs(qa_path)
    print(f'Evaluating {len(qa_pairs)} QA pairs with top-{top_k} retrieval...')

    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    # Evaluate
    correct_counts = []
    detailed_results = []
    
    for i, qa in enumerate(qa_pairs):
        # Retrieve documents
        docs = retriever.invoke(qa['prompt'])
        
        # Check if the expected response text appears in any retrieved chunk
        hit = False
        retrieved_sources = []
        
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            chunk_idx = doc.metadata.get('chunk_index', 0)
            retrieved_sources.append(f"{source}_chunk{chunk_idx:03d}")
            
            # Check if response text appears in this chunk
            if qa['response'].strip().lower() in doc.page_content.lower():
                hit = True
        
        correct_counts.append(1 if hit else 0)
        
        # Store detailed results for analysis
        detailed_results.append({
            'question': qa['prompt'],
            'expected_source': qa['source'],
            'expected_chunk': qa['chunk_index'],
            'retrieved_sources': retrieved_sources,
            'hit': hit
        })
        
        # Show progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(qa_pairs)} questions...")

    # Calculate metrics
    precision_at_k = np.mean(correct_counts)
    
    print(f'\n{"="*50}')
    print(f'EVALUATION RESULTS')
    print(f'{"="*50}')
    print(f'Total QA pairs: {len(qa_pairs)}')
    print(f'Precision@{top_k}: {precision_at_k:.4f}')
    print(f'Hits: {sum(correct_counts)}/{len(qa_pairs)}')
    
    # Show some failures for analysis
    failures = [r for r in detailed_results if not r['hit']]
    if failures:
        print(f'\nSample failures (showing first 3):')
        for i, failure in enumerate(failures[:3]):
            print(f'\nFailure {i+1}:')
            print(f"  Question: {failure['question'][:80]}...")
            print(f"  Expected: {failure['expected_source']} chunk {failure['expected_chunk']}")
            print(f"  Retrieved: {failure['retrieved_sources']}")
    
    # Source analysis
    print(f'\nSource distribution in retrieved results:')
    all_sources = []
    for result in detailed_results:
        for source in result['retrieved_sources']:
            all_sources.append(source.split('_chunk')[0])
    
    from collections import Counter
    source_counts = Counter(all_sources)
    for source, count in source_counts.most_common(5):
        short_name = source.replace('.pdf', '')[:50]
        print(f"  {short_name}: {count} retrievals")

if __name__ == '__main__':
    main()