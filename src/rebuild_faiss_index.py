#!/usr/bin/env python3
"""
Rebuild FAISS index from processed JSON chunks with proper text content
Save this file as: rebuild_faiss_index.py in your project root directory
"""

import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
import glob

# LangChain imports
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("Using langchain-huggingface package")
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("Using langchain-community embeddings")

from langchain_community.vectorstores import FAISS
from langchain.schema import Document


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load the YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_processed_chunks(processed_dir: str) -> List[Dict[str, Any]]:
    """Load all processed JSON chunks from the directory"""
    chunks = []
    json_files = glob.glob(os.path.join(processed_dir, "*.json"))
    
    # Filter out qa_pairs.jsonl
    json_files = [f for f in json_files if not f.endswith('qa_pairs.jsonl')]
    
    print(f"Found {len(json_files)} JSON chunk files")
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                
                # Extract filename and chunk number from filename
                filename = os.path.basename(json_file)
                if '_chunk' in filename:
                    base_name = filename.split('_chunk')[0]
                    chunk_num = filename.split('_chunk')[1].split('.')[0]
                else:
                    base_name = filename.replace('.json', '')
                    chunk_num = '001'
                
                # Add metadata
                chunk_data['source_file'] = json_file
                chunk_data['base_document'] = base_name + '.pdf'
                chunk_data['chunk_index'] = int(chunk_num)
                
                chunks.append(chunk_data)
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(chunks)} chunks")
    return chunks


def inspect_chunk_structure(chunks: List[Dict]) -> None:
    """Inspect the structure of the first few chunks to understand the format"""
    if not chunks:
        print("No chunks to inspect")
        return
    
    print("\n" + "="*50)
    print("CHUNK STRUCTURE ANALYSIS")
    print("="*50)
    
    # Analyze first chunk
    first_chunk = chunks[0]
    print(f"First chunk keys: {list(first_chunk.keys())}")
    print(f"First chunk structure:")
    
    for key, value in first_chunk.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {type(value).__name__} (length: {len(value)}) - '{value[:100]}...'")
        else:
            print(f"  {key}: {type(value).__name__} - {value}")
    
    # Look for content fields
    content_fields = []
    for key in first_chunk.keys():
        if any(word in key.lower() for word in ['content', 'text', 'body', 'chunk']):
            content_fields.append(key)
    
    print(f"\nPotential content fields: {content_fields}")
    
    # Check a few more chunks to see if structure is consistent
    if len(chunks) > 1:
        print(f"\nChecking consistency across {min(3, len(chunks))} chunks...")
        all_keys = set(first_chunk.keys())
        for i in range(1, min(3, len(chunks))):
            chunk_keys = set(chunks[i].keys())
            all_keys = all_keys.intersection(chunk_keys)
        print(f"Common keys across chunks: {list(all_keys)}")


def create_documents_from_chunks(chunks: List[Dict]) -> List[Document]:
    """Convert chunks to LangChain Documents"""
    documents = []
    
    for i, chunk in enumerate(chunks):
        # Try to find the text content in various possible fields
        page_content = ""
        
        # Common field names for content
        content_candidates = ['content', 'text', 'page_content', 'chunk_text', 'body']
        
        for field in content_candidates:
            if field in chunk and chunk[field]:
                page_content = chunk[field]
                break
        
        # If no direct content field, look for fields containing 'content' or 'text'
        if not page_content:
            for key, value in chunk.items():
                if (any(word in key.lower() for word in ['content', 'text']) 
                    and isinstance(value, str) and len(value) > 50):
                    page_content = value
                    break
        
        # If still no content, use the longest string field
        if not page_content:
            longest_field = ""
            longest_length = 0
            for key, value in chunk.items():
                if isinstance(value, str) and len(value) > longest_length:
                    longest_length = len(value)
                    longest_field = value
            page_content = longest_field
        
        # Fallback if still no content
        if not page_content:
            page_content = f"No content found for chunk {i}"
            print(f"Warning: No content found for chunk {i} from {chunk.get('source_file', 'unknown')}")
        
        # Create metadata (everything except the content)
        metadata = {}
        for key, value in chunk.items():
            if key not in content_candidates and not (isinstance(value, str) and value == page_content):
                metadata[key] = value
        
        # Ensure required metadata fields
        if 'source' not in metadata:
            metadata['source'] = chunk.get('base_document', f'unknown_document_{i}')
        if 'chunk_index' not in metadata:
            metadata['chunk_index'] = chunk.get('chunk_index', i + 1)
        
        # Create document
        doc = Document(
            page_content=page_content,
            metadata=metadata
        )
        documents.append(doc)
        
        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(chunks)} chunks...")
    
    return documents


def rebuild_faiss_index(cfg: dict, documents: List[Document]) -> None:
    """Rebuild the FAISS index with proper content"""
    
    # Setup embeddings
    embed_model_name = (
        cfg.get('retrieval', {}).get('embedding_model') or
        'sentence-transformers/all-MiniLM-L6-v2'
    )
    print(f"Using embedding model: {embed_model_name}")
    embedder = HuggingFaceEmbeddings(model_name=embed_model_name)
    
    # Create FAISS index
    print("Creating FAISS index from documents...")
    print(f"Processing {len(documents)} documents...")
    
    # Create index in batches to manage memory
    batch_size = 50
    vector_store = None
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        
        if vector_store is None:
            # Create initial index
            vector_store = FAISS.from_documents(batch, embedder)
        else:
            # Add to existing index
            batch_store = FAISS.from_documents(batch, embedder)
            vector_store.merge_from(batch_store)
    
    # Save the index
    faiss_path = cfg['paths']['faiss_index']
    
    # Create directory if it doesn't exist
    faiss_dir = os.path.dirname(faiss_path)
    if faiss_dir:
        os.makedirs(faiss_dir, exist_ok=True)
    
    # Save as directory (recommended approach)
    if os.path.isfile(faiss_path):
        # Backup old index
        backup_path = faiss_path + ".backup"
        print(f"Backing up old index to {backup_path}")
        os.rename(faiss_path, backup_path)
        
        # Convert to directory path
        faiss_path = faiss_path + "_dir"
    
    print(f"Saving FAISS index to: {faiss_path}")
    vector_store.save_local(faiss_path)
    
    # Also save metadata separately for inspection
    metadata_path = os.path.join(os.path.dirname(faiss_path), "metadata_with_content.json")
    metadata_list = []
    for doc in documents:
        metadata_list.append({
            "page_content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "full_content_length": len(doc.page_content),
            "metadata": doc.metadata
        })
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    
    print(f"Saved detailed metadata to: {metadata_path}")
    
    return vector_store


def test_rebuilt_index(vector_store: FAISS, test_query: str = "What is retrieval-augmented generation?") -> None:
    """Test the rebuilt index"""
    print("\n" + "="*50)
    print("TESTING REBUILT INDEX")
    print("="*50)
    
    print(f"Test query: {test_query}")
    
    # Test retrieval
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(test_query)
    
    print(f"\nRetrieved {len(docs)} documents:")
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Chunk: {doc.metadata.get('chunk_index', 'Unknown')}")
        print(f"Content length: {len(doc.page_content)}")
        print(f"Content preview: {doc.page_content[:300]}...")


def main():
    """Main function to rebuild the FAISS index"""
    print("FAISS Index Rebuilder")
    print("="*50)
    
    # Load configuration
    cfg = load_config()
    
    # Load processed chunks
    processed_dir = cfg['paths']['data_processed']
    chunks = load_processed_chunks(processed_dir)
    
    if not chunks:
        print("No chunks found! Check your processed data directory.")
        return
    
    # Inspect chunk structure
    inspect_chunk_structure(chunks)
    
    # Convert to documents
    print("\nConverting chunks to documents...")
    documents = create_documents_from_chunks(chunks)
    
    if not documents:
        print("No documents created! Check chunk processing.")
        return
    
    print(f"Created {len(documents)} documents")
    
    # Show sample document
    print("\nSample document:")
    sample_doc = documents[0]
    print(f"Content length: {len(sample_doc.page_content)}")
    print(f"Content preview: {sample_doc.page_content[:200]}...")
    print(f"Metadata: {sample_doc.metadata}")
    
    # Rebuild index
    print("\nRebuilding FAISS index...")
    vector_store = rebuild_faiss_index(cfg, documents)
    
    # Test the rebuilt index
    test_rebuilt_index(vector_store)
    
    print("\n" + "="*50)
    print("INDEX REBUILD COMPLETED!")
    print("="*50)
    print("Next steps:")
    print("1. Update your config.yaml to point to the new index directory")
    print("2. Run your pipeline.py to test with real content")


if __name__ == "__main__":
    main()