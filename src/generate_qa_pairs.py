#!/usr/bin/env python3
"""
Improved QA Pair Generator - Creates better quality training data
Focus on chunk-level questions rather than sentence extraction
"""

import os
import json
import yaml
import random
from typing import List, Dict, Tuple
import glob
import re

def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load the YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_processed_chunks(processed_dir: str) -> List[Dict]:
    """Load all processed JSON chunks"""
    chunks = []
    json_files = glob.glob(os.path.join(processed_dir, "*.json"))
    json_files = [f for f in json_files if not f.endswith('qa_pairs.jsonl')]
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                chunks.append(chunk_data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    print(f"Loaded {len(chunks)} chunks")
    return chunks

def clean_text_for_qa(text: str) -> str:
    """Clean text while preserving important content"""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip metadata but keep substantive content
        if any(skip in line.lower() for skip in [
            'received', 'accepted', 'date of publication', 'digital object identifier',
            'authorized licensed use', 'downloaded on', 'restrictions apply',
            'ieee xplore', 'doi:'
        ]):
            continue
        
        # Skip very short lines, but keep meaningful ones
        if len(line) < 10:
            continue
            
        cleaned_lines.append(line)
    
    return ' '.join(cleaned_lines)

def extract_key_concepts(text: str, source: str) -> List[str]:
    """Extract key concepts and definitions from the text"""
    concepts = []
    text_lower = text.lower()
    
    # Look for definition patterns
    definition_patterns = [
        r'(.+?)\s+is\s+defined\s+as\s+(.+?)\.', 
        r'(.+?)\s+refers\s+to\s+(.+?)\.', 
        r'(.+?)\s+is\s+a\s+(.+?)\.', 
        r'(.+?)\s+are\s+(.+?)\.', 
        r'there\s+are\s+(.+?)\s+types?\s+of\s+(.+?):', 
        r'(.+?)\s+involves?\s+(.+?)\.', 
        r'(.+?)\s+includes?\s+(.+?)\.', 
    ]
    
    for pattern in definition_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if len(match) == 2 and all(len(part.strip()) > 10 for part in match):
                concepts.append(match)
    
    return concepts

def generate_chunk_based_qa(chunk: Dict) -> List[Tuple[str, str]]:
    """Generate QA pairs that work at the chunk level"""
    text = chunk['text']
    source = chunk['metadata']['source']
    chunk_index = chunk['metadata']['chunk_index']
    
    # Clean the text
    cleaned_text = clean_text_for_qa(text)
    
    if len(cleaned_text) < 100:
        return []
    
    qa_pairs = []
    
    # Strategy 1: Use the full chunk as context for comprehensive questions
    domain_questions = get_domain_questions(source, cleaned_text)
    
    for question in domain_questions:
        # Use the entire cleaned chunk as the answer context
        # This ensures the answer actually exists in the chunk
        answer = create_comprehensive_answer(question, cleaned_text)
        
        if answer and len(answer) > 50:
            qa_pairs.append((question, answer))
    
    # Strategy 2: Extract actual definitions and concepts
    concepts = extract_key_concepts(cleaned_text, source)
    
    for concept_pair in concepts:
        if len(concept_pair) == 2:
            term, definition = concept_pair
            term = term.strip()
            definition = definition.strip()
            
            if len(term) < 100 and len(definition) > 20:
                question = f"What is {term}?"
                answer = f"{term} is {definition}."
                qa_pairs.append((question, answer))
    
    return qa_pairs

def get_domain_questions(source: str, text: str) -> List[str]:
    """Get relevant questions based on document domain and content"""
    questions = []
    text_lower = text.lower()
    
    # Cybersecurity domain
    if 'cybersecurity' in source.lower() or 'cyber' in source.lower():
        if 'tabletop' in text_lower:
            questions.append("What types of cybersecurity exercises are mentioned?")
        if 'exercise' in text_lower and 'scenario' in text_lower:
            questions.append("How are cybersecurity exercise scenarios created?")
        if 'llm' in text_lower or 'language model' in text_lower:
            questions.append("How are language models used in cybersecurity?")
        if 'training' in text_lower:
            questions.append("What cybersecurity training approaches are discussed?")
    
    # RAG domain
    elif 'rag' in text_lower or 'retrieval' in text_lower:
        if 'similarity' in text_lower and 'threshold' in text_lower:
            questions.append("How do similarity thresholds work in this context?")
        if 'performance' in text_lower or 'evaluation' in text_lower:
            questions.append("How is performance evaluated in this system?")
        if 'retrieval' in text_lower and 'generation' in text_lower:
            questions.append("What is the retrieval-augmented generation approach described?")
    
    # Cloud security domain
    elif 'cloud' in source.lower() and 'security' in source.lower():
        if 'secnavigator' in text_lower:
            questions.append("What is Cloud SecNavigator and how does it work?")
        if 'ragas' in text_lower:
            questions.append("How is RAGAS used for assessment?")
        if 'security practice' in text_lower:
            questions.append("What cloud security practices are discussed?")
    
    # Audio/Language domain
    elif 'audio' in source.lower() or 'language' in source.lower():
        if 'low-resource' in text_lower:
            questions.append("How does this approach help low-resource languages?")
        if 'enhancement' in text_lower:
            questions.append("What enhancements are provided by this system?")
    
    # Generic questions that work for any domain
    if 'method' in text_lower or 'approach' in text_lower:
        questions.append("What methodology is described in this section?")
    if 'result' in text_lower or 'finding' in text_lower:
        questions.append("What are the key findings discussed?")
    if 'challenge' in text_lower or 'limitation' in text_lower:
        questions.append("What challenges or limitations are mentioned?")
    
    return questions

def create_comprehensive_answer(question: str, text: str) -> str:
    """Create an answer by finding the most relevant part of the text"""
    question_words = set(question.lower().split())
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    
    # Score sentences based on question word overlap
    scored_sentences = []
    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        overlap = len(question_words.intersection(sentence_words))
        if overlap > 0:
            scored_sentences.append((sentence, overlap))
    
    # Sort by relevance and take top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    if scored_sentences:
        # Take top 2-3 most relevant sentences
        selected_sentences = [s[0] for s in scored_sentences[:3]]
        answer = '. '.join(selected_sentences)
        
        # Clean and format the answer
        answer = answer.strip()
        if not answer.endswith('.'):
            answer += '.'
        
        return answer
    
    return ""

def generate_all_qa_pairs(chunks: List[Dict]) -> List[Dict]:
    """Generate high-quality QA pairs from all chunks"""
    all_qa_pairs = []
    
    for i, chunk in enumerate(chunks):
        qa_pairs = generate_chunk_based_qa(chunk)
        
        for question, answer in qa_pairs:
            qa_pair = {
                "prompt": question,
                "response": answer,
                "source": chunk['metadata']['source'],
                "chunk_index": chunk['metadata']['chunk_index']
            }
            all_qa_pairs.append(qa_pair)
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(chunks)} chunks...")
    
    # Remove duplicates
    seen = set()
    unique_qa_pairs = []
    for qa in all_qa_pairs:
        key = (qa['prompt'], qa['response'][:100])
        if key not in seen:
            seen.add(key)
            unique_qa_pairs.append(qa)
    
    print(f"Generated {len(unique_qa_pairs)} unique QA pairs")
    return unique_qa_pairs

def save_qa_pairs(qa_pairs: List[Dict], output_path: str):
    """Save QA pairs to JSONL format"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(qa_pairs)} QA pairs to {output_path}")

def main():
    """Main function"""
    print("Improved QA Pair Generator")
    print("="*60)
    print("Creating chunk-aligned training data for better retrieval...")
    
    # Load configuration
    cfg = load_config()
    
    # Load processed chunks
    processed_dir = cfg['paths']['data_processed']
    chunks = load_processed_chunks(processed_dir)
    
    if not chunks:
        print("No chunks found! Run preprocess.py first.")
        return
    
    # Generate QA pairs
    print("\nGenerating improved QA pairs...")
    qa_pairs = generate_all_qa_pairs(chunks)
    
    if not qa_pairs:
        print("No QA pairs generated!")
        return
    
    # Save QA pairs
    output_path = os.path.join(processed_dir, 'qa_pairs.jsonl')
    save_qa_pairs(qa_pairs, output_path)
    
    # Show statistics
    print(f"\nStatistics:")
    print(f"- Total QA pairs: {len(qa_pairs)}")
    
    sources = {}
    for qa in qa_pairs:
        source = qa['source']
        sources[source] = sources.get(source, 0) + 1
    
    print(f"- Sources covered: {len(sources)}")
    for source, count in sorted(sources.items()):
        short_name = source.replace('.pdf', '').replace('_', ' ')[:50]
        print(f"  - {short_name}: {count} pairs")
    
    # Show sample QA pairs
    print(f"\nSample QA pairs:")
    for i, qa in enumerate(qa_pairs[:3]):
        print(f"\nQ{i+1}: {qa['prompt']}")
        print(f"A{i+1}: {qa['response'][:200]}...")
        print(f"Source: {qa['source']} (Chunk {qa['chunk_index']})")
    
    print(f"\n🎯 Improved QA pairs ready for training!")
    print(f"✅ These answers should actually exist in the retrieved chunks")

if __name__ == "__main__":
    main()