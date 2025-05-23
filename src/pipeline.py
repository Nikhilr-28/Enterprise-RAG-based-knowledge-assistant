# src/pipeline.py
# LangChain RAG pipeline connecting FAISS retriever with fine-tuned LLMs

import os
import yaml
import torch
import json
import faiss as _faiss
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline as hf_pipeline
)
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Modern imports for updated chains
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """Load the YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_vector_store(cfg: dict, embedder):
    """Load FAISS vector store - supports both directory and file formats"""
    faiss_path = cfg['paths']['faiss_index']
    
    # Check for rebuilt directory first
    if os.path.exists(faiss_path + "_dir"):
        faiss_path = faiss_path + "_dir"
        print(f"Loading rebuilt FAISS directory: {faiss_path}")
        vector_store = FAISS.load_local(
            faiss_path,
            embedder,
            allow_dangerous_deserialization=True
        )
        print(f"DEBUG_FAISS: type=rebuilt_dir, path={faiss_path}")
    
    # Check for regular directory
    elif os.path.isdir(faiss_path):
        print(f"Loading FAISS directory: {faiss_path}")
        vector_store = FAISS.load_local(
            faiss_path,
            embedder,
            allow_dangerous_deserialization=True
        )
        print(f"DEBUG_FAISS: type=directory, path={faiss_path}")
    
    # Check for single file (legacy format)
    elif os.path.isfile(faiss_path):
        print(f"Loading FAISS index file: {faiss_path}")
        index = _faiss.read_index(faiss_path)  # Fixed syntax
        
        # Load metadata JSON
        meta_json = cfg['paths'].get('faiss_metadata_json', 'models/metadata.json')
        print(f"Loading FAISS metadata from JSON: {meta_json}")
        
        if not os.path.exists(meta_json):
            raise FileNotFoundError(f"Metadata file not found: {meta_json}")
        
        with open(meta_json, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
        
        print(f"DEBUG_FAISS: type=legacy_file, index_dims={index.d}, metadata_count={len(metadata_list)}")
        
        # Build docstore mapping
        docstore = {
            str(i): Document(
                page_content=md.get('page_content', ''),
                metadata=md.get('metadata', {})
            ) for i, md in enumerate(metadata_list)
        }
        index_to_docstore_id = {i: str(i) for i in range(len(metadata_list))}
        
        vector_store = FAISS(
            embedder,
            index,
            docstore,
            index_to_docstore_id
        )
    
    else:
        raise FileNotFoundError(
            f"FAISS index not found at {faiss_path}. "
            "Please run rebuild_faiss_index.py first."
        )
    
    return vector_store


def setup_llm(cfg: dict):
    """Setup the language model with fallback to base model"""
    flan_ckpt = cfg['paths']['flan_t5_ckpt']
    
    # Try to load fine-tuned model first
    if os.path.exists(os.path.join(flan_ckpt, 'flan_t5_final')):
        model_path = os.path.join(flan_ckpt, 'flan_t5_final')
        print(f"Loading fine-tuned FLAN-T5: {model_path}")
        try:
            flan_tokenizer = AutoTokenizer.from_pretrained(model_path)
            flan_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            print("✅ Fine-tuned model loaded successfully!")
            print(f"DEBUG_MODEL: type=fine-tuned, path={model_path}")
        except Exception as e:
            print(f"⚠️ Error loading fine-tuned model: {e}")
            print("Falling back to base model...")
            model_path = "google/flan-t5-base"
            flan_tokenizer = AutoTokenizer.from_pretrained(model_path)
            flan_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            print(f"DEBUG_MODEL: type=base_fallback, error={str(e)[:50]}")
    
    # Fallback to base model if fine-tuned doesn't exist
    else:
        print("Fine-tuned model not found, using base FLAN-T5...")
        model_path = "google/flan-t5-base"
        flan_tokenizer = AutoTokenizer.from_pretrained(model_path)
        flan_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        print(f"DEBUG_MODEL: type=base, path={model_path}")
    
    # Setup pipeline
    device = 0 if torch.cuda.is_available() else -1
    print(f"Device: {'CUDA' if device == 0 else 'CPU'}")
    
    flan_pipe = hf_pipeline(
        'text2text-generation',
        model=flan_model,
        tokenizer=flan_tokenizer,
        device=device,
        max_length=256,
        temperature=0.3,
        do_sample=True,
        top_p=0.9
    )
    
    return HuggingFacePipeline(pipeline=flan_pipe)


def create_rag_chain(vector_store, llm, cfg: dict):
    """Create the RAG chain with context length limiting"""
    # Create retriever
    top_k = cfg.get('retrieval', {}).get('top_k', 3)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    
    # Create prompt template
    template = """Use the context below to answer the question. Be specific and cite relevant information.

Context: {context}

Question: {input}

Answer:"""
    
    prompt = PromptTemplate(
        input_variables=['context', 'input'], 
        template=template
    )
    
    # Custom RAG function that limits context length
    def rag_chain_func(inputs):
        question = inputs["input"]
        # Retrieve documents
        docs = retriever.invoke(question)
        
        # Limit context to prevent token overflow
        context_parts = []
        total_chars = 0
        max_chars = 800  # Reduced from 4916 to fit in 512 tokens
        
        for doc in docs:
            content = doc.page_content
            if total_chars + len(content) > max_chars:
                # Take partial content to fit limit
                remaining = max_chars - total_chars
                if remaining > 100:  # Only add if meaningful length
                    content = content[:remaining] + "..."
                    context_parts.append(content)
                break
            else:
                context_parts.append(content)
                total_chars += len(content)
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        full_prompt = prompt.format(context=context, input=question)
        
        # Generate answer
        answer = llm.invoke(full_prompt)
        
        return {
            "input": question,
            "context": docs,
            "answer": answer
        }
    
    from langchain.schema.runnable import RunnableLambda
    return RunnableLambda(rag_chain_func)


def run_interactive_session(rag_chain):
    """Interactive Q&A session"""
    print("\n" + "="*60)
    print("🤖 RAG SYSTEM READY - Interactive Mode")
    print("="*60)
    print("Ask questions about your research papers!")
    print("Type 'quit' or 'exit' to stop")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                print("Please enter a question.")
                continue
            
            print("\n🔍 Searching and generating answer...")
            
            # Get response
            result = rag_chain.invoke({"input": question})
            
            # Display answer
            print("\n" + "="*50)
            print("🎯 ANSWER:")
            print("="*50)
            print(result['answer'])
            
            # Display sources
            print("\n" + "="*50)
            print("📄 SOURCES:")
            print("="*50)
            for i, doc in enumerate(result['context'], 1):
                source = doc.metadata.get('source', 'Unknown source')
                chunk = doc.metadata.get('chunk_index', 'Unknown chunk')
                print(f"\n{i}. {source} (Chunk {chunk})")
                content_preview = doc.page_content[:150].replace('\n', ' ')
                print(f"   Preview: {content_preview}...")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing question: {e}")
            print("Please try again with a different question.")


def main():
    """Main pipeline execution"""
    print("🚀 Enhanced RAG Pipeline")
    print("="*50)
    
    try:
        # Load configuration
        cfg = load_config()
        print("✅ Configuration loaded")
        
        # Setup embeddings
        embed_model_name = (
            cfg.get('retrieval', {}).get('embedding_model')
            or 'sentence-transformers/all-MiniLM-L6-v2'
        )
        print(f"Using embedding model: {embed_model_name}")
        embedder = HuggingFaceEmbeddings(model_name=embed_model_name)
        
        # Load vector store
        vector_store = load_vector_store(cfg, embedder)
        print("✅ Vector store loaded")
        
        # DEBUG: Test vector store
        test_retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        test_docs = test_retriever.invoke("test")
        print(f"DEBUG_VECTOR: docs={len(test_docs)}, first_doc_len={len(test_docs[0].page_content) if test_docs else 0}")
        
        # Setup LLM
        llm = setup_llm(cfg)
        print("✅ Language model ready")
        
        # DEBUG: Test LLM
        test_prompt = "What is AI?"
        test_response = llm.invoke(test_prompt)
        print(f"DEBUG_LLM: input_len={len(test_prompt)}, output_len={len(test_response)}, output_preview='{test_response[:30]}...'")
        
        # Create RAG chain
        rag_chain = create_rag_chain(vector_store, llm, cfg)
        print("✅ RAG chain created")
        
        # Smoke test
        print("\n" + "="*50)
        print("🧪 SMOKE TEST")
        print("="*50)
        question = "What is retrieval-augmented generation?"
        print(f"Test question: {question}")
        
        result = rag_chain.invoke({"input": question})
        
        # DEBUG: Critical pipeline metrics
        answer_len = len(result['answer'])
        context_docs = len(result['context'])
        sources = [doc.metadata.get('source', 'Unknown')[:20] for doc in result['context']]
        
        print(f"DEBUG_RAG: answer_len={answer_len}, docs_retrieved={context_docs}, context_chars=limited_to_800")
        print(f"DEBUG_SOURCES: {sources}")
        print(f"DEBUG_ANSWER_PREVIEW: '{result['answer'][:80]}...'")
        
        print(f"\n🎯 Answer: {result['answer']}")
        
        # Show sources
        print(f"\n📚 Sources used: {len(result['context'])} documents")
        for i, doc in enumerate(result['context'][:2], 1):  # Show first 2 sources
            source = doc.metadata.get('source', 'Unknown')[:50]
            print(f"  {i}. {source}...")
        
        # Ask user if they want interactive mode
        print("\n" + "="*60)
        response = input("Would you like to enter interactive mode? (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            run_interactive_session(rag_chain)
        else:
            print("Pipeline test completed successfully! 🎉")
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        print(f"DEBUG_ERROR: {type(e).__name__}: {str(e)[:100]}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you've run rebuild_faiss_index.py first")
        print("2. Check that your config.yaml paths are correct")
        print("3. Ensure you have the required packages installed")


if __name__ == '__main__':
    main()