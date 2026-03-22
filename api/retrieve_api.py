from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pickle
import numpy as np
import faiss
import json
import re
from collections import defaultdict, Counter
from openrouter_embedder import OpenRouterEmbedder
from typing import List, Optional
import uvicorn

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 3
    use_smart_retrieval: Optional[bool] = True

class RetrievalResponse(BaseModel):
    query: str
    retrieved_contexts: List[str]
    metadata: dict
    processing_info: dict

# --- Configuration ---
EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
EMBEDDING_DIM = 2048  # Nemotron model produces 2048-dimensional vectors
DATA_DIR = os.path.join(os.path.dirname(__file__), "..")  # Parent directory where pkl files are stored

# --- Initialize FastAPI ---
app = FastAPI(
    title="CSEC Retrieval API",
    description="API for retrieving relevant context from CSEC syllabus content",
    version="1.0.0"
)

# --- Global Variables (loaded on startup) ---
embedder = None
paragraphs = None
chunk_metadata = None
stats = None
index = None

@app.on_event("startup")
async def load_models():
    """Load all models and data on startup"""
    global embedder, paragraphs, chunk_metadata, stats, index
    
    print("🚀 Loading models and data...")
    
    # Load embedding model (OpenRouter)
    embedder = OpenRouterEmbedder(model=EMBED_MODEL)
    print("✅ Embedding model loaded from OpenRouter")
    
    # Load FAISS index and data
    try:
        embeddings = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
        with open(os.path.join(DATA_DIR, "paragraphs.pkl"), "rb") as f:
            paragraphs = pickle.load(f)
        index = faiss.read_index(os.path.join(DATA_DIR, "faiss.index"))
        print(f"✅ Loaded {len(paragraphs)} paragraphs and FAISS index")
    except FileNotFoundError as e:
        print(f"❌ Error loading required files: {e}")
        raise
    
    # Load metadata (optional)
    try:
        with open(os.path.join(DATA_DIR, "chunk_metadata.pkl"), "rb") as f:
            chunk_metadata = pickle.load(f)
        print(f"✅ Loaded metadata for {len(chunk_metadata)} chunks")
    except FileNotFoundError:
        print("⚠️ No metadata file found - using basic retrieval only")
        chunk_metadata = None
    
    # Load stats (optional)
    try:
        with open(os.path.join(DATA_DIR, "indexing_stats.json"), "r") as f:
            stats = json.load(f)
        print(f"✅ Loaded stats: {stats.get('files_processed', 'unknown')} subjects")
    except FileNotFoundError:
        print("⚠️ No stats file found")
        stats = {'subjects': [], 'content_types': {}}

def analyze_query(query: str) -> dict:
    """Analyze query to understand what type of information is needed"""
    query_lower = query.lower()
    
    # Intent detection
    intent_patterns = {
        'explanation': ['explain', 'what is', 'how does', 'describe', 'define'],
        'procedure': ['how to', 'steps', 'process', 'method', 'procedure'],
        'examples': ['example', 'show me', 'demonstrate', 'instance'],
        'study_plan': ['study plan', 'prepare', 'review', 'study guide', 'what should i study'],
        'assessment': ['exam', 'test', 'assessment', 'marks', 'grade'],
        'comparison': ['difference', 'compare', 'versus', 'vs', 'contrast'],
        'list': ['list', 'types of', 'kinds of', 'categories']
    }
    
    detected_intent = 'general'
    for intent, patterns in intent_patterns.items():
        if any(pattern in query_lower for pattern in patterns):
            detected_intent = intent
            break
    
    # Enhanced subject detection with aliases
    subject_aliases = {
        'it': ['INFORMATION_TECHNOLOGY', 'COMPUTING', 'IT_CSEC', 'COMPUTER'],
        'information technology': ['INFORMATION_TECHNOLOGY', 'COMPUTING'],
        'math': ['MATHEMATICS', 'MATHS'],
        'maths': ['MATHEMATICS', 'MATH'],
        'additional mathematics': ['ADDITIONAL_MATHEMATICS', 'ADVANCED_MATHEMATICS'],
        'add math': ['ADDITIONAL_MATHEMATICS', 'ADVANCED_MATHEMATICS'],
        'addl math': ['ADDITIONAL_MATHEMATICS', 'ADVANCED_MATHEMATICS'],
        'add maths': ['ADDITIONAL_MATHEMATICS', 'ADVANCED_MATHEMATICS'],
        'bio': ['BIOLOGY', 'BIO'],
        'chem': ['CHEMISTRY', 'CHEM'],
        'physics': ['PHYSICS', 'PHYSICS_CSEC', 'PHYS'],
        'history': ['HISTORY'],
        'english': ['ENGLISH'],
        'geography': ['GEOGRAPHY'],
        'cxc': ['CSEC', 'CXC_CSEC', 'CARIBBEAN_EXAMINATIONS_COUNCIL'],
        'csec': ['CSEC', 'CXC_CSEC', 'CARIBBEAN_EXAMINATIONS_COUNCIL'],
        'spanish': ['SPANISH', 'span', 'español'],
        'french': ['FRENCH'],
    }
    
    detected_subjects = []
    
    # Check aliases first
    for alias, full_names in subject_aliases.items():
        if alias in query_lower:
            for subject in stats.get('subjects', []):
                if any(name.lower() in subject.lower() for name in full_names):
                    detected_subjects.append(subject)
    
    # Also check direct matches
    for subject in stats.get('subjects', []):
        if subject.lower() in query_lower:
            detected_subjects.append(subject)
    
    # Extract key terms from query
    query_terms = re.findall(r'\b[a-zA-Z]{3,}\b', query_lower)
    
    return {
        'intent': detected_intent,
        'subjects': list(set(detected_subjects)),
        'key_terms': query_terms
    }

def smart_retrieve(query: str, k: int = 3) -> tuple:
    """Enhanced retrieval with intent-aware ranking"""
    if chunk_metadata is None:
        return basic_retrieve(query, k)
    
    query_analysis = analyze_query(query)
    
    # Create enhanced query for better semantic matching
    enhanced_query = f"query: {query}"  # E5-style prompting
    if query_analysis['intent'] != 'general':
        enhanced_query = f"query: {query_analysis['intent']}: {query}"
    
    # Get initial candidates (more if filtering by subject)
    search_k = k * 10 if query_analysis['subjects'] else k * 3
    query_emb = embedder.encode([enhanced_query], convert_to_numpy=True)
    D, I = index.search(query_emb, search_k)
    
    processing_info = {
        'detected_intent': query_analysis['intent'],
        'detected_subjects': query_analysis['subjects'],
        'enhanced_query': enhanced_query,
        'initial_candidates': len(I[0]),
        'subject_filtered': False
    }
    
    # Filter by subject if detected
    if query_analysis['subjects']:
        subject_filtered = []
        for distance, idx in zip(D[0], I[0]):
            if idx < len(chunk_metadata):
                chunk_subject = chunk_metadata[idx].get('subject', '')
                if any(subj.lower() in chunk_subject.lower() for subj in query_analysis['subjects']):
                    subject_filtered.append((distance, idx))
        
        if subject_filtered:
            processing_info['subject_filtered'] = True
            processing_info['filtered_candidates'] = len(subject_filtered)
            # Use filtered results
            candidates = []
            for distance, idx in subject_filtered[:k*2]:
                if idx < len(paragraphs):
                    candidates.append({
                        'text': paragraphs[idx],
                        'metadata': chunk_metadata[idx],
                        'distance': distance,
                        'doc_index': idx
                    })
        else:
            # Fallback to general search
            candidates = []
            for distance, idx in zip(D[0][:k*2], I[0][:k*2]):
                if idx < len(paragraphs) and idx < len(chunk_metadata):
                    candidates.append({
                        'text': paragraphs[idx],
                        'metadata': chunk_metadata[idx],
                        'distance': distance,
                        'doc_index': idx
                    })
    else:
        # General search
        candidates = []
        for distance, idx in zip(D[0][:k*2], I[0][:k*2]):
            if idx < len(paragraphs) and idx < len(chunk_metadata):
                candidates.append({
                    'text': paragraphs[idx],
                    'metadata': chunk_metadata[idx],
                    'distance': distance,
                    'doc_index': idx
                })
    
    # Re-rank based on intent and relevance
    if candidates:
        scored_candidates = []
        for candidate in candidates:
            score = 1.0 / (1.0 + candidate['distance'])  # Convert distance to similarity
            
            # Boost based on content type matching intent
            content_type = candidate['metadata'].get('content_type', 'general')
            intent = query_analysis['intent']
            
            type_intent_match = {
                'explanation': ['definitions', 'general', 'objectives'],
                'procedure': ['procedures', 'requirements', 'format'],
                'examples': ['examples'],
                'study_plan': ['objectives', 'assessment', 'requirements'],
                'assessment': ['assessment', 'format', 'requirements'],
                'comparison': ['definitions', 'examples'],
                'list': ['general', 'objectives']
            }
            
            if intent in type_intent_match and content_type in type_intent_match[intent]:
                score *= 1.3
            
            # Boost based on key term overlap
            candidate_terms = set(candidate['metadata'].get('key_terms', []))
            query_terms = set(query_analysis['key_terms'])
            term_overlap = len(candidate_terms.intersection(query_terms))
            if term_overlap > 0:
                score *= (1.0 + 0.1 * term_overlap)
            
            scored_candidates.append((score, candidate))
        
        # Sort by score and return top k
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        results = [candidate['text'] for score, candidate in scored_candidates[:k]]
        
        # Add metadata for each result
        result_metadata = []
        for score, candidate in scored_candidates[:k]:
            result_metadata.append({
                'subject': candidate['metadata'].get('subject', 'unknown'),
                'content_type': candidate['metadata'].get('content_type', 'unknown'),
                'similarity_score': float(score),
                'distance': float(candidate['distance'])
            })
        
        processing_info['result_metadata'] = result_metadata
    else:
        results = []
        processing_info['result_metadata'] = []
    
    return results, processing_info

def basic_retrieve(query: str, k: int = 3) -> tuple:
    """Basic retrieval function (fallback when no metadata available)"""
    formatted_query = f"query: {query}"  # E5-style prompting
    query_emb = embedder.encode([formatted_query], convert_to_numpy=True)
    D, I = index.search(query_emb, k)

    results = [paragraphs[i] for i in I[0]]
    
    processing_info = {
        'method': 'basic_retrieval',
        'query_used': formatted_query,
        'distances': [float(d) for d in D[0]],
        'indices': [int(i) for i in I[0]]
    }
    
    return results, processing_info

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "CSEC Retrieval API is running",
        "loaded_paragraphs": len(paragraphs) if paragraphs else 0,
        "has_metadata": chunk_metadata is not None,
        "available_subjects": stats.get('subjects', []) if stats else []
    }

@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_context(request: QueryRequest):
    """
    Retrieve relevant context chunks for a given query
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if request.k < 1 or request.k > 20:
            raise HTTPException(status_code=400, detail="k must be between 1 and 20")
        
        # Perform retrieval
        if request.use_smart_retrieval and chunk_metadata is not None:
            contexts, processing_info = smart_retrieve(request.query, request.k)
            processing_info['method'] = 'smart_retrieval'
        else:
            contexts, processing_info = basic_retrieve(request.query, request.k)
        
        # Prepare response
        response = RetrievalResponse(
            query=request.query,
            retrieved_contexts=contexts,
            metadata={
                'total_available_chunks': len(paragraphs),
                'retrieved_count': len(contexts),
                'has_metadata': chunk_metadata is not None,
                'available_subjects': stats.get('subjects', []) if stats else []
            },
            processing_info=processing_info
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get statistics about the loaded data"""
    if not paragraphs:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    response = {
        "total_chunks": len(paragraphs),
        "has_metadata": chunk_metadata is not None,
        "embedding_model": EMBED_MODEL,
        "embedding_dimension": EMBEDDING_DIM
    }
    
    if stats:
        response.update({
            "subjects": stats.get('subjects', []),
            "content_types": stats.get('content_types', {}),
            "files_processed": stats.get('files_processed', 0)
        })
    
    if chunk_metadata:
        # Calculate some statistics
        subjects = [meta.get('subject', 'unknown') for meta in chunk_metadata]
        content_types = [meta.get('content_type', 'unknown') for meta in chunk_metadata]
        
        response.update({
            "subject_distribution": dict(Counter(subjects)),
            "content_type_distribution": dict(Counter(content_types))
        })
    
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)