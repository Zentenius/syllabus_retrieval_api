import os
import glob
import pickle
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
import faiss
import json
import re
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
syllabus_folder = "syllabus"
output_preview = "preview_chunks.txt"
debug_sample = "debug_sample_chunks.txt"
output_pickle = "paragraphs.pkl"
output_metadata = "chunk_metadata.pkl"
output_npy = "embeddings.npy"
output_faiss = "faiss.index"
output_stats = "indexing_stats.json"

# Delete previous files (optional reset)
for path in [output_preview, output_pickle, output_npy, output_faiss, debug_sample, output_metadata, output_stats]:
    if os.path.exists(path):
        os.remove(path)

# Load spaCy model for sentence segmentation and NER
nlp = spacy.load("en_core_web_sm")

# Use consistent embedding model (same for indexing and querying)
embedder = SentenceTransformer("intfloat/multilingual-e5-base")

def extract_key_terms(text, top_n=5):
    """Extract key terms using TF-IDF and NER"""
    # Clean text for TF-IDF
    clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # Use TF-IDF to get important terms
    try:
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform([clean_text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Get top terms by TF-IDF score
        top_indices = scores.argsort()[-top_n:][::-1]
        tfidf_terms = [feature_names[i] for i in top_indices if scores[i] > 0]
    except:
        tfidf_terms = []
    
    # Extract named entities and important nouns
    doc = nlp(text[:1000000])  # Limit text length for spaCy
    entities = [ent.text.lower() for ent in doc.ents if len(ent.text) > 2]
    nouns = [token.lemma_.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2]
    
    # Combine and deduplicate
    all_terms = list(set(tfidf_terms + entities + nouns))
    return all_terms[:top_n]

def detect_content_type(text):
    """Automatically detect the type of content (objectives, examples, definitions, etc.)"""
    text_lower = text.lower()
    
    # Pattern matching for different content types
    patterns = {
        'objectives': [r'students? should be able to', r'objectives?:', r'aims?:', r'learning outcomes?:', r'specific objectives?:'],
        'assessment': [r'assessment', r'examination', r'paper \d+', r'marks?', r'grading', r'evaluation'],
        'examples': [r'example', r'for instance', r'such as', r'e\.g\.', r'illustration'],
        'definitions': [r'definition', r'defined as', r'refers? to', r'means?', r'is the'],
        'procedures': [r'steps?', r'procedure', r'method', r'process', r'algorithm', r'technique'],
        'requirements': [r'must', r'required', r'should', r'need', r'essential', r'mandatory'],
        'format': [r'format', r'structure', r'layout', r'organization', r'arrangement']
    }
    
    content_scores = defaultdict(int)
    for content_type, type_patterns in patterns.items():
        for pattern in type_patterns:
            matches = len(re.findall(pattern, text_lower))
            content_scores[content_type] += matches
    
    if content_scores:
        return max(content_scores, key=content_scores.get)
    return "general"

def extract_section_context(text, chunk_start_pos):
    """Extract surrounding section context for a chunk"""
    lines = text.split('\n')
    chunk_line = 0
    char_count = 0
    
    # Find which line the chunk starts on
    for i, line in enumerate(lines):
        if char_count >= chunk_start_pos:
            chunk_line = i
            break
        char_count += len(line) + 1
    
    # Look backwards for section headers
    section_headers = []
    for i in range(max(0, chunk_line - 10), chunk_line):
        line = lines[i].strip()
        if (line.isupper() and len(line) > 5) or \
           re.match(r'^\d+\.', line) or \
           re.match(r'^[A-Z][A-Z\s]{5,}', line) or \
           any(word in line.upper() for word in ['SECTION', 'PAPER', 'UNIT', 'CHAPTER']):
            section_headers.append(line)
    
    return section_headers[-1] if section_headers else "General"

def enhanced_semantic_chunk(filepath, max_words=120):
    """
    Enhanced chunking with automatic content analysis
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().strip()
    
    subject_name = os.path.basename(filepath).replace(".txt", "")
    
    nlp.max_length = 2_000_000
    doc = nlp(text)
    chunks = []
    chunk_metadata = []
    current_chunk = []
    current_length = 0
    current_pos = 0
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_len = len(sent_text.split())
        
        if current_length + sent_len > max_words and current_chunk:
            # Create chunk
            chunk_text = " ".join(current_chunk).strip()
            
            if len(chunk_text.split()) >= 8:
                # Analyze chunk content
                key_terms = extract_key_terms(chunk_text)
                content_type = detect_content_type(chunk_text)
                section_context = extract_section_context(text, current_pos)
                
                # Create enhanced label
                enhanced_label = f"[{subject_name.upper()}|{content_type}|{section_context[:30]}]"
                tagged_chunk = f"{enhanced_label} {chunk_text}"
                
                chunks.append(tagged_chunk)
                chunk_metadata.append({
                    'subject': subject_name,
                    'content_type': content_type,
                    'section_context': section_context,
                    'key_terms': key_terms,
                    'word_count': len(chunk_text.split()),
                    'char_position': current_pos
                })
            
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sent_text)
        current_length += sent_len
        current_pos += len(sent_text) + 1
    
    # Handle final chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        if len(chunk_text.split()) >= 8:
            key_terms = extract_key_terms(chunk_text)
            content_type = detect_content_type(chunk_text)
            section_context = extract_section_context(text, current_pos)
            
            enhanced_label = f"[{subject_name.upper()}|{content_type}|{section_context[:30]}]"
            tagged_chunk = f"{enhanced_label} {chunk_text}"
            
            chunks.append(tagged_chunk)
            chunk_metadata.append({
                'subject': subject_name,
                'content_type': content_type,
                'section_context': section_context,
                'key_terms': key_terms,
                'word_count': len(chunk_text.split()),
                'char_position': current_pos
            })
    
    return chunks, chunk_metadata

# Process each file and create enhanced chunks
all_paragraphs = []
all_metadata = []
stats = {'files_processed': 0, 'total_chunks': 0, 'content_types': Counter(), 'subjects': []}

with open(debug_sample, "w", encoding="utf-8") as debug_file:
    for filepath in glob.glob(os.path.join(syllabus_folder, "*.txt")):
        subject = os.path.basename(filepath).replace(".txt", "").upper()
        chunks, metadata = enhanced_semantic_chunk(filepath, max_words=120)
        
        print(f"📄 {subject}.txt → {len(chunks)} chunks")
        debug_file.write(f"\n=== {subject}.txt ===\n")
        
        # Show sample chunks with metadata
        for i, (chunk, meta) in enumerate(zip(chunks[:3], metadata[:3])):
            debug_file.write(f"[{i+1}] Content Type: {meta['content_type']}\n")
            debug_file.write(f"    Key Terms: {', '.join(meta['key_terms'])}\n")
            debug_file.write(f"    Section: {meta['section_context']}\n")
            debug_file.write(f"    Chunk: {chunk}\n{'-'*60}\n")
        
        all_paragraphs.extend(chunks)
        all_metadata.extend(metadata)
        
        # Update stats
        stats['files_processed'] += 1
        stats['total_chunks'] += len(chunks)
        stats['subjects'].append(subject)
        for meta in metadata:
            stats['content_types'][meta['content_type']] += 1

# Save full chunk preview
with open(output_preview, "w", encoding="utf-8") as f:
    for i, (para, meta) in enumerate(zip(all_paragraphs, all_metadata)):
        f.write(f"[{i+1}] Type: {meta['content_type']} | Terms: {', '.join(meta['key_terms'][:3])}\n")
        f.write(f"{para}\n{'-'*60}\n")

# Save data
with open(output_pickle, "wb") as f:
    pickle.dump(all_paragraphs, f)

with open(output_metadata, "wb") as f:
    pickle.dump(all_metadata, f)

with open(output_stats, "w") as f:
    json.dump(stats, f, indent=2)

print("🔍 Embedding all chunks...")
embeddings = embedder.encode(all_paragraphs, convert_to_numpy=True, batch_size=32, show_progress_bar=True)
np.save(output_npy, embeddings)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, output_faiss)

print(f"✅ Processed {len(all_paragraphs)} total chunks from {stats['files_processed']} files.")
print(f"📊 Content types found: {dict(stats['content_types'])}")
print(f"📄 Preview saved to: {output_preview}")
print(f"🧪 Debug sample saved to: {debug_sample}")
print(f"📈 Stats saved to: {output_stats}")