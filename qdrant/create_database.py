import ast
import os
from pathlib import Path
from typing import List, Dict, Any
import json
import qdrant_client
from qdrant_client import models
from sentence_transformers import SentenceTransformer
import shutil

this_dir = os.path.dirname(os.path.abspath(__file__))

def extract_docstring(node: ast.AST) -> tuple[str, str]:
    """Extract docstring from an AST node and its header.
    
    Returns:
        tuple[str, str]: (full_docstring, docstring_header)
    """
    if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
        return "", ""
    
    docstring = ast.get_docstring(node)
    if not docstring:
        return "", ""
    
    # Split the docstring into lines and remove leading/trailing whitespace
    lines = [line.strip() for line in docstring.split('\n')]
    
    # Get the first non-empty line as the header
    header = next((line for line in lines if line), "")
    
    # If the header is a single line and the next line is empty, that's our header
    # Otherwise, we might be in a multi-line description, so we need to find the first paragraph
    if len(lines) > 1 and not lines[1]:
        return docstring, header
    
    # Find the first paragraph (until we hit an empty line or a line that starts with a space)
    header_lines = []
    for line in lines:
        if not line or line.startswith(' '):
            break
        header_lines.append(line)
    
    return docstring, ' '.join(header_lines)

def get_source_code(node: ast.AST, source: str) -> str:
    """Get the source code for a node."""
    if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return ""
    
    start_lineno = node.lineno
    end_lineno = node.end_lineno if hasattr(node, 'end_lineno') else start_lineno
    lines = source.split('\n')
    return '\n'.join(lines[start_lineno-1:end_lineno])

def analyze_file(file_path: str) -> List[Dict[str, Any]]:
    """Analyze a Python file and extract function and class information."""
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    tree = ast.parse(source)
    results = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            docstring, docstring_header = extract_docstring(node)
            result = {
                'name': node.name,
                'type': 'function' if isinstance(node, ast.FunctionDef) else 'class',
                'docstring': docstring,
                'docstring_header': docstring_header,
                'source_code': get_source_code(node, source),
                'file': file_path
            }
            results.append(result)
    
    return results

def analyze_package(package_path: str) -> List[Dict[str, Any]]:
    """Analyze all Python files in a package directory."""
    package_dir = Path(package_path)
    all_results = []
    
    for py_file in package_dir.rglob('*.py'):
        if py_file.name != '__init__.py':  # Skip __init__.py files
            results = analyze_file(str(py_file))
            all_results.extend(results)
    
    return all_results

def create_database(docs: List[Dict[str, Any]], db_path: str):
    # Remove existing database if it exists
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        
    # Create a qdrant client using an in-memory instance
    client = qdrant_client.QdrantClient(path=db_path)    

    # Create a sentence transformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Create a collection
    client.create_collection(
        collection_name="docs",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
            distance=models.Distance.COSINE,
        ),
    )

    # Upload the docs
    client.upload_points(
        collection_name="docs",
        points=[
            models.PointStruct(
                id=idx, vector=encoder.encode(doc["docstring_header"]).tolist(), payload=doc
            )
            for idx, doc in enumerate(docs)
        ],
    )    

def main(verbose: bool = False):
    # Path to the rastertoolkit package
    package_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'rastertoolkit')
    
    # Analyze the package
    results = analyze_package(package_path)

    # Save results to a JSONL file
    with open(os.path.join(this_dir, 'data', 'rastertoolkit_docs.jsonl'), 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')

    # Create the database
    create_database(results, os.path.join(this_dir, 'data', 'rastertoolkit_docs.db'))
    
    # Print results
    if verbose:
        for item in results:
            print(f"\n{'='*80}")
            print(f"Name: {item['name']}")
            print(f"Type: {item['type']}")
            print(f"File: {item['file']}")
            print(f"\nDocstring Header:\n{item['docstring_header']}")
            print(f"\nFull Docstring:\n{item['docstring']}")
            print(f"\nSource Code:\n{item['source_code']}")

if __name__ == "__main__":
    main()
