import os
import re
from nbconvert import PythonExporter
import nbformat

def convert_notebook_to_script(input_notebook_path, output_script_path):
    # Load the notebook
    with open(input_notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Convert the notebook to Python code
    python_exporter = PythonExporter()
    python_code, _ = python_exporter.from_notebook_node(notebook)
    
    # Post-processing the generated Python code
    python_code = clean_code(python_code)
    
    # Save the cleaned Python code to the output file
    with open(output_script_path, "w", encoding="utf-8") as f:
        f.write(python_code)
    
    print(f"Converted '{input_notebook_path}' to Python script '{output_script_path}'")


def clean_code(python_code):
    """
    Function to clean up the converted Python code:
    1. Remove Jupyter-specific code (e.g., `In[]`, `Out[]`).
    2. Remove IPython magics like `%time`, `%matplotlib inline`.
    3. Organize imports at the top.
    4. Remove unnecessary comments.
    """

    # Remove Jupyter's `In[]`, `Out[]` cell markers
    python_code = re.sub(r'# In\[\d+\]:', '', python_code)  # Remove In[x]: cell headers
    python_code = re.sub(r'# Out\[\d+\]:', '', python_code)  # Remove Out[x]: cell markers
    
    # Remove IPython magics like `%time`, `%matplotlib inline`, etc.
    python_code = re.sub(r'^%[^\n]*\n?', '', python_code, flags=re.MULTILINE)
    python_code = re.sub(r'^[!#][^\n]*\n?', '', python_code, flags=re.MULTILINE)  # Remove shell commands
    
    # Remove unnecessary comments (e.g., In[ ]: cells, or redundant code comments)
    python_code = re.sub(r'^\s*#.*\n', '', python_code)  # Remove all comment lines
    
    # Ensure imports are at the top of the script
    python_code = organize_imports(python_code)
    
    # Optional: Modularize the code into functions (if desired)
    # (You could manually refactor the code here or add specific patterns for refactoring)
    
    return python_code


def organize_imports(python_code):
    """
    Function to organize the imports at the top of the script.
    """
    # Regular expression to capture all imports
    import_pattern = r'^(import .*\n|from .*\n)'
    
    # Find all import statements in the code
    imports = re.findall(import_pattern, python_code)
    
    # Remove the import statements from their current locations
    python_code = re.sub(import_pattern, '', python_code)
    
    # Place the imports at the top of the script
    python_code = '\n'.join(imports) + '\n\n' + python_code
    
    return python_code


if __name__ == '__main__':
    # Define the input notebook path and the output Python script path
    #input_notebook_path = "../notebooks/2_document_embedding.ipynb"  # Path to the notebook you want to convert
    #output_script_path = "../modules/document_embedding.py"  # Path to save the converted Python script

    #input_notebook_path = "../notebooks/3_retrieval_system.ipynb"  # Path to the notebook you want to convert
    #output_script_path = "../modules/retrieval_system.py"  # Path to save the converted Python script

    #input_notebook_path = "../notebooks/4_context_integration.ipynb"  # Path to the notebook you want to convert
    #output_script_path = "../modules/context_integration.py"  # Path to save the converted Python script

    input_notebook_path = "../notebooks/5_generation.ipynb"  # Path to the notebook you want to convert
    output_script_path = "../modules/generation.py"  # Path to save the converted Python script

    #input_notebook_path = "../notebooks/6_rag_pipeline.ipynb"  # Path to the notebook you want to convert
    #output_script_path = "../modules/rag_pipeline.py"  # Path to save the converted Python script
    
    # Convert and clean the notebook
    convert_notebook_to_script(input_notebook_path, output_script_path)
