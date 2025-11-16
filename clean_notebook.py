import json
import sys

def clean_notebook(input_path, output_path):
    """
    Reads a Jupyter notebook, clears all cell outputs, resets execution counts,
    and saves the cleaned notebook to a new file.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                # Clear outputs
                cell['outputs'] = []
                # Reset execution count
                cell['execution_count'] = None
                # Optional: clear metadata that might contain execution info
                if 'metadata' in cell:
                    cell['metadata'].pop('execution', None)
                    cell['metadata'].pop('scrolled', None)
                    cell['metadata'].pop('collapsed', None)

        # Clear notebook-level metadata related to execution
        if 'metadata' in notebook:
            notebook['metadata'].pop('kernelspec', None)
            notebook['metadata'].pop('language_info', None)


        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)

        print(f"Successfully cleaned notebook and saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python clean_notebook.py <input_notebook> <output_notebook>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    clean_notebook(input_file, output_file)
