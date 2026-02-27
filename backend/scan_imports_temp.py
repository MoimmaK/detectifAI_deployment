import ast
import os
import sys

# Debug print
print("Script started")

def get_imports(path):
    with open(path, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=path)
        except SyntaxError:
            print(f"SyntaxError in {path}")
            return set()
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return set()
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports

start_dir = r"d:\fyp\noErrorsFinSem1\sem1\sem1\backend"
print(f"Scanning directory: {start_dir}")

all_imports = set()
file_count = 0

if not os.path.exists(start_dir):
    print(f"Directory does not exist: {start_dir}")

for root, dirs, files in os.walk(start_dir):
    if 'venv' in dirs:
        dirs.remove('venv')
    if '__pycache__' in dirs:
        dirs.remove('__pycache__')
        
    for file in files:
        if file.endswith(".py"):
            file_count += 1
            full_path = os.path.join(root, file)
            found = get_imports(full_path)
            # print(f"File: {file}, Imports: {found}")
            all_imports.update(found)

print(f"Scanned {file_count} Python files.")

# Common stdlib modules to filter out
common_stdlib = {
    'os', 'sys', 're', 'json', 'time', 'datetime', 'math', 'random', 'subprocess', 
    'threading', 'multiprocessing', 'collections', 'itertools', 'functools', 'logging', 
    'abc', 'types', 'typing', 'uuid', 'base64', 'hashlib', 'io', 'shutil', 'glob', 
    'pickle', 'socket', 'urllib', 'http', 'email', 'argparse', 'csv', 'sqlite3', 
    'unittest', 'doctest', 'pdb', 'inspect', 'traceback', 'platform', 'warnings', 
    'contextlib', 'copy', 'weakref', 'enum', 'struct', 'textwrap', 'pathlib', 'signal',
    'tempfile', 'shlex', 'mimetypes', 'calendar', 'ast', 'dis', 'gc', 'site', 'builtins',
    'queue', 'errno', 'stat', 'fnmatch', 'token', 'tokenize'
}

# Try to get stdlib from sys if available (3.10+)
if hasattr(sys, 'stdlib_module_names'):
    common_stdlib.update(sys.stdlib_module_names)

non_std_imports = sorted([i for i in all_imports if i not in common_stdlib])

print("FOUND_IMPORTS:", ", ".join(non_std_imports))
