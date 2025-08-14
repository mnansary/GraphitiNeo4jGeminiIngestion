import os
import fnmatch

# --- SCRIPT CONFIGURATION ---

# 1. The root folder of your project that you want to scan.
#    (e.g., 'my_cool_project')
ROOT_DIRECTORY = "./"  # <--- CHANGE THIS

# 2. The name of the final combined file.
OUTPUT_FILENAME = 'codebase.txt'

# 3. Exclusion lists: Add any patterns, names, or extensions to ignore.
#    Uses standard Unix shell-style wildcards (e.g., *, ?, [abc]).

# Ignored directory names (will not be entered)
EXCLUDED_DIRECTORIES = [
    '__pycache__',
    '.git',
    '.vscode',
    'venv',
    '.venv',
    'node_modules',
    'dist',
    'build',
    '*.egg-info'
]

# Ignored file names or patterns
EXCLUDED_FILES = [
    '.gitignore',
    'debug*',
    '*.pyc',
    '*.swp',
    '.DS_Store',
    'configs/apis.csv',
    OUTPUT_FILENAME # Exclude the output file itself
]
# --- END OF CONFIGURATION ---


def should_exclude(path, is_dir):
    """Check if a file or directory should be excluded based on the lists."""
    base_name = os.path.basename(path)
    
    exclusion_list = EXCLUDED_DIRECTORIES if is_dir else EXCLUDED_FILES
    
    for pattern in exclusion_list:
        if fnmatch.fnmatch(base_name, pattern):
            return True
    return False


def create_codebase_file():
    """
    Walks through the project directory, reads the content of allowed files,
    and consolidates them into a single output file.
    """
    if not os.path.isdir(ROOT_DIRECTORY):
        print(f"Error: Root directory '{ROOT_DIRECTORY}' not found.")
        return

    print(f"Starting to process files in '{ROOT_DIRECTORY}'...")
    file_count = 0

    try:
        # Open the output file with UTF-8 encoding
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8', errors='ignore') as outfile:
            
            for dirpath, dirnames, filenames in os.walk(ROOT_DIRECTORY, topdown=True):
                # --- Directory Exclusion ---
                # Modify dirnames in-place to prevent os.walk from descending
                # into the excluded directories.
                dirnames[:] = [d for d in dirnames if not should_exclude(os.path.join(dirpath, d), is_dir=True)]
                
                for filename in filenames:
                    # --- File Exclusion ---
                    if should_exclude(os.path.join(dirpath, filename), is_dir=False):
                        continue

                    file_path = os.path.join(dirpath, filename)
                    
                    try:
                        # Write a clear header for each file
                        header = f"--- File: {os.path.relpath(file_path, ROOT_DIRECTORY)} ---"
                        outfile.write("=" * 80 + "\n")
                        outfile.write(header + "\n")
                        outfile.write("=" * 80 + "\n\n")
                        
                        # Read and write the file content
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                            outfile.write(infile.read())
                        
                        outfile.write("\n\n")
                        file_count += 1

                    except Exception as e:
                        print(f"Could not read file {file_path}: {e}")

    except IOError as e:
        print(f"Error writing to output file {OUTPUT_FILENAME}: {e}")
        return

    print(f"Finished! Consolidated {file_count} files into '{OUTPUT_FILENAME}'.")


if __name__ == '__main__':
    create_codebase_file()
