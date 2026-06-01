import os

def combine_files(output_filename="combined_file.txt", target_dir="."):
    # Directories to skip entirely
    ignored_dirs = {"venv", ".git", "__pycache__", ".vscode"}
    
    # Specific files to ignore (for privacy and avoiding self-inclusion)
    ignored_files = {output_filename, ".env", ".gitignore", ".DS_Store"}
    
    # Binary or non-text extensions to skip
    ignored_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".pyc", ".db", ".zip", ".tar", ".gz"}

    print(f"Starting compilation into '{output_filename}'...\n")

    with open(output_filename, "w", encoding="utf-8") as outfile:
        for root, dirs, files in os.walk(target_dir):
            # Modify dirs in-place so os.walk does not traverse ignored directories
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            
            for file in files:
                # Check if the file is explicitly ignored
                if file in ignored_files:
                    continue
                
                # Check if the file has a binary or non-text extension
                _, ext = os.path.splitext(file)
                if ext.lower() in ignored_extensions:
                    continue
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, target_dir)
                
                try:
                    # 'errors="ignore"' ensures the script won't crash on minor encoding issues
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
                        content = infile.read()
                    
                    # Write clear file markers so it is easy to read or parse
                    outfile.write(f"=========================================\n")
                    outfile.write(f"FILE: {rel_path}\n")
                    outfile.write(f"=========================================\n")
                    outfile.write(content)
                    outfile.write("\n\n")
                    print(f"Added: {rel_path}")
                    
                except Exception as e:
                    print(f"Could not read {rel_path}: {e}")

if __name__ == "__main__":
    combine_files()
    print("\nFile compilation is complete.")