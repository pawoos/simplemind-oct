import os
import ast
import re

TOOLS_DIR = "tools"
README_FILE = "tools/README.md"

def get_module_docstring(file_path):
    """Return the module-level docstring of a Python file."""
    with open(file_path, "r", encoding="utf-8") as f:
        node = ast.parse(f.read())
    return ast.get_docstring(node)

def is_tool_file(file_path):
    """Check if a .py file is eligible (docstring starts with 'Tool Name:')."""
    doc = get_module_docstring(file_path)
    if doc and doc.strip().startswith("Tool Name:"):
        return True
    return False

def slugify_gitlab_header(header_text):
    """
    Convert header text to GitLab Markdown anchor link format.
    GitLab rules:
    - Lowercase
    - Remove punctuation (., /, etc.)
    - Remove spaces
    """
    text = header_text.lower()
    text = re.sub(r"[^\w\- ]+", "", text)
    text = text.replace(" ", "")
    return text

def collect_tools(root_dir):
    """
    Recursively walk root_dir and return a dict:
    { first_level_folder: [list of eligible .py files (with relative path)] }
    Files in the root go under "Root".
    Only includes files whose docstring starts with 'Tool Name:'.
    Folders without eligible files are skipped.
    """
    tools = {}
    for dirpath, _, filenames in os.walk(root_dir):
        py_files = [f for f in filenames if f.endswith(".py")]
        eligible_files = []
        for f in py_files:
            full_path = os.path.join(dirpath, f)
            if is_tool_file(full_path):
                rel_path = os.path.relpath(full_path, root_dir)
                eligible_files.append(rel_path)

        if not eligible_files:
            continue  # Skip folder with no eligible files

        # Determine first-level folder
        rel_dir = os.path.relpath(dirpath, root_dir)
        parts = rel_dir.split(os.sep)
        first_level = parts[0] if parts[0] != "." else "Root"

        if first_level not in tools:
            tools[first_level] = []

        tools[first_level].extend(eligible_files)

    # Sort each folder's tools
    for folder in tools:
        tools[folder].sort()

    return tools

def main():
    tools_dict = collect_tools(TOOLS_DIR)

    lines = ["# Tool Documentation\n"]

    # 1. List of Tools grouped by first-level folder
    #lines.append("## List of Tools\n")
    for folder, files in sorted(tools_dict.items()):
        lines.append(f"### {folder}")
        for filepath in files:
            filename = os.path.basename(filepath)
            anchor = slugify_gitlab_header(filename)
            lines.append(f"- [{filename}](#{anchor})")
        lines.append("")  # blank line after each folder
    lines.append("***") # horizontal line separator
    
    # 2. Docstrings grouped by first-level folder
    for folder, files in sorted(tools_dict.items()):
        lines.append(f"## {folder}")
        lines.append("")  # blank line
        for filepath in files:
            path = os.path.join(TOOLS_DIR, filepath)
            doc = get_module_docstring(path)
            if doc:
                filename = os.path.basename(filepath)
                lines.append(f"### {filename}")
                lines.append("")
                lines.append(f"```\n{doc.strip()}\n```")
                lines.append("")  # blank line between tools

    # Write README.md
    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
