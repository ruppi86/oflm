import os

def summarize_texts_to_markdown(root_dir='.'):
    output_lines = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
            #if filename.endswith(('.md', '.py')):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root_dir)

                try:
                    with open(full_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        non_empty_lines = [line.rstrip() for line in lines if line.strip()]
                        content = '\n'.join(non_empty_lines)
                except Exception as e:
                    content = f"Error reading file {rel_path}: {e}"

                lang = filename.split('.')[-1]
                output_lines.append(f"# {rel_path}\n\n```{lang}\n{content}\n```\n\n---\n")

    summary_markdown = '\n'.join(output_lines)
    output_file_path = os.path.join(root_dir, 'summarized_files.py')

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(summary_markdown)

    print(f"Summary written to {output_file_path}")
    return output_file_path

def clean_markdown_formatting(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        # Ta bort markdown-tecken: # och *
        cleaned = content.replace('#', '').replace('*', '')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print("Markdown formatting cleaned.")
    except Exception as e:
        print(f"Error cleaning markdown formatting: {e}")

if __name__ == '__main__':
    output_file = summarize_texts_to_markdown()
    clean_markdown_formatting(output_file)
