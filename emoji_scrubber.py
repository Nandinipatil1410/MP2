import os
import re

def scrub_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove non-ASCII characters
    clean_content = re.sub(r'[^\x00-\x7F]+', '', content)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(clean_content)

for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            p = os.path.join(root, file)
            print(f"Scrubbing {p}...")
            scrub_file(p)
