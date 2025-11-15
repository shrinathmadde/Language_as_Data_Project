import requests
import os
from pathlib import Path

def download_ud_treebank(repo_name, lang_code, save_dir="data/ud"):
    """
    Download UD treebank files from GitHub
    
    repo_name: e.g., 'UD_Finnish-TDT', 'UD_Hindi-HDTB'
    lang_code: e.g., 'fi_tdt', 'hi_hdtb'
    """
    base_url = f"https://raw.githubusercontent.com/UniversalDependencies/{repo_name}/master/"
    
    # Create directory
    treebank_dir = Path(save_dir) / repo_name
    treebank_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to download (correct naming format)
    files = [
        f"{lang_code}-ud-train.conllu",
        f"{lang_code}-ud-dev.conllu",
        f"{lang_code}-ud-test.conllu"
    ]
    
    downloaded_files = []
    
    for filename in files:
        url = base_url + filename
        save_path = treebank_dir / filename
        
        print(f"Downloading {filename}...")
        print(f"  URL: {url}")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            # Check file size
            file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
            print(f"  ✓ Saved to {save_path} ({file_size:.2f} MB)")
            downloaded_files.append(save_path)
            
        except requests.exceptions.HTTPError as e:
            print(f"  ✗ HTTP Error: {e}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    return treebank_dir, downloaded_files

# Download Finnish TDT
print("=" * 60)
# print("Downloading German GSD Treebank")
print("Downloading Finnish TDT Treebank")
print("=" * 60)
finnish_dir, finnish_files = download_ud_treebank("UD_Finnish-TDT", "fi_tdt")

print("\n" + "=" * 60)
print("Downloading Hindi HDTB Treebank")
print("=" * 60)
hindi_dir, hindi_files = download_ud_treebank("UD_Hindi-HDTB", "hi_hdtb")

# Verify downloads
print("\n" + "=" * 60)
print("Download Summary")
print("=" * 60)
print(f"Finnish files: {len(finnish_files)}/3")
print(f"Hindi files: {len(hindi_files)}/3")

# Count sentences if files were downloaded
def count_sentences_in_conllu(filepath):
    """Count sentences in a CoNLL-U file"""
    if not os.path.exists(filepath):
        return 0
    
    sentence_count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('# sent_id'):
                sentence_count += 1
    return sentence_count

if finnish_files:
    print("\nFinnish TDT sentence counts:")
    for filepath in finnish_files:
        count = count_sentences_in_conllu(filepath)
        print(f"  {filepath.name}: {count} sentences")

if hindi_files:
    print("\nHindi HDTB sentence counts:")
    for filepath in hindi_files:
        count = count_sentences_in_conllu(filepath)
        print(f"  {filepath.name}: {count} sentences")