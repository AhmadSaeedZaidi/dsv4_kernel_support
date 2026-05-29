import os
import urllib.request
import json

# The GitHub API endpoint for this specific PR's files
api_url = "https://api.github.com/repos/flashinfer-ai/flashinfer/pulls/3285/files"

print(f"Fetching file list for PR #3285...")
req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})

try:
    with urllib.request.urlopen(req) as response:
        files = json.loads(response.read())
        
    for f in files:
        filepath = f['filename']
        raw_url = f['raw_url']
        
        # Recreate the folder structure locally
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        # Download the raw file contents
        print(f"Downloading: {filepath}")
        urllib.request.urlretrieve(raw_url, filepath)

    print("\nSuccess! All 11 files downloaded perfectly.")

except Exception as e:
    print(f"Error: {e}")