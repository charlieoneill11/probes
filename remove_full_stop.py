import os
import json

# Define the source and destination directories
SOURCE_DIR = "examples"
DEST_DIR = "examples_no_end"

# Ensure the destination directory exists
os.makedirs(DEST_DIR, exist_ok=True)

def remove_trailing_period(text: str) -> str:
    """
    Removes a trailing period ('.') at the very end of the string, if present.
    """
    text = text.rstrip()  # just in case there's trailing whitespace
    if text.endswith('.'):
        return text[:-1]  # remove the last character
    return text

def main():
    # Loop over all files in the SOURCE_DIR
    for filename in os.listdir(SOURCE_DIR):
        if not filename.endswith(".json"):
            continue  # skip non-JSON files
        
        source_path = os.path.join(SOURCE_DIR, filename)
        dest_path = os.path.join(DEST_DIR, filename)
        
        # Load the JSON
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # data should have "examples", possibly "concept", "domain_description", etc.
        # We'll modify the "examples" portion
        if "examples" in data:
            for ex in data["examples"]:
                if "positive" in ex:
                    ex["positive"] = remove_trailing_period(ex["positive"])
                if "negative" in ex:
                    ex["negative"] = remove_trailing_period(ex["negative"])
        
        # Write the updated data to the new folder
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Processed {filename}, saved to {dest_path}")

if __name__ == "__main__":
    main()

