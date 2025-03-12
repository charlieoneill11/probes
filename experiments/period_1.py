import os
import json
import random

SOURCE_DIR = "../examples"
DEST_DIR = "../experiment_examples/period_1"
CHANCE_TO_REMOVE_PERIOD = 1  # default: 80%

def remove_trailing_period(text: str) -> str:
    """
    Removes a trailing period ('.') at the very end of the string, if present.
    """
    text = text.rstrip()  # remove trailing whitespace so that . is last
    if text.endswith('.'):
        return text[:-1]  # remove the last character
    return text

def main():
    # Ensure the destination directory exists
    os.makedirs(DEST_DIR, exist_ok=True)
    
    # Loop over all JSON files in the SOURCE_DIR
    for filename in os.listdir(SOURCE_DIR):
        if not filename.endswith(".json"):
            continue  # skip non-JSON files
        
        source_path = os.path.join(SOURCE_DIR, filename)
        dest_path = os.path.join(DEST_DIR, filename)
        
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # data should have "examples", "concept", "domain_description", etc.
        if "examples" in data:
            for ex in data["examples"]:
                # For "positive" example text
                if "positive" in ex:
                    # Only remove the period if random check passes
                    if random.random() < CHANCE_TO_REMOVE_PERIOD:
                        ex["positive"] = remove_trailing_period(ex["positive"])
                
                # For "negative" example text
                if "negative" in ex:
                    # Only remove the period if random check passes
                    if random.random() < CHANCE_TO_REMOVE_PERIOD:
                        ex["negative"] = remove_trailing_period(ex["negative"])
        
        # Save the updated JSON into DEST_DIR
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Processed {filename}, saved to {dest_path}")

if __name__ == "__main__":
    main()
