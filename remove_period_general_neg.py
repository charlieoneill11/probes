import os
import json
import random

# File paths
SOURCE_FILE = "inputs/general_negative_examples.json"
DEST_FILE = "inputs/general_negative_examples_no_period.json"
CHANCE_TO_REMOVE_PERIOD = 0.8  # 80% chance to remove the trailing period

def remove_trailing_period(text: str) -> str:
    """
    Removes a trailing period ('.') at the very end of the string, if present.
    """
    text = text.rstrip()  # Remove trailing whitespace so that . is last
    if text.endswith('.'):
        return text[:-1]
    return text

def main():
    # Load source file
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Assuming the JSON has a top-level "examples" key that is a list of strings
    if "examples" in data:
        new_examples = []
        for ex in data["examples"]:
            # Remove trailing period based on probability check - remove 80%
            if random.random() < CHANCE_TO_REMOVE_PERIOD:
                ex = remove_trailing_period(ex)
            new_examples.append(ex)
        data["examples"] = new_examples

    # Write the updated data to the destination file
    with open(DEST_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {SOURCE_FILE}, saved to {DEST_FILE}")

if __name__ == "__main__":
    main()
