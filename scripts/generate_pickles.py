"""
Generates train/filenames.pickle and test/filenames.pickle from the existing
data/birds/images/ directory.

AttnGAN/StackGAN use a class-based split:
  - Classes 001–150  →  training set  (~8855 images)
  - Classes 151–200  →  test set      (~2933 images)

Each pickle is a plain Python list of path stems:
  '001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111'
(no .jpg extension, no leading data/birds/images/)

NOTE: This script only generates the split pickles.
      Text captions (data/birds/text/) must still be downloaded separately —
      see scripts/download_birds_text.sh or the README.
"""

import os
import pickle

IMAGES_DIR = os.path.join("data", "birds", "images")
DATA_DIR   = os.path.join("data", "birds")


def main():
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(
            f"Images directory not found: {IMAGES_DIR}\n"
            "Run this script from the project root."
        )

    train_fnames = []
    test_fnames  = []

    for class_dir in sorted(os.listdir(IMAGES_DIR)):
        class_path = os.path.join(IMAGES_DIR, class_dir)
        if not os.path.isdir(class_path):
            continue

        # Class number is the first three digits of the folder name
        try:
            class_num = int(class_dir.split(".")[0])
        except ValueError:
            continue

        for img_file in sorted(os.listdir(class_path)):
            if not img_file.lower().endswith(".jpg"):
                continue
            stem = img_file[:-4]  # strip .jpg
            entry = f"{class_dir}/{stem}"

            if class_num <= 150:
                train_fnames.append(entry)
            else:
                test_fnames.append(entry)

    os.makedirs(os.path.join(DATA_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "test"),  exist_ok=True)

    train_path = os.path.join(DATA_DIR, "train", "filenames.pickle")
    test_path  = os.path.join(DATA_DIR, "test",  "filenames.pickle")

    with open(train_path, "wb") as f:
        pickle.dump(train_fnames, f)
    with open(test_path, "wb") as f:
        pickle.dump(test_fnames, f)

    print(f"train/filenames.pickle  → {len(train_fnames)} images")
    print(f"test/filenames.pickle   → {len(test_fnames)} images")
    print(f"\nSaved to {DATA_DIR}/train/ and {DATA_DIR}/test/")
    print("\nReminder: text captions (data/birds/text/) must still be downloaded.")
    print("See: bash scripts/download_birds_text.sh")


if __name__ == "__main__":
    main()
