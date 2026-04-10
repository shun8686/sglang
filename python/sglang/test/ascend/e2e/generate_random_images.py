import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def generate_random_images(jsonl_path, size=(1024, 1024)):
    if not os.path.exists(jsonl_path):
        print("jsonl file does not exist")
        return

    with open(jsonl_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    print(f"begin to generate, total {total_lines}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines):
            try:
                data = json.loads(line.strip())
                image_paths = data.get("path")

                for image_path in image_paths:
                    if not image_path:
                        continue

                    dir_name = os.path.dirname(image_path)
                    if dir_name and not os.path.exists(dir_name):
                        os.makedirs(dir_name, exist_ok=True)

                    random_array = np.random.randint(
                        0, 256, (size[1], size[0], 3), dtype=np.uint8
                    )

                    img = Image.fromarray(random_array)
                    img.save(image_path, quality=95)

            except Exception as e:
                print(e)

    print("Finished")
