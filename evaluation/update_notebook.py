import json

filepath = "/Users/vientu/AI2026/Linear Learning/evaluation/linear_learning.ipynb"
with open(filepath, 'r') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb.get("cells", [])):
    if cell["cell_type"] == "markdown":
        source = "".join(cell["source"])
        print(f"--- CELL {idx} ---")
        print(source)
