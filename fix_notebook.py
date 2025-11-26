import nbformat as nbf
from pathlib import Path

# change this to your notebook name
in_path = Path("full_assignment.ipynb")
out_path = Path("full_assignment_fixed.ipynb")

nb = nbf.read(in_path, as_version=4)

if "widgets" in nb.metadata:
    print("Found metadata.widgets – removing it.")
    del nb.metadata["widgets"]
else:
    print("No metadata.widgets found; nothing to remove.")

nbf.write(nb, out_path)
print(f"Written cleaned notebook to: {out_path}")
