# Installation and recovery notes

**The current repository requires source-code repair before it can be launched reliably.** The entry point `vehicle_detection/main.py` contains unresolved Git merge-conflict markers, so the Python interpreter cannot import it as-is.

## Before attempting installation

1. Inspect `vehicle_detection/main.py` and resolve each conflict against the functionality that should be retained.
2. Search the tracked `.py` files for `<<<<<<<`, `=======` and `>>>>>>>` conflict markers.
3. Confirm the dependencies used by the chosen entry point and create a virtual environment.
4. Compile the package and run a small synthetic-image test before attempting real video.

For a **source inspection environment only**, clone the repo and install a chosen set of dependencies. Do not treat this as a verified executable installation:

```bash
git clone https://github.com/BalajiKoushik01/licence-plate-detection-and-masking.git
cd licence-plate-detection-and-masking
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
```

The previous installation guide referred to a generic third-party package and an unrelated GitHub placeholder. Those instructions were not applicable to this repository.

## Privacy and safety

Only process images and vehicle plate information with permission. Detection and masking accuracy must be assessed with representative data; never assume the system masks every plate in an image or video.
