# Vehicle detection and licence-plate processing — code under repair

This repository contains experimental Python modules for vehicle detection, licence-plate processing, image/video workflows and a GUI.

**Current status: not known to be runnable.** During repository review, `vehicle_detection/main.py` still contained unresolved Git conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`). Python cannot import that file until the conflict is resolved and the resulting code is tested. This README deliberately does not offer a misleading one-command launch instruction.

## Areas of the codebase

- `vehicle_detection/detector.py` — YOLO-based vehicle detection.
- `vehicle_detection/plate_recognizer.py` — plate-recognition related logic.
- `vehicle_detection/video_processor.py` — video-processing routines.
- `vehicle_detection/gui/` — GUI-related modules.
- `utils/` — additional helpers.
- `docs/` — earlier setup and usage material, some of which refers to an unverified PyPI package.

## Recovery checklist

1. Resolve the conflict in `vehicle_detection/main.py` **against both sides of the intended functionality**, not by blindly retaining one side.
2. Inspect other tracked Python files for conflict markers and stale imports.
3. Create a virtual environment, install the dependencies actually used by the selected entry point and run syntax checks.
4. Add a minimal smoke test for the CLI and single-image processing before updating public installation instructions.
5. Remove or ignore generated `__pycache__`, output images and logs where appropriate.

The code and models have not been verified for recognition accuracy, privacy or production use. Use only images and videos you have permission to process.
