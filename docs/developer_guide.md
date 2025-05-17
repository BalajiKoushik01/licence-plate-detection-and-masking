# Developer Guide

This guide provides information for developers who want to contribute to the Vehicle Detection System.

## Development Environment Setup

1. Clone the repository
   ```bash
   git clone https://github.com/username/vehicle_detection.git
   cd vehicle_detection
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies
   ```bash
   pip install -r requirements.txt
   pip install -e .
   pip install pytest pytest-cov flake8 black
   ```

## Project Structure

The project is organized as follows:

- `vehicle_detection/`: Main package directory
  - `__init__.py`: Package initialization
  - `detector.py`: Core detection functionality
  - `main.py`: Command-line interface
- `tests/`: Unit tests
- `docs/`: Documentation
- `examples/`: Example scripts
- `inputs/`: Sample input images
- `output/`: Output directory for processed images

## Coding Standards

### Style Guide

This project follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide. You can use tools like `flake8` and `black` to ensure your code adheres to these standards:

```bash
# Check code style
flake8 vehicle_detection

# Format code
black vehicle_detection
```

### Docstrings

All functions, classes, and modules should have docstrings following the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings):

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of the function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value
    """
    # Function implementation
```

### Type Hints

Use type hints for all function parameters and return values:

```python
from typing import List, Dict, Tuple, Optional

def process_data(data: List[str], options: Optional[Dict[str, int]] = None) -> Tuple[int, str]:
    # Function implementation
```

## Testing

### Running Tests

Run the tests using pytest:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=vehicle_detection
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with the prefix `test_`
- Name test functions with the prefix `test_`
- Use descriptive test names that explain what is being tested
- Use pytest fixtures for setup and teardown
- Mock external dependencies

Example:

```python
import pytest
from unittest.mock import patch, MagicMock

from vehicle_detection.detector import detect_vehicles

def test_detect_vehicles_returns_empty_list_when_no_vehicles():
    # Test implementation
```

## Pull Request Process

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Run tests to ensure they pass
5. Update documentation if necessary
6. Submit a pull request

## Release Process

1. Update version number in `vehicle_detection/__init__.py`
2. Update CHANGELOG.md
3. Create a new release on GitHub
4. Build and upload the package to PyPI:
   ```bash
   python setup.py sdist bdist_wheel
   twine upload dist/*
   ```

## Troubleshooting

### Common Development Issues

1. **Import errors**: Ensure you've installed the package in development mode with `pip install -e .`

2. **Test failures**: Check that you haven't broken existing functionality. Run `pytest -v` for more detailed output.

3. **Model loading issues**: Ensure the YOLOv8 model file is available and correctly referenced.

## Additional Resources

- [YOLOv8 Documentation](https://github.com/ultralytics/ultralytics)
- [OpenCV Documentation](https://docs.opencv.org/4.x/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [pytest Documentation](https://docs.pytest.org/)