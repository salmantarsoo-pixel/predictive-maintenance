# Contributing to Predictive Maintenance ML

First off, thank you for considering contributing! It's people like you that make this such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps which reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed after following the steps**
* **Explain which behavior you expected to see instead and why**
* **Include screenshots and animated GIFs if possible**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested enhancement**
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior and the expected behavior**

### Pull Requests

* Fill in the required template
* Follow the Python styleguides
* Include appropriate test cases
* Update documentation as needed
* End all files with a newline

## Development Setup

```bash
# Clone the repository
git clone https://github.com/salmantarsoo-pixel/predictive-maintenance.git
cd predictive-maintenance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"
pip install -r requirements.txt

# Run tests
pytest

# Format code
black src/ tests/

# Run linter
flake8 src/ tests/
```

## Styleguides

### Python Code Style

* Use [PEP 8](https://www.python.org/dev/peps/pep-0008/)
* Use [Black](https://black.readthedocs.io/) for code formatting
* Use docstrings for all functions and classes (Google style)
* Use type hints for function arguments and returns

```python
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1
    """
    # Implementation
    return metrics
```

### Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Documentation

* Use Markdown for documentation
* Include docstrings for all public functions and classes
* Add type hints to all functions
* Update README.md if you change functionality

## Pull Request Process

1. Fork the repository and create your branch from `main`
2. If you've added code that should be tested, add tests
3. Ensure the test suite passes (`pytest`)
4. Make sure your code follows the style guidelines
5. Write a clear commit message
6. Push to your fork and submit a pull request

## Areas We Need Help With

- [ ] Adding real-time monitoring dashboard
- [ ] Implementing SHAP for model interpretability
- [ ] Adding support for multiclass failure types
- [ ] Optimizing inference latency
- [ ] Adding more examples and tutorials
- [ ] Improving documentation

## Additional Notes

### Issue and Pull Request Labels

* `bug` - Something isn't working
* `enhancement` - New feature or request
* `documentation` - Improvements or additions to documentation
* `good first issue` - Good for newcomers
* `help wanted` - Extra attention is needed
* `question` - Further information is requested

## Recognition

Contributors will be recognized in:
- README.md contributors section
- GitHub contributors page
- Release notes

Thank you for contributing!
