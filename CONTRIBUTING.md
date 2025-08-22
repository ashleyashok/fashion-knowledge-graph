# Contributing to Complete the Look

Thank you for your interest in contributing to the Complete the Look project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

We welcome contributions from the community! Here are the main ways you can contribute:

### 🐛 Bug Reports

If you find a bug, please create an issue with:

- **Clear title**: Brief description of the issue
- **Detailed description**: What happened, what you expected, and steps to reproduce
- **Environment**: Python version, OS, and relevant package versions
- **Screenshots**: If applicable, include screenshots or error messages

### 💡 Feature Requests

We love new ideas! When suggesting features:

- **Clear use case**: Explain the problem you're trying to solve
- **Proposed solution**: Describe your suggested approach
- **Impact**: How would this benefit users?
- **Implementation**: Any technical considerations or approaches

### 🔧 Code Contributions

#### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/ashleyashok/fashion-knowledge-graph.git
   cd fashion-knowledge-graph
   ```

2. **Set up development environment**
   ```bash
   poetry install
   poetry shell
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

5. **Test your changes**
   ```bash
   poetry run pytest
   poetry run black src/ tests/
   poetry run flake8 src/ tests/
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide a clear description of your changes
   - Reference any related issues
   - Include screenshots if UI changes

## 📋 Coding Standards

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black formatter default)
- **Docstrings**: Use Google style docstrings
- **Type hints**: Use type hints for all function parameters and return values
- **Imports**: Group imports: standard library, third-party, local

### Code Example

```python
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger

from src.database.graph_database import GraphDatabaseHandler


def process_fashion_data(
    data: pd.DataFrame,
    graph_db: GraphDatabaseHandler,
    threshold: float = 0.75,
) -> Dict[str, List[str]]:
    """Process fashion data and create graph relationships.
    
    Args:
        data: Input fashion data DataFrame
        graph_db: Graph database handler instance
        threshold: Similarity threshold for relationships
        
    Returns:
        Dictionary containing processed results
        
    Raises:
        ValueError: If data is empty or invalid
    """
    if data.empty:
        raise ValueError("Input data cannot be empty")
    
    logger.info(f"Processing {len(data)} fashion items")
    
    # Your implementation here
    results = {}
    
    return results
```

### Testing Standards

- **Test coverage**: Aim for at least 80% code coverage
- **Test naming**: Use descriptive test names that explain the scenario
- **Test organization**: Group related tests in classes
- **Mocking**: Use mocks for external dependencies

```python
import pytest
from unittest.mock import Mock, patch
from src.inference.recommender import Recommender


class TestRecommender:
    """Test cases for the Recommender class."""
    
    @pytest.fixture
    def mock_graph_db(self):
        """Create a mock graph database."""
        return Mock()
    
    @pytest.fixture
    def recommender(self, mock_graph_db):
        """Create a Recommender instance with mocked dependencies."""
        return Recommender(
            graph_db=mock_graph_db,
            catalog_csv_path="test_data.csv",
            vector_db_image=Mock(),
            vector_db_style=Mock(),
        )
    
    def test_get_recommendations_returns_valid_structure(self, recommender):
        """Test that get_recommendations returns expected structure."""
        # Arrange
        product_id = "test_product_123"
        
        # Act
        result = recommender.get_recommendations(product_id)
        
        # Assert
        assert isinstance(result, dict)
        assert "selected_product" in result
        assert "worn_with" in result
        assert "complemented" in result
```

## 📚 Documentation Standards

### Code Documentation

- **Functions**: Document all public functions with docstrings
- **Classes**: Document class purpose and key methods
- **Complex logic**: Add inline comments for complex algorithms
- **Examples**: Include usage examples in docstrings

### README Updates

When adding new features:

- Update the README.md with new functionality
- Add appropriate badges and links
- Update the project structure if needed
- Include usage examples

## 🚀 Pull Request Process

### Before Submitting

1. **Self-review**: Review your code for:
   - Code style compliance
   - Test coverage
   - Documentation completeness
   - Performance implications

2. **Local testing**: Ensure all tests pass locally
   ```bash
   poetry run pytest
   poetry run black --check src/ tests/
   poetry run flake8 src/ tests/
   ```

3. **Update documentation**: Add or update relevant documentation

### Pull Request Template

Use this template when creating PRs:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Added tests for new functionality
- [ ] All existing tests pass
- [ ] Manual testing completed

## Documentation
- [ ] Updated README if needed
- [ ] Added docstrings for new functions
- [ ] Updated API documentation if applicable

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] No breaking changes (or documented if necessary)
```

## 🏷️ Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```
feat: add social media trend analysis

fix(recommender): handle empty product catalog gracefully

docs: update installation instructions for Windows

test: add unit tests for graph traversal queries
```

## 🐛 Issue Templates

### Bug Report Template

```markdown
## Bug Description
Clear and concise description of the bug

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- OS: [e.g., macOS, Windows, Linux]
- Python Version: [e.g., 3.11.0]
- Package Versions: [relevant packages]

## Additional Context
Any other context, screenshots, or logs
```

### Feature Request Template

```markdown
## Problem Statement
Clear description of the problem you're trying to solve

## Proposed Solution
Description of your suggested solution

## Alternative Solutions
Any alternative solutions you've considered

## Additional Context
Any other context, use cases, or examples
```

## 📞 Getting Help

If you need help with contributing:

1. **Check existing issues**: Your question might already be answered
2. **Read the documentation**: Check README.md and BLOG_POST.md
3. **Create an issue**: Use the "Question" template for general questions
4. **Contact maintainers**: Reach out to the project maintainers

## 🙏 Recognition

Contributors will be recognized in:

- **README.md**: Listed as contributors
- **Release notes**: Mentioned in relevant releases
- **Project documentation**: Credited for significant contributions

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Complete the Look! 🎉
