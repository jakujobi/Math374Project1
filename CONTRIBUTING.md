# Contributing to Numerical Differentiation Error Analysis

Thank you for considering contributing to this project! This document provides guidelines for contributing.

## Project Status

**Note:** This is an academic project completed for Math 374 - Scientific Computing (Spring 2024). While the project is feature-complete for its original academic purpose, contributions that improve code quality, fix bugs, or enhance documentation are welcome.

## Ways to Contribute

### Bug Reports

If you find a bug, please open an issue with:
- **Title:** Clear, descriptive summary
- **Description:** What you expected vs. what happened
- **Reproduction Steps:** Minimal steps to reproduce the issue
- **Environment:** Python version, OS, browser (for UI issues)
- **Screenshots:** If applicable

**Example:**
```
Title: Error plot not rendering for h < 1e-15

Description:
When setting h_max to values beyond 15, the error plot fails to render.
Expected: Plot should display even for very small h values.
Actual: Plot area is blank.

Steps to Reproduce:
1. Launch app
2. Set h_max slider to 16
3. Observe empty plot area

Environment:
- Python 3.12.3
- Streamlit 1.54.0
- Chrome 120.0
```

### Feature Requests

For new features, open an issue with:
- **Use case:** Why is this feature needed?
- **Proposed solution:** How would it work?
- **Alternatives considered:** What other approaches did you think about?

**Note:** Since this is an academic project, feature requests will be evaluated based on educational value and alignment with numerical analysis pedagogy.

### Documentation Improvements

Documentation contributions are highly valued:
- Fix typos or unclear explanations
- Add examples or diagrams
- Improve code comments
- Translate documentation (if desired)

### Code Contributions

#### Before You Start

1. **Check existing issues:** Avoid duplicate work
2. **Open an issue first:** Discuss your proposed changes
3. **Small PRs preferred:** Easier to review and merge

#### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/Math374Project1.git
cd Math374Project1

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```

#### Code Standards

While this project doesn't enforce strict style guidelines, please follow these conventions:

**Python Style:**
- Follow PEP 8 generally
- Use descriptive variable names
- Add docstrings for functions
- Keep functions focused (single responsibility)

**Good Example:**
```python
def calculate_forward_difference(x: float, h: float) -> float:
    """
    Calculate forward difference approximation of derivative.
    
    Args:
        x: Point at which to evaluate derivative
        h: Step size
        
    Returns:
        Approximate derivative value
    """
    return (np.sin(x + h) - np.sin(x)) / h
```

**Comments:**
```python
# Good: Explains why
# Use cached results to avoid recomputation on parameter changes
@st.cache_data
def calculate_errors(...):
    ...

# Avoid: States the obvious
# Calculate the error
error = abs(approx - exact)
```

**Streamlit Patterns:**
```python
# Good: Clear structure
with st.sidebar:
    st.header("Controls")
    h_min = st.slider("Minimum h", 1, 16, 1)

# Avoid: Scattered UI elements
h_min = st.slider("Minimum h", 1, 16, 1)  # In main content
st.sidebar.header("Controls")  # Header appears after slider
```

#### Testing

Currently, no automated tests exist. When adding new features:

1. **Manual Testing:**
   - Test all interactive controls
   - Verify plots render correctly
   - Check console for errors

2. **Boundary Cases:**
   - Test extreme h values (very small, very large)
   - Test edge cases (single point, maximum points)
   - Verify numerical stability

3. **Documentation:**
   - Update README if user-facing changes
   - Update DEVELOPMENT.md if internal changes
   - Add inline comments for complex logic

#### Pull Request Process

1. **Branch Naming:**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Commit Messages:**
   ```
   # Good
   Add error ratio plot for method comparison
   Fix division by zero in rounding error calculation
   Update README with deployment instructions
   
   # Avoid
   Update code
   Fix bug
   Changes
   ```

3. **PR Description Template:**
   ```markdown
   ## Summary
   Brief description of changes
   
   ## Motivation
   Why are these changes needed?
   
   ## Changes Made
   - [ ] Updated X to Y
   - [ ] Added Z feature
   - [ ] Fixed bug in W
   
   ## Testing
   How did you test these changes?
   
   ## Screenshots (if applicable)
   Before/after images for UI changes
   
   ## Checklist
   - [ ] Code follows project style
   - [ ] Documentation updated
   - [ ] Manually tested
   - [ ] No new warnings/errors
   ```

4. **Review Process:**
   - Maintainer will review within 1 week
   - Address feedback promptly
   - Be open to suggestions

5. **Merge:**
   - PRs merged via squash commit
   - Maintainer merges after approval

## Types of Contributions We're Looking For

### High Priority
- **Bug fixes:** Correctness is critical for educational tools
- **Performance improvements:** Caching, optimization
- **Documentation:** Clearer explanations, more examples
- **Accessibility:** Screen reader support, keyboard navigation

### Medium Priority
- **New visualizations:** Additional plot types (3D, interactive)
- **Extended analysis:** More differentiation methods
- **Export features:** Save plots, export data to CSV
- **UI improvements:** Better layout, theming

### Low Priority (Academic Scope)
- **New functions:** Beyond sin(x) - would require significant refactoring
- **Machine learning:** Out of scope for numerical analysis focus
- **Database integration:** Not needed for this use case

## What We're NOT Looking For

- **Rewrites in other languages:** Project is Python/Streamlit
- **Framework changes:** Streamlit is core to the project
- **Breaking changes:** Must maintain backward compatibility
- **Unrelated features:** Must align with numerical differentiation focus

## Code of Conduct

### Our Standards

**Positive Behavior:**
- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the project
- Show empathy toward others

**Unacceptable Behavior:**
- Harassment or discriminatory language
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information
- Unprofessional conduct

### Enforcement

Violations will result in:
1. **Warning:** First offense, informal notice
2. **Temporary ban:** Repeated minor offenses
3. **Permanent ban:** Severe or repeated violations

Report issues to: [Maintainer - see GitHub profile]

## Attribution

By contributing, you agree that your contributions will be licensed under the same [Hippocratic License 3.0](LICENSE.md) as the project.

## Questions?

- **General questions:** Open a discussion on GitHub
- **Bug reports:** Open an issue
- **Security concerns:** See [SECURITY.md](SECURITY.md)

## Recognition

Contributors will be acknowledged in:
- GitHub contributors list (automatic)
- Release notes (for significant contributions)

Thank you for helping improve this educational resource!
