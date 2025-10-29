# Contributing to HyperAnalyze

Thank you for your interest in contributing to HyperAnalyze! 🎉

## How to Contribute

### Reporting Bugs
1. Check if the bug has already been reported in [Issues](https://github.com/ConejoCapital/hyperanalyze/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots if applicable
   - Your environment (OS, Python version, etc.)

### Suggesting Features
1. Open an issue with the `enhancement` label
2. Describe the feature and its use case
3. Explain why it would be valuable
4. Provide examples if possible

### Code Contributions

#### Setup
```bash
# Fork and clone the repo
git clone https://github.com/YOUR_USERNAME/hyperanalyze.git
cd hyperanalyze

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Making Changes
1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Test thoroughly: `python3 test_installation.py`
4. Commit with clear messages: `git commit -m "Add: description"`
5. Push to your fork: `git push origin feature/your-feature-name`
6. Open a Pull Request

#### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Comment complex logic
- Keep functions focused and small

#### Pull Request Guidelines
- Link related issues
- Describe what changes you made and why
- Include screenshots for UI changes
- Ensure all tests pass
- Update documentation if needed

### Documentation
- Fix typos and improve clarity
- Add examples and use cases
- Update README for new features
- Create tutorials or guides

## Development Workflow

### Testing
```bash
# Run installation test
python3 test_installation.py

# Launch dev server with auto-reload
streamlit run dashboard.py --server.runOnSave=true
```

### Adding New Visualizations
1. Add class to `visualizations.py`
2. Follow existing patterns (Plotly figures)
3. Add to `dashboard.py` with proper layout
4. Update documentation

### Adding New Data Sources
1. Update `data_loader.py`
2. Add preprocessing logic
3. Update schema documentation
4. Add example in README

## Community

- Be respectful and constructive
- Help others in issues and discussions
- Share your use cases and insights
- Spread the word if you find it useful!

## Questions?

Open an issue with the `question` label or reach out through GitHub discussions.

Thank you for making HyperAnalyze better! 🚀

