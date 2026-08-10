# CI troubleshooting

If CI fails, inspect the first failing quality gate in order: package installation, Python compilation, Ruff, then pytest. Fix the underlying error before re-running the workflow rather than weakening the test suite.
