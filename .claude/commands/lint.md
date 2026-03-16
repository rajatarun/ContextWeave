# Lint and type-check all Python source files

Run code quality checks across all Lambda source modules.

**Check for ruff (fast linter)**:
```bash
ruff check src/ --fix
```

**If ruff is not installed, fall back to flake8 or pylint**:
```bash
python -m flake8 src/ --max-line-length=120 --ignore=E501,W503
```

**Type checking with mypy**:
```bash
python -m mypy src/ --ignore-missing-imports --python-version 3.12
```

Check each of these modules for issues:
- `src/preprocessor/` — handler, extractors, graph_builder, routing_analyzer, models
- `src/query_api/` — handler, rag_router, retriever, graph_expander, synthesizer, models
- `src/ingestion_trigger/` — handler
- `src/shared/` — models

Report all issues found, grouped by file. For any errors, read the relevant file and suggest fixes that maintain the existing architecture and AWS Lambda constraints (Python 3.12, no external dependencies beyond boto3 and PyYAML).
