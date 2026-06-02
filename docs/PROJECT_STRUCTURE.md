# Project structure

This cleanup uses a conservative structure because the application already works and the current code should not be refactored before release testing.

## Current release-candidate structure

```text
README.md                 public project description
LICENSE                   MIT license
CHANGELOG.md              change history
ROADMAP.md                planned development
requirements.txt          dependency reference
.gitignore                blocks generated/private/heavy files
.gitattributes            line-ending and binary rules
start.bat                 Windows launcher
project_paths.py          central project paths and local package helpers
*.py                      current working application code
docs/                     documentation
docs/legacy/              older documents kept for reference
docs/planning/            migration and roadmap notes
presets/                  tracked example presets
input/                    local input data, ignored except README
models/                   local model files, ignored except README
output/                   generated outputs, ignored
sample_data/              documented place for tiny non-private samples
tests/                    lightweight behavior-protection tests
tools/                    future helper scripts
```

## Documentation structure

Active user-facing and release-candidate docs live directly under `docs/`.
Historical files and old generated quick-start documents live under `docs/legacy/`.
Planning notes and migration checklists live under `docs/planning/`.

## Why code is not moved to `src/` yet

Moving the Python files into `src/` is a real refactor, not only a folder cleanup. The current code used paths relative to the Python file location in several places. Moving files without checking path handling can change where the app looks for `input/`, `models/`, `output/` and local package folders.

Path handling has started moving into `project_paths.py`. Recommendation: keep strengthening that boundary and tests first, then solve the `src/` refactor as a separate issue with startup/output smoke testing.
