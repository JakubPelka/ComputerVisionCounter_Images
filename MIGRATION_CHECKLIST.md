# Cleanup migration checklist

Use this package as an overlay for the current repository. It intentionally focuses on repository hygiene and documentation. It does not refactor the working Python application code.

## Before copying

1. Make sure the current working code is committed or otherwise safely saved.
2. Check GitHub Desktop / Working Copy for uncommitted changes.
3. Do not delete working `.py` files.

## Copy from this package into the repository root

Copy or replace:

- `README.md`
- `LICENSE`
- `CHANGELOG.md`
- `ROADMAP.md`
- `requirements.txt`
- `.gitignore`
- `.gitattributes`
- `docs/`
- `input/README.md`
- `models/README.md`
- `output/.gitkeep`
- `presets/README.md`
- `sample_data/README.md`
- `tools/README.md`

Optional: replace `start.bat` with the included version after checking that it matches how you currently start the app.

## Move old documentation

Recommended manual move:

- `Quick Start — Computer Vision Counter.docx` -> `docs/legacy/Quick Start — Computer Vision Counter.docx`
- `QuickStart_Computer Vision Counter.pdf` -> `docs/legacy/QuickStart_Computer Vision Counter.pdf`
- `quickstart.md` -> compare with `docs/QUICKSTART.md`; keep only if it contains extra useful details.

## Do not move code yet

Do not move the Python files into `src/` in this cleanup commit. Treat that as a separate refactor after release testing.

## Suggested commit

Commit message:

```text
chore: clean repository structure for v0.1.0 release
```

Commit scope:

- documentation;
- MIT license;
- git hygiene;
- folder placeholders;
- no application logic changes.
