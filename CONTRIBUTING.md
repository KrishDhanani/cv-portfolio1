# Contributing

Thanks for contributing to this group project.

## Branches

Use one branch per task/member:

- `person-1-dataset-analysis`
- `person-2-data-pipeline`
- `person-3-training-evaluation`
- `person-4-visualization`

Create your branch from the latest `main`:

```bash
git checkout main
git pull origin main
git checkout -b person-4-visualization
```

## Commit and push

```bash
git add <files>
git commit -m "Add error analysis visualizations"
git push -u origin person-4-visualization
```

Use a short, descriptive commit message. Do not commit temporary files, virtual environments, caches, or large datasets.

## Pull requests

1. Open a pull request from your task branch into `main`.
2. Explain what changed and how it was tested.
3. Request review from at least one teammate.
4. Resolve review comments before merging.

## Shared-file rules

- Coordinate before changing shared configuration files such as `configs/data.yaml`.
- Do not overwrite another member's work.
- Keep notebooks task-specific; move reusable functions into `src/`.
- Never commit passwords, API keys, private tokens, or personal data.
- Do not commit the dataset or large model weights unless the team has agreed on an appropriate storage method.

## Code and notebook conventions

- Use clear names and add brief docstrings to reusable functions.
- Keep paths configurable rather than hard-coded to one computer.
- Restart and run notebooks before committing, where practical.
- Record random seeds and relevant experiment settings.
