# Contributing

This repository is mainly used as a personal log to keep track of the work done in the project.

The goal is not to enforce a strict collaborative workflow, but to make the Git history useful and readable.

## Purpose of this repository

This repository is used to organize and record:

- writing progress
- corrections
- refactors
- structural changes
- important notes
- partial progress
- pending ideas

## Commit convention

Commits should be short but descriptive. The idea is that the commit history should make it easy to remember what was done.

Suggested format:

```bash
type: short description of the change
```

Examples:

```bash
docs: add preliminary introduction
docs: improve BQM explanation
refactor: reorganize chapter structure
refactor: separate definitions into their own section
fix: correct equation notation
chore: update gitignore
notes: add pending ideas for later review
```

## Commit types

### `docs`

Use this for changes related to writing, explanations, references, or written content.

Examples:

```bash
docs: add references about Ocean SDK
docs: improve explanation of the BQM model
docs: revise background section wording
```

### `refactor`

Use this for reorganizing content without necessarily changing the main ideas.

Examples:

```bash
refactor: move QUBO explanation to the previous section
refactor: split introduction into subsections
refactor: restructure mathematical derivation
```

### `fix`

Use this for specific corrections.

Examples:

```bash
fix: correct inconsistent notation
fix: fix sign error in the Hamiltonian
fix: correct wrong citation
```

### `chore`

Use this for repository maintenance tasks.

Examples:

```bash
chore: add .gitignore
chore: reorganize folders
chore: remove temporary files
```

### `notes`

Use this to save ideas, reminders, doubts, or pending work.

Examples:

```bash
notes: add pending questions about references
notes: write down doubts about the results section
notes: save ideas for the discussion section
```

## Suggested workflow

Before working:

```bash
git status
```

After making changes:

```bash
git add .
git commit -m "type: short description of the change"
```

To check the commit history:

```bash
git log --oneline
```

## Recommendations

Make small and frequent commits.

It is better to have several clear commits than one very large commit where it is difficult to remember what changed.

Good example:

```bash
docs: add initial BQM explanation
docs: add references to D-Wave and Ocean SDK
refactor: reorganize binary model section
fix: correct binary variable notation
notes: add pending tasks about QCobalt
```

## Pending tasks

Use this section to keep track of things that still need to be reviewed:

- [ ] Review references
- [ ] Improve wording
- [ ] Check notation
- [ ] Verify equations
- [ ] Add additional comments or explanations
