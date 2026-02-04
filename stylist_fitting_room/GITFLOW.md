## Quick Reference

```bash
# Start new feature
git checkout main && git pull origin main
git checkout -b feature/your-feature-name

# Daily workflow
git add -A && git commit -m "feat(scope): description"
git push origin feature/your-feature-name

# Create PR: feature/* → dev → main
```

---

## Branch Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                         BRANCH STRATEGY                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   main ──────────────────────────────────────────────────► main  │
│     │                                               ▲            │
│     │ checkout                                      │ PR+merge   │
│     ▼                                               │            │
│   feature/* ────► PR ────► dev ─────────────────────┘            │
│   bugfix/*       (1 review)  (staging/integration)               │
│                                                                  │
│   hotfix/* ──────────────────────────────────────────► main      │
│              (emergency only, bypass dev)                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Branch Types

| Branch        | Purpose                    | Protected | Merge Target |
| ------------- | -------------------------- | --------- | ------------ |
| `main`      | Production-ready code      | Yes       | -            |
| `dev`       | Integration & staging      | Yes       | main         |
| `feature/*` | New features               | No        | dev          |
| `bugfix/*`  | Bug fixes                  | No        | dev          |
| `hotfix/*`  | Emergency production fixes | No        | main         |

---

## Branch Naming Convention

```
<type>/<short-description>
```

### Examples

| Type    | Example                                 | Use Case                        |
| ------- | --------------------------------------- | ------------------------------- |
| feature | `feature/gemini-retry-logic`          | New Gemini API retry mechanism  |
| feature | `feature/outfit-recommendation-cache` | Add caching for recommendations |
| bugfix  | `bugfix/image-validation-error`       | Fix image validation bug        |
| bugfix  | `bugfix/vto-timeout-handling`         | Fix VTO service timeout         |
| hotfix  | `hotfix/api-key-exposure`             | Emergency security fix          |
| hotfix  | `hotfix/production-crash`             | Critical production issue       |

### Rules

- Use lowercase letters and hyphens only
- Keep names short but descriptive (3-5 words max)
- Include ticket number if applicable: `feature/STYLE-123-user-auth`

---

## Complete Workflow

### 1. Starting a New Feature

```bash
# Ensure you're on the latest main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name

# Verify branch
git branch
# * feature/your-feature-name
```

### 2. Development Cycle

```bash
# Make changes and stage them
git add -A

# Commit with conventional message
git commit -m "feat(gemini): add retry logic for API calls"

# Push to remote
git push origin feature/your-feature-name

# For subsequent pushes
git push
```

### 3. Creating a Pull Request

```bash
# Push final changes
git push origin feature/your-feature-name

# Create PR via GitHub CLI (optional)
gh pr create --base dev --title "feat(gemini): add retry logic" --body "Description here"
```

**PR Target:**

- `feature/*` → `dev`
- `bugfix/*` → `dev`
- `hotfix/*` → `main`

### 4. Code Review Process

1. Request review from 1 team member
2. Address feedback with new commits
3. Get approval (1 required)
4. Merge via GitHub (squash or merge commit)

### 5. Merging to Main

After features are tested in `dev`:

```bash
# Create PR from dev to main
gh pr create --base main --head dev --title "Release: feature descriptions"
```

---

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/).

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type         | Description                 | Example                               |
| ------------ | --------------------------- | ------------------------------------- |
| `feat`     | New feature                 | `feat(vto): add clothing overlay`   |
| `fix`      | Bug fix                     | `fix(search): handle empty results` |
| `docs`     | Documentation               | `docs(readme): update setup guide`  |
| `style`    | Formatting (no code change) | `style(app): fix indentation`       |
| `refactor` | Code restructure            | `refactor(gemini): extract client`  |
| `test`     | Add/update tests            | `test(outfit): add unit tests`      |
| `chore`    | Maintenance                 | `chore(deps): update gradio`        |
| `perf`     | Performance                 | `perf(vto): optimize image loading` |

### Scopes (Project-Specific)

| Scope      | Area                        |
| ---------- | --------------------------- |
| `gemini` | Gemini AI service           |
| `vto`    | Virtual try-on pipeline     |
| `search` | Image search functionality  |
| `app`    | Main Gradio application     |
| `config` | Configuration & environment |
| `deps`   | Dependencies                |

### Examples

```bash
# Feature
git commit -m "feat(gemini): implement structured output parsing"

# Bug fix
git commit -m "fix(vto): resolve image dimension mismatch"

# Breaking change
git commit -m "feat(api)!: change response format

BREAKING CHANGE: Response now returns array instead of object"
```

---

## Pull Request Guidelines

### PR Title Format

Same as commit message: `type(scope): description`

### PR Template

When creating a PR, include:

```markdown
## Summary
Brief description of changes.

## Changes
- Added X
- Fixed Y
- Updated Z

## Testing
- [ ] Unit tests pass
- [ ] Manual testing completed
- [ ] No regressions found

## Screenshots (if applicable)
[Add screenshots for UI changes]

## Related Issues
Closes #123
```

### Review Checklist

Reviewers should verify:

- [ ] Code follows project style
- [ ] No hardcoded secrets or credentials
- [ ] Error handling is appropriate
- [ ] Changes are tested
- [ ] No unnecessary dependencies added
- [ ] Documentation updated if needed

### Review Requirements

| Target Branch | Reviewers Required | Additional Rules    |
| ------------- | ------------------ | ------------------- |
| `dev`       | 1                  | -                   |
| `main`      | 1                  | All tests must pass |

---

## Conflict Resolution

### When Conflicts Occur

```bash
# Update your branch with latest dev
git checkout feature/your-branch
git fetch origin
git rebase origin/dev

# If conflicts appear:
# 1. Open conflicted files
# 2. Look for conflict markers:
#    <<<<<<< HEAD
#    your changes
#    =======
#    incoming changes
#    >>>>>>> origin/dev

# 3. Resolve conflicts manually
# 4. Stage resolved files
git add <resolved-file>

# 5. Continue rebase
git rebase --continue

# 6. Force push (required after rebase)
git push --force-with-lease
```

### Rebase vs Merge

| Situation                              | Use                       |
| -------------------------------------- | ------------------------- |
| Updating feature branch with dev       | `git rebase origin/dev` |
| Merging PR (via GitHub)                | Squash and merge          |
| Long-running feature with many commits | Rebase before PR          |

### Communication Protocol

1. **Before resolving:** Notify in team chat if conflicts involve others' code
2. **Complex conflicts:** Request pair session with original author
3. **After resolving:** Comment on PR about resolution approach

---

## Release Process

### Version Format

We use [Semantic Versioning](https://semver.org/): `vMAJOR.MINOR.PATCH`

| Change Type      | Version Bump | Example          |
| ---------------- | ------------ | ---------------- |
| Breaking changes | MAJOR        | v1.0.0 → v2.0.0 |
| New features     | MINOR        | v1.0.0 → v1.1.0 |
| Bug fixes        | PATCH        | v1.0.0 → v1.0.1 |

### Release Steps

```bash
# 1. Ensure dev is stable and tested
git checkout dev
git pull origin dev

# 2. Create PR from dev to main
gh pr create --base main --head dev --title "Release v1.2.0"

# 3. After merge, tag the release
git checkout main
git pull origin main
git tag -a v1.2.0 -m "Release v1.2.0: Feature descriptions"
git push origin v1.2.0

# 4. Create GitHub release (optional)
gh release create v1.2.0 --title "v1.2.0" --notes "Release notes here"
```

### Release Checklist

- [ ] All PRs merged to dev
- [ ] Integration tests pass on dev
- [ ] No critical bugs open
- [ ] Version number updated in code (if applicable)
- [ ] Changelog updated
- [ ] Team notified of release

---

## Hotfix Procedure

### When to Use Hotfix

- Critical production bug
- Security vulnerability
- Data corruption risk
- Service outage

### Hotfix Workflow

```bash
# 1. Create hotfix from main
git checkout main
git pull origin main
git checkout -b hotfix/critical-issue-name

# 2. Make minimal fix
git add -A
git commit -m "fix(scope): emergency fix description"

# 3. Push and create PR directly to main
git push origin hotfix/critical-issue-name
gh pr create --base main --title "HOTFIX: critical issue description"

# 4. Fast-track review (still requires 1 approval)
# 5. After merge, immediately backport to dev
git checkout dev
git pull origin dev
git merge main
git push origin dev
```

### Hotfix Rules

1. **Minimal changes only** - Fix the issue, nothing else
2. **Fast-track review** - Reviewer should prioritize
3. **Immediate backport** - Sync fix to dev after main merge
4. **Post-mortem required** - Document what happened and why

### Post-Hotfix Checklist

- [ ] Fix deployed and verified in production
- [ ] Fix backported to dev branch
- [ ] Incident documented
- [ ] Root cause identified
- [ ] Prevention measures discussed
