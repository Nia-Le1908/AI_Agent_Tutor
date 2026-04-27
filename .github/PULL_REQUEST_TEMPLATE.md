## Summary

Describe what this PR changes and why.

## Scope

- Type of change:
  - [ ] Feature
  - [ ] Bug fix
  - [ ] Refactor
  - [ ] Documentation
  - [ ] Infrastructure/DevOps
  - [ ] Test-only

- Related issue/task:
  - Link issue or write N/A

## Files and Modules Affected

List key files touched and a short note for each.

- file/path.py: reason for change

## Technical Details

### Design decisions

Briefly explain architecture choices, trade-offs, and alternatives considered.

### Data/Schema impact

- [ ] No database changes
- [ ] Database schema changed
- [ ] Data migration required

If changed, describe impact on schema.sql, init_db.py, and compatibility.

### API/Contract impact

- [ ] No cross-module signature changes
- [ ] Updated cross-module signatures

If signatures changed, confirm interfaces.md is updated:
- [ ] interfaces.md updated

## Security and Privacy Checklist

- [ ] No secrets added to committed files
- [ ] .env and sensitive files remain ignored
- [ ] No large binary/model/index/database files committed
- [ ] Input validation added/updated where needed
- [ ] Error messages avoid leaking sensitive values

## Testing Checklist

### Local quality checks

- [ ] Code runs without syntax/type errors
- [ ] Relevant lint/format checks passed
- [ ] No new warnings in edited files

### Functional checks

- [ ] init_db.py runs successfully
- [ ] Streamlit app launches (app.py)
- [ ] If RAG code changed: embedder/retriever smoke-tested
- [ ] If generation code changed: Gemini JSON parsing tested
- [ ] If adaptive logic changed: streak behavior tested (3 correct up, 2 wrong down)

### Regression checks

- [ ] Existing flows still work
- [ ] Backward compatibility considered for affected modules

## RAG Evaluation (Fill if relevant)

- [ ] Not applicable
- [ ] Ran rag_tester.py

If run, include metrics:
- Precision@3:
- MRR:
- Notes:

## UI Verification (Fill if relevant)

- [ ] Not applicable
- [ ] Streamlit rerun behavior validated with session_state
- [ ] Chat history persists across interactions
- [ ] Current difficulty persists and updates correctly
- [ ] Dashboard renders expected charts

## Deployment / Runbook Notes

Any setup, migration, or rollout steps needed after merge.

## Reviewer Guidance

Suggested review order and areas of focus.

1. High-risk files:
2. Test evidence to verify:
3. Open questions:

## Author Self-Check

- [ ] I reviewed my own diff end-to-end
- [ ] I added/updated docs where needed
- [ ] I updated tests or provided rationale when tests were not added
- [ ] This PR is ready for review
