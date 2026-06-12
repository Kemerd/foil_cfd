# notes/ — integration notes convention

This directory is the cross-module coordination channel for FoilCFD development.

## The rule

File ownership in this repository is strict: each module agent creates and edits only the
files assigned to it. If, while implementing your module, you discover that you need a
change to an interface you do **not** own (a new parameter, a different return type, an
additional accessor, a new struct field), you must NOT edit the foreign header. Instead:

1. Create or append to `notes/integration_<yourmodule>.md`
   (e.g. `notes/integration_sim.md`, `notes/integration_render.md`).
2. Write the **exact proposed signature** — full C++ declaration, including namespace,
   const-ness, and Doxygen `@brief` — plus one short paragraph explaining why the change
   is needed and which call sites depend on it.
3. Code your module **against your proposal** as if it had already been accepted.

The integration agent reads every `integration_*.md`, reconciles conflicting proposals,
applies the accepted signatures to the owning headers, and deletes resolved entries.

## Format for each entry

```markdown
## <header path> : <symbol>
Proposed:
    <exact C++ declaration>
Reason: <1-3 sentences>
Status: PROPOSED   (integration agent flips to ACCEPTED / REJECTED / SUPERSEDED)
```

Keep entries append-only; never rewrite another module's proposals.
