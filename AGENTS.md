# Agent Instructions

For every repository change, use the normal change-management flow:

- Work on a feature branch rather than committing directly to `main`.
- Run the full relevant test and verification suite before committing.
- Commit the completed change with a concise, imperative message.
- Push the branch, open a pull request, and include the verification performed.
- Merge the pull request after the change is ready, then resync local `main`.

If a verification step cannot run in the current environment, document the exact
blocker in the pull request and final handoff.
