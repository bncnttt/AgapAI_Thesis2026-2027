# Restore instructions for Codex

When asked to restore this project, follow `RESTORE.md` and run
`restore-kit\restore.ps1` step by step. Ask the user before any download or
install expected to exceed about 500 MB, and wait for the answer before doing
that step.

Never edit project source, configuration, or lockfiles under `AgapAI\` as part
of a restore. Never push to any remote. Never contact or touch
`github.com/bncnttt/AgapAI` or the local `.git_group_backup` history. Keep model
weights, MongoDB exports, credentials, virtual environments, caches, and
`node_modules` out of Git.
