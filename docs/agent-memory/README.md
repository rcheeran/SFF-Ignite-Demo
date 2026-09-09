# Claude Code memory — snapshot

These are the memory files Claude Code keeps for this machine. They live outside any repo, at

    ~/.claude/projects/-home-jetson/memory/

and are loaded automatically at the start of every session on that machine. They are copied here so a
**reimage does not lose them** — see `../RESTORE.md` for how to put them back.

They hold things that are true about the *work* rather than the code: what was tried and failed, why a
constant has the value it does, and where the expensive-to-rediscover facts are. Several of them cover
projects other than this one (Asimov/FR3, GR00T, the Thor CAN kernel module, the openpi V1 stack) and are
kept because that context is equally lost on a reimage.

They are a SNAPSHOT, not the live copy: editing these files changes nothing until they are copied back.
Re-snapshot with

    cp ~/.claude/projects/-home-jetson/memory/*.md docs/agent-memory/

Verified free of credentials before committing. Keep it that way — these files are in a git repo now.
