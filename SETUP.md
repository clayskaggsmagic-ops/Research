# Contributor Setup

This repo is private but not public. Treat credentials accordingly: **each contributor brings their own API keys**; the shared database credential is distributed through a password manager, never via git, Slack, or email.

## 1. Get your own API keys

Each contributor creates their own. Do not share these with anyone, including other contributors on this project.

| Service | Signup | Notes |
|---|---|---|
| Google AI (Gemini) | https://aistudio.google.com/apikey | Free tier covers dev work |
| Tavily (web search) | https://app.tavily.com | Free tier: 1,000 searches/month |
| Anthropic (optional) | https://console.anthropic.com | Only needed for cross-model resolution |

## 2. Get the shared database credential

Ask the project owner (Clay) for the `DATABASE_URL` **through a password manager** (1Password vault, Bitwarden share, etc). Never request it via chat.

Two credentials exist:
- **`neondb_owner`** — full read/write. Only contributors who need to write to the DB.
- **`chronos_readonly`** — read-only. Default for new contributors.

## 3. Create your `.env` files

Copy the examples and fill in the values you got above:

```bash
cp temporal_knowledge_base/.env.example temporal_knowledge_base/.env
cp pipeline/.env.example pipeline/.env
```

Then edit each `.env` with your keys. Both `.env` files are gitignored at the project and repo-root level — they cannot be accidentally committed.

**Format for `DATABASE_URL`** (CHRONOS only):
```
postgresql+asyncpg://USER:PASSWORD@HOST/neondb?ssl=require
```
Note `postgresql+asyncpg://` (not `postgresql://`) and `ssl=require` (not `sslmode=require`).

## 4. Use your own Neon branch (recommended)

If you're going to run the CHRONOS pipeline and write to the DB, ask Clay to create a Neon branch for you (they're cheap — copy-on-write). Work on your branch so you don't collide with anyone else.

```
main                  ← stable
├── dev-clay          ← Clay's workspace
├── dev-alice         ← contributor branches
└── dev-bob
```

Contributors with `chronos_readonly` can query `main` directly — no branch needed.

## 5. If you leak a key

1. Tell Clay immediately.
2. Rotate the key at the provider console.
3. Update your `.env`.

No blame — keys leak. Fast rotation is what matters.

## 6. Don't commit secrets

If you ever see credentials about to be committed:
```bash
git diff --cached | grep -E "(AIzaSy|tvly-|npg_|sk-)"
```
Root `.gitignore` catches `.env`, but defense in depth: check before you push.
