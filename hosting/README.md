# Hosting platform

Control plane that lets users sign in with Telegram, connect one shared secretary bot under Telegram Settings > Chat Automation, and runs each user's secretary as a per-tenant pm2 process (`tgs-<id>`). The control plane is the only process polling the shared bot; it forwards each update to the owning tenant over loopback. Package name is `hosting`, never `platform` (shadows the stdlib).

The server must have direct egress to Telegram, the configured model API, and certificate issuance services; check the provider's regional availability before deploying.

## Setup

1. **Platform bot** in @BotFather: create a dedicated bot (never one another process already polls), then in its settings turn on **Secretary Mode**. Put its token and username in `hosting.env`.
2. **Login**: in BotFather switch the bot to OpenID login, copy the client id and client secret, and add the allowed redirect URL `https://<domain>/auth/callback`. Without a client id, `/login` falls back to the classic Login Widget (run `/setdomain` for the bot).
3. **Model provider**: set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY` for Google's official API (`GEMINI_MODEL=gemini-3.8-flash`, `GEMINI_EXTRACTOR_MODEL=gemini-3.1-flash-lite`, native IDs without `google/`). Replies, memory, style, summaries and OGG voice transcription go directly to Google. Or use `LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY` (`ANTHROPIC_MODEL=claude-haiku-5-5`; voice transcription is unavailable). Both native providers share the platform key but record each tenant's spend in its own `llm_usage` table; `CREDIT_LIMIT_USD` is the absolute sum of payments, and running tenants restart on top-up. Gemini uses published paid-tier estimates including thinking/audio/cache tokens, not Google's billing balance; hosted models are restricted to `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.1-flash-lite`, and `gemini-3.5-flash-lite`. For OpenRouter, set `LLM_PROVIDER=openrouter` and `OPENROUTER_MGMT_KEY` (from openrouter.ai/settings/provisioning-keys); the platform mints one capped key per tenant. The default `auto` selects Anthropic → Gemini → OpenRouter.
4. **Config**: `cp hosting.env.example hosting.env` and fill it in (`PUBLIC_URL`, `ADMIN_TG_ID`, client id/secret, keys).
5. **Install**: `uv sync`
6. **Run**: `pm2 start hosting/ecosystem.config.cjs && pm2 save` (first time also `pm2 startup systemd -u $USER --hp $HOME`).
7. **Reverse proxy**: Caddy (`hosting/Caddyfile.example`) or an nginx server block proxying to `127.0.0.1:8700`, with HTTPS.

After changing the hosting provider, restart the control plane, then restart each existing tenant from `/admin`. Provisioning writes an explicit tenant `LLM_PROVIDER` and removes obsolete credentials while preserving unrelated settings. Tenant dashboards cannot change platform keys/providers or select unpriced Gemini models.

## Admin

Sign in with the Telegram account matching `ADMIN_TG_ID`, open `/admin`. After a user pays, record the amount there: it raises their OpenRouter key limit or locally metered native-provider cap and starts their tenant. The same page can restart or stop a tenant.

## Tenant data

Each tenant lives in `tenants/<id>/` (its `.env`, `secretary.db`, prompts). The control plane state is `hosting.db`. Back both up.

Message text inside a tenant DB is pruned after `MESSAGE_RETENTION_DAYS` (default 7, user-adjustable down to 1 in their dashboard); only summaries and memory facts persist. `/privacy` on the site states this, says plainly that the operator can technically read tenant files, and links the self-host option. Nothing in the control plane reads message contents — keep it that way, the page promises it.

## Deleting a user

The user can delete their account from the account page (`/account`). That stops and removes the pm2 app `tgs-<id>`, deletes `tenants/<id>/`, disables the OpenRouter key and releases the tenant slot. The user should also remove the bot under Chat Automation; until then the router simply drops their updates.
