# AEDISIGHT — Stage 3 Vercel deployment

This repository exports the FastAPI application as `app` from the root-level
`main.py`. Vercel's native FastAPI/Python runtime detects that entry point; the
legacy `@vercel/python` build-and-rewrite configuration and the Mangum adapter
are not required. `vercel.json` is intentionally limited to the Vercel schema.

## 1. Before connecting the repository

- Push the reviewed Stage 1–3 changes to the practice Git repository.
- Confirm `.env` is not tracked (`git status --short` and `git check-ignore .env`).
- Do not upload `.env`, database dumps, or credentials to GitHub.
- Use Python 3.12 locally to match `.python-version` and the documented project
  runtime. The pin controls Vercel's build/runtime selection; it does not change
  an already-created local virtual environment.

## 2. Create the production database

Create a PostgreSQL database in the team's Neon project. Use the pooled
connection string for a serverless application where Neon provides one. Keep
the full connection string private; it contains database credentials. Ensure
SSL is required by the provider's connection string/settings.

The application creates missing SQLAlchemy tables on startup through
`Base.metadata.create_all()`. This creates missing tables such as `users` and
`audit_logs`; it does not migrate or alter existing table schemas, and it does
not copy local SQLite users into Neon. Create the production administrator
account through the team's approved secure process after deployment.

## 3. Import the project into Vercel

1. Sign in to the team's Vercel account.
2. Select **Add New → Project** and import the practice Git repository.
3. Keep the project root at the repository root (`phase2-main` contents, not a
   parent folder).
4. Do not add a custom Build Command or Output Directory. Let Vercel detect the
   FastAPI application from `main.py` and `requirements.txt`.
5. Deploy first as a Preview if the team wants to validate environment values
   before assigning the Production domain.

## 4. Configure Environment Variables

In **Project → Settings → Environment Variables**, add these values for the
appropriate Preview and Production environments:

| Name | Value |
|---|---|
| `APP_ENV` | `production` |
| `DATABASE_URL` | The team's Neon PostgreSQL connection string (secret) |
| `SESSION_SECRET` | A fresh random secret of at least 32 characters (secret) |
| `COOKIE_SECURE` | `true` |
| `SESSION_MAX_AGE` | `3600` |

Generate a secret locally with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Do not reuse the local development session secret or the `.env.example`
placeholder. Do not paste the generated value into chat, source code, or Git.

If the project is configured with separate Preview and Production values,
use a separate session secret and database for Preview. After changing
variables, trigger a new deployment; existing deployments do not receive
changed environment values retroactively.

## 5. Verify the deployment

After deployment, test these flows over the deployed HTTPS URL:

1. `/` loads and static CSS/JavaScript assets load successfully.
2. `/login` loads; valid admin credentials authenticate and redirect to
   `/admin_dashboard`.
3. An unauthenticated visitor is redirected to `/login` from
   `/admin_dashboard` and admin API endpoints.
4. A hospital account cannot open the admin dashboard or admin APIs.
5. The admin dashboard summary, user list, activity feed, and system status
   return successfully and reflect the production database.
6. Registration/login/logout continue to work and create expected audit
   records.
7. A hospital user can still open `/user_dashboard`; prediction requests still
   require authentication and a valid `X-CSRF-Token`.

Check Vercel Function logs if any request returns 500. Common configuration
failures include a missing `DATABASE_URL`, invalid Neon credentials/network
settings, missing `SESSION_SECRET`, or omitted runtime dependencies.

## 6. Data and model status caveat

The checked-in processed dengue dataset ends on 2025-12-29. The application
uses a 14-day freshness policy, so current forecasting is expected to report
stale/unavailable until recent validated data is supplied. Existing model
artifacts support the application's historical replay behavior; do not label
that replay as live surveillance. This deployment work does not change the
dataset, models, training code, or prediction logic.

## 7. Rollback

If production verification fails, use Vercel's Deployments page to promote the
last known-good deployment. Do not rotate database credentials or session
secrets as a first response unless they are compromised. Diagnose the Function
logs and environment configuration first.
