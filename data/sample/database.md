# Database handbook

## Connection
The primary database is PostgreSQL 16. Applications connect through PgBouncer on port 6432, not directly on port 5432.

## Migrations
Schema migrations are written with Alembic and must be backwards compatible, because old and new application versions run side by side during a deploy.

## Backups
Full backups run nightly at 02:00 UTC and are kept for 35 days. Point-in-time recovery is available for the last 7 days.
