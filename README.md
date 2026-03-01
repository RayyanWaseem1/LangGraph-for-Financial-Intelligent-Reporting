# LangGraph-for-Financial-Intelligent-Reporting

## Secrets handling

Keep real secrets in `.env` (ignored by git) and commit only placeholder values.

1. Create your local env file:
```bash
cp .env.example .env
```
2. Fill in real keys in `.env`.
3. Run the repo scan before commit/push:
```bash
./scripts/scan_secrets.sh
```
