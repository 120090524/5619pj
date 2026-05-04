#!/usr/bin/env bash
set -euo pipefail

git init
git branch -m main || true
git add .
git commit -m "Initial course project scaffold for SENTRY-Jury"
echo "Done. Now add your remote and push:"
echo "git remote add origin <your-github-repo-url>"
echo "git push -u origin main"
