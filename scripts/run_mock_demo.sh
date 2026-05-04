#!/usr/bin/env bash
set -euo pipefail

python -m sentry_jury.cli --config configs/course_project_mock.yaml
