#!/usr/bin/env bash
# Start Celery worker consuming all task-type queues
celery -A workers.celery_app worker \
  --loglevel=info \
  --concurrency="${CELERY_WORKER_CONCURRENCY:-4}" \
  --queues=eval_grammar_evaluation,eval_fluency_evaluation,eval_content_evaluation,eval_pronunciation_evaluation,eval_confidence_evaluation \
  --max-tasks-per-child=100
