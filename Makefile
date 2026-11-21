.PHONY: up seed test eval-retrieval eval-agent eval-baseline churn

up:
	docker compose up --build -d

seed:
	docker compose exec api python -m scripts.ingest data/jobs.jsonl

test:
	pytest -q

eval-retrieval:
	python -m scripts.eval_retrieval

eval-agent:
	python -m scripts.eval_agent

eval-baseline:
	python -m scripts.eval_agent --baseline

churn:
	python -m scripts.feed_churn
