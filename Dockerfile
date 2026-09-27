FROM python:3.13-slim

WORKDIR /app

# Install core runtime dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic httpx requests

# Copy deliverables and datasets
COPY bot.py README.md STRATEGY.md submission.jsonl test_guardrails_and_dataset.py judge_simulator.py ./
COPY dataset/ ./dataset/
COPY examples/ ./examples/

# Expose standard challenge port
EXPOSE 8080

# Single worker to guarantee atomic in-memory state preservation across requests
CMD ["uvicorn", "bot:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
