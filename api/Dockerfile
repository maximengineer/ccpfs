FROM python:3.12-slim

WORKDIR /app

# Dependencies first (cached layer)
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Framework code that the API imports
COPY config.py ./config.py
COPY policy/ ./policy/
COPY evaluation/ ./evaluation/
COPY api/ ./api/

# Ensure project root is on PYTHONPATH for config/policy/evaluation imports
ENV PYTHONPATH=/app
# Data and models are volume-mounted at runtime
ENV CCPFS_DATA_DIR=/data
ENV CCPFS_MODEL_DIR=/models

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
