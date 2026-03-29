FROM python:3.12-slim

WORKDIR /app

COPY dashboard/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard/ ./dashboard/

ENV PYTHONPATH=/app
EXPOSE 3000
CMD ["python", "-m", "dashboard.app"]
