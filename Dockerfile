FROM python:3.11-slim

WORKDIR /app

COPY requirements-container.txt .
RUN pip install --no-cache-dir -r requirements-container.txt

COPY . .

CMD ["gunicorn", "app.index:app", "--bind", "0.0.0.0:8000"]
