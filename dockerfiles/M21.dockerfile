FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/messi/main.py main.py

CMD ["python", "src/messi/main.py"]
