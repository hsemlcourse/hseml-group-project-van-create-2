FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY notebooks/ notebooks/

CMD ["python3", "-m", "src.preprocessing.clean"]
