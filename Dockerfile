FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -c "from wordllama import WordLlama; WordLlama.load()"
COPY aeromatch aeromatch
COPY scripts scripts
COPY data data
EXPOSE 8000
CMD ["uvicorn", "aeromatch.main:app", "--host", "0.0.0.0", "--port", "8000"]
