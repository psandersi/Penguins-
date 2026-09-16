FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY penguins/ penguins/
COPY models/penguins.joblib models/penguins.joblib
EXPOSE 8000
CMD ["uvicorn", "penguins.deployment.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
