FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY penguins/__init__.py penguins/config.py penguins/
COPY penguins/deployment/ penguins/deployment/
COPY penguins/display.json penguins/display.json
COPY penguins/assets/ penguins/assets/
EXPOSE 8501
ENV PYTHONPATH=/app
CMD ["streamlit", "run", "penguins/deployment/app/web.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
