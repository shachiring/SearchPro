FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONUNBUFFERED=1
EXPOSE 8501
CMD ["sh", "-c", "streamlit run main.py --server.port $PORT --server.headless true --server.enableCORS false"]
