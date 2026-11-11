
FROM python:3.12-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY streamlit_app.py .
COPY streamlit_app_tfdl.py .

# Copy policy documents
COPY _data/ ./_data/

# Streamlit port
EXPOSE 8501

# Start application
CMD ["streamlit", "run", "streamlit_app_tfdl", "--server.address=0.0.0.0", "--server.port=8501"]
