FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt
RUN uv add -r requirements.txt

COPY streamlit_app_tfdl.py
COPY streamlit_app.py

COPY _data ./_data

CMD ["streamlit","run", "streamlit_app_tfdl.py", "--server.port=10000", "--server.address=0.0.0.0" ]