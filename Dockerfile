FROM python:3.10-slim-buster

WORKDIR /app/src

COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "main.py"]
