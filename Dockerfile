FROM python:3.8

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

ADD app.py app.py
ADD pipeline.py pipeline.py

CMD ["flask", "run", "--host", "0.0.0.0"]
EXPOSE 5000