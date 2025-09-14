FROM python:3.11-slim
WORKDIR /app
ADD . /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 5000
ENV NAME=OpenTOAll
CMD ["python", "app.py"]