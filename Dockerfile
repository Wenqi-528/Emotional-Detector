FROM tiangolo/uwsgi-nginx-flask:python3.8

# the port number need to be incremented as each new project is added
# script will edit the two lines below
ENV LISTEN_PORT opsport
EXPOSE opsport

COPY ./app /app

RUN apt-get update ##[edited]

RUN pip install -r requirements.txt