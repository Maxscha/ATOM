FROM python:3.6

RUN apt-get update && apt-get install ctags

WORKDIR /app

COPY ./* /app/

RUN pip3 install -r requirements.txt

CMD ["bash"]