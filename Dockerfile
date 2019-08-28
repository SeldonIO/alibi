FROM python:3.7

COPY requirements/requirements_all.txt ./

RUN apt-get update
RUN apt-get install pandoc -y
RUN pip install -r requirements_all.txt
RUN python -m spacy download en_core_web_md

