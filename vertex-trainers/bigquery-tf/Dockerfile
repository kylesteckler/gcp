FROM gcr.io/deeplearning-platform-release/base-cu110

ENV PYTHONBUFFERED True
WORKDIR /app 

RUN pip install tensorflow[gpu]
RUN pip install --no-deps tensorflow-io
RUN pip install fire 

COPY train.py .
COPY config.py . 
ENTRYPOINT ["python", "train.py"]
