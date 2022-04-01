FROM gcr.io/deeplearning-platform-release/pytorch-gpu
RUN pip install -U fire fastai pandas
WORKDIR /app
COPY train.py .

ENTRYPOINT ["python", "train.py"]