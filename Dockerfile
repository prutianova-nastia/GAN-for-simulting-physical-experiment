FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN pip install --upgrade pip

RUN pip3 install pandas==1.0.1 \
                 numpy==1.18.1 \
                 matplotlib==3.1.3 \
                 seaborn==0.10.0 \
                 sklearn \
                 scipy==1.4.1

COPY ./ /

CMD ["python", "main.py"]