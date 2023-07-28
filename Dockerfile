FROM python:3.9-slim
WORKDIR /code

RUN pip install \
    torch==1.12.0+cpu \
    torchvision==0.13.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html \
    && rm -rf /root/.cache/pip

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./vit_api.py ./gpt_api.py  /code/

COPY ./Models /code/Models

COPY ./run.sh ./run.sh

RUN chmod a+x ./run.sh

EXPOSE 8000 8080

CMD ["./run.sh"]

# docker build -t fast_api .
# docker run -it -p 8000:8000 -p 8080:8080 fast_api