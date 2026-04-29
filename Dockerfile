FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir \
    fastapi uvicorn peft transformers \
    Pillow accelerate

WORKDIR /app
COPY server.py .

ENV MODEL_PATH=/model
ENV LORA_PATH=/lora
ENV PORT=8080

COPY model /model
COPY lora  /lora

CMD ["python", "server.py"]
