FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3.12 python3-pip python3.12-venv

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install torch --index-url https://download.pytorch.org/whl/cu126

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# ENTRYPOINT ["python", "-m", "trainer.task"]