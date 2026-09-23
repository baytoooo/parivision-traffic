# Hugging Face Space (Docker SDK). Build context = the Space repo, laid out by demo/build_space.sh.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
# created as the user, so the app can be unpacked into it (WORKDIR alone would make it root's)
RUN mkdir -p /home/user/app
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    YOLO_OFFLINE=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    DEMO_WORKDIR=/tmp/parivision-demo \
    OMP_NUM_THREADS=2
WORKDIR /home/user/app

COPY --chown=user demo/requirements.txt demo/requirements.txt
RUN pip install --no-cache-dir --user torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r demo/requirements.txt

COPY --chown=user . .
EXPOSE 7860
CMD ["uvicorn", "app:app", "--app-dir", "demo", "--host", "0.0.0.0", "--port", "7860"]
