FROM torchtitan-npu:cann9.0.0-torch2.12

WORKDIR /workspace/ScaleTorch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

ENV FLASH_ATTEN=1
ENV DTYPE=bfloat16
ENV HCCL_CONNECT_TIMEOUT=7200
ENV HCCL_EXEC_TIMEOUT=7200
ENV TASK_QUEUE_ENABLE=2
ENV COMBINED_ENABLE=1
ENV HCCL_BUFFSIZE=120
ENV HCCL_OP_BASE_FFTS_MODE_ENABLE=1
ENV ASCEND_LAUNCH_BLOCKING=0
ENV PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

ENTRYPOINT ["python", "tools/train.py"]
