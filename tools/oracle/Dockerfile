# CPU oracle image. Runs the HF reference dump inside a Linux guest, which
# escapes the host EDR/code-signing block (see README "Why Python / the EDR catch").
# No Metal/AMX here — this is the CPU correctness oracle only.
FROM python:3.12-slim

# torch on linux/arm64 from PyPI is already a CPU build.
RUN pip install --no-cache-dir \
        numpy \
        torch \
        "transformers>=4.46" \
        "huggingface_hub>=0.26" \
        "gguf>=0.10.0" \
        safetensors \
        accelerate

WORKDIR /work
# scripts are bind-mounted at run time; default to the dumper
ENTRYPOINT ["python"]
CMD ["dump_reference.py", "--help"]
