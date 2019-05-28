ARG PARENT_IMAGE=rlworkgroup/garage-base
FROM $PARENT_IMAGE

WORKDIR /root/code/garage
RUN ["/bin/bash", "-c", "source activate garage && pip install -e .[intel]"]

# Ready, set, go.
ENTRYPOINT ["docker/entrypoint-runtime.sh"]
