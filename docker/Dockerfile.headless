ARG PARENT_IMAGE=rlworkgroup/garage-base
FROM $PARENT_IMAGE

# apt dependencies
RUN \
  apt-get -y -q update && \
  # Prevents debconf from prompting for user input
  # See https://github.com/phusion/baseimage-docker/issues/58
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # Dummy X server
    xvfb \
    pulseaudio && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Ready, set, go.
ENTRYPOINT ["docker/entrypoint-headless.sh"]
