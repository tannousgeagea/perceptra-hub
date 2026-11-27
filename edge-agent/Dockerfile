# The sidecar. Run it next to your inference container, sharing the model volume:
#
#   CI publishes it as tannousgeagea/perceptra-edge-agent:<hub release>; to build it yourself:
#   docker build -t malumetrix-edge-agent edge-agent/
#   docker run -d --name malumetrix-edge --restart unless-stopped \
#     -v malumetrix-data:/var/lib/malumetrix \
#     -e MALUMETRIX_URL=https://malumetrix.example.com \
#     -e MALUMETRIX_ENROLL_CODE=K7QM-4XPA-9TWE \
#     -e MALUMETRIX_NOTIFY_URL=http://detector:8000/reload \
#     -e MALUMETRIX_HEALTH_URL=http://detector:8000/healthz \
#     -e MALUMETRIX_LOCAL_API=0.0.0.0:7070 \
#     malumetrix-edge-agent
#
# Your container mounts the same volume read-only and loads
# /var/lib/malumetrix/models/<model_id>/current/model.<format>.
# To restart a container as the reload hook, also mount /var/run/docker.sock
# and install the docker CLI in a derived image.
FROM python:3.11-slim

RUN useradd --system --uid 10001 malumetrix && mkdir -p /var/lib/malumetrix && chown malumetrix /var/lib/malumetrix
WORKDIR /opt/malumetrix-edge
COPY pyproject.toml README.md ./
COPY malumetrix_edge ./malumetrix_edge
RUN pip install --no-cache-dir . && rm -rf /root/.cache

USER malumetrix
VOLUME ["/var/lib/malumetrix"]
ENV MALUMETRIX_DATA_DIR=/var/lib/malumetrix PYTHONUNBUFFERED=1

# The hub release this image was published with (CI sets it); the agent reports it
ARG AGENT_VERSION=""
ENV MALUMETRIX_EDGE_VERSION=${AGENT_VERSION}
LABEL org.opencontainers.image.title="Malumetrix edge agent" \
      org.opencontainers.image.version="${AGENT_VERSION}"
EXPOSE 7070
ENTRYPOINT ["malumetrix-edge"]
CMD ["run"]
