# WashU RIS Compute1 wrapper for official Isaac Sim 5.1.0
#
# Content is identical to nvcr.io/nvidia/isaac-sim:5.1.0.
# Only file permissions are opened so LSF can run the container as the
# submitting user's UID (not root / not uid 1234).

FROM nvcr.io/nvidia/isaac-sim:5.1.0

USER root

# Read + execute for all users on the Isaac Sim install tree.
RUN chmod -R a+rX /isaac-sim

# Runtime caches/logs should live on mounted Storage1 volumes (see run_c1_isaac.sh).
# Ensure default cache paths exist and are writable if not masked by mounts.
RUN mkdir -p \
    /isaac-sim/.cache \
    /isaac-sim/.nv/ComputeCache \
    /isaac-sim/.nvidia-omniverse/logs \
    /isaac-sim/.nvidia-omniverse/config \
    /isaac-sim/.local/share/ov/data \
    /isaac-sim/.local/share/ov/pkg \
    && chmod -R a+rwx \
    /isaac-sim/.cache \
    /isaac-sim/.nv \
    /isaac-sim/.nvidia-omniverse \
    /isaac-sim/.local/share/ov
