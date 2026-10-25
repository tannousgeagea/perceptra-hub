#! /bin/bash
set -e

# Never run a writing makemigrations here. A migration generated at start is an
# unreviewed schema change that migrate then applies to a shared database, and
# the file dies with the container while django_migrations keeps its row, so
# later images believe it applied and never recreate what it dropped. Migrations
# are generated in the repo and committed. Drift is only reported; startup
# continues so a model/migration mismatch does not take the service down.
drift_rc=0
drift_output=$(python3 "/home/$user/src/perceptra_hub/manage.py" makemigrations --check --dry-run 2>&1) || drift_rc=$?
if [ "$drift_rc" -ne 0 ]; then
    echo "WARNING: makemigrations --check reported drift or failed (exit $drift_rc); not generating migrations at runtime. If models changed, run makemigrations in the repo and commit it; otherwise read the output below." >&2
    echo "$drift_output" >&2
fi

/bin/bash -c "python3 /home/$user/src/perceptra_hub/manage.py migrate"
/bin/bash -c "python3 /home/$user/src/perceptra_hub/manage.py create_superuser"
/bin/bash -c "python3 /home/$user/src/perceptra_hub/manage.py collectstatic --noinput"

# supervisord drops to ${user}, then reopens the container's stdout/stderr to
# give each program /dev/fd/1 and /dev/fd/2. Docker creates those pipes as
# root:root 0600, so the reopen would fail with EACCES after the privilege drop
# and no program would start. Hand the pipes over while still root.
# Do not redirect this command's stderr: that would resolve /proc/self/fd/2 to
# the redirect target instead of the real stderr pipe.
# Fatal, not a warning: without this every program fails to spawn, and because
# supervisord's own log is a file rather than the container's stdout, that
# outage would be invisible in `docker logs`. Crash-looping is what puts it on
# the healthcheck and in Prometheus.
if ! chown "${user}:${group}" /proc/self/fd/1 /proc/self/fd/2; then
    echo "FATAL: cannot give ${user} the container log pipes; every supervisord program would fail to start with EACCES" >&2
    exit 1
fi

sudo -E supervisord -n -c /etc/supervisord.conf