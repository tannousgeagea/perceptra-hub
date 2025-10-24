#! /bin/bash
set -e

/bin/bash -c "python3 /home/$user/src/perceptra_hub/manage.py makemigrations"
/bin/bash -c "python3 /home/$user/src/perceptra_hub/manage.py migrate"
/bin/bash -c "python3 /home/$user/src/perceptra_hub/manage.py create_superuser"
/bin/bash -c "python3 /home/$user/src/perceptra_hub/manage.py collectstatic --noinput"

sudo -E supervisord -n -c /etc/supervisord.conf