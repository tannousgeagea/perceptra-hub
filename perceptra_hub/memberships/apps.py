from django.apps import AppConfig
from django.db.models.signals import post_migrate


def seed_roles(sender, **kwargs):
    """
    Make sure every role named in memberships/roles.py has a Role row.

    roles.py is the single source of truth; this runs after every `migrate`,
    so a role added there exists in the database without a data migration.
    Existing rows are left alone (only missing ones are created).
    """
    from memberships.models import Role
    from memberships.roles import ALL_ROLE_NAMES, ROLE_DESCRIPTIONS

    existing = set(Role.objects.values_list("name", flat=True))
    Role.objects.bulk_create(
        Role(name=name, description=ROLE_DESCRIPTIONS.get(name, ""))
        for name in ALL_ROLE_NAMES
        if name not in existing
    )


class MembershipsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'memberships'

    def ready(self):
        post_migrate.connect(seed_roles, sender=self)
