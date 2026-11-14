from django.test import SimpleTestCase, TestCase

from memberships.models import Role
from memberships.roles import (
    ALL_ROLE_NAMES,
    PERMISSION_TO_ORG_ROLES,
    PERMISSION_TO_PROJECT_ROLES,
    OrgRole,
    ProjectRole,
    project_permissions_for,
    role_catalog,
)


class RoleRowsTests(TestCase):
    """Every role in roles.py has a Role row (seeded after migrate)."""

    def test_every_role_has_a_row(self):
        names = set(Role.objects.values_list("name", flat=True))
        self.assertEqual(set(ALL_ROLE_NAMES) - names, set())


class RoleCatalogTests(SimpleTestCase):
    """The catalog the console reads matches the rules the API enforces."""

    def test_catalog_uses_plain_names(self):
        catalog = role_catalog()
        self.assertEqual([r["name"] for r in catalog["project_roles"]], [r.value for r in ProjectRole])
        self.assertEqual([r["name"] for r in catalog["org_roles"]], [r.value for r in OrgRole])
        for roles in catalog["project_permissions"].values():
            for name in roles:
                self.assertIn(name, {r.value for r in ProjectRole})

    def test_catalog_matrix_matches_enforced_matrix(self):
        catalog = role_catalog()
        for perm, roles in PERMISSION_TO_PROJECT_ROLES.items():
            self.assertEqual(set(catalog["project_permissions"][perm]), {r.value for r in roles})
        for perm, roles in PERMISSION_TO_ORG_ROLES.items():
            self.assertEqual(set(catalog["org_permissions"][perm]), {r.value for r in roles})

    def test_project_permissions(self):
        self.assertNotIn("annotate", project_permissions_for("viewer", "viewer"))
        self.assertIn("annotate", project_permissions_for("viewer", "annotator"))
        self.assertNotIn("review", project_permissions_for("viewer", "annotator"))
        self.assertIn("review", project_permissions_for("viewer", "reviewer"))
        self.assertNotIn("edit", project_permissions_for("viewer", "reviewer"))
        self.assertIn("edit", project_permissions_for("viewer", "editor"))
        self.assertIn("manage", project_permissions_for("admin", None))  # org admin bypass
        self.assertEqual(project_permissions_for("editor", None), frozenset())  # not a member
