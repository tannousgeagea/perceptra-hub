"""
Image tag aliases: what an old tag name means after a rename or merge.

Ingest sources (POST /api/v1/ingest/images) know tags by name. When someone
renames "Kat" to "cat" (or merges it into "cat"), a source that keeps sending
"Kat" would otherwise create "Kat" again. Each rename or merge records
old name -> current name here, and ingest translates incoming names through it.

Stored in Organization.metadata["image_tag_aliases"] as {lower-case old name:
current name}; metadata is platform-owned (the settings API replaces
Organization.settings wholesale, so it would lose them).
"""
from typing import Dict

from django.db import transaction

KEY = "image_tag_aliases"


def load(organization_id) -> Dict[str, str]:
    from organizations.models import Organization

    metadata = Organization.objects.filter(pk=organization_id).values_list("metadata", flat=True).first()
    return dict((metadata or {}).get(KEY) or {})


def resolve(aliases: Dict[str, str], name: str) -> str:
    """The current name for `name` (unchanged when it was never renamed)."""
    return aliases.get(name.lower(), name)


def _update(organization_id, change) -> None:
    from organizations.models import Organization

    with transaction.atomic():
        metadata = (
            Organization.objects.select_for_update()
            .filter(pk=organization_id)
            .values_list("metadata", flat=True)
            .first()
        )
        metadata = dict(metadata or {})
        aliases = dict(metadata.get(KEY) or {})
        change(aliases)
        if aliases:
            metadata[KEY] = aliases
        else:
            metadata.pop(KEY, None)
        Organization.objects.filter(pk=organization_id).update(metadata=metadata)


def record_rename(organization_id, old: str, new: str) -> None:
    """`old` is now called `new` (a rename, or a merge into `new`)."""

    def change(aliases: Dict[str, str]) -> None:
        # Names that already led to `old` now lead to `new`.
        for key, value in list(aliases.items()):
            if value.lower() == old.lower():
                aliases[key] = new
        # `new` is a real tag again: it never stands for something else.
        aliases.pop(new.lower(), None)
        # Set after the pop, so a case-only rename ("cat" -> "Cat") keeps its alias.
        aliases[old.lower()] = new

    _update(organization_id, change)


def forget_target(organization_id, name: str) -> None:
    """The tag `name` was deleted: old names stop leading to it."""

    def change(aliases: Dict[str, str]) -> None:
        for key, value in list(aliases.items()):
            if value.lower() == name.lower():
                del aliases[key]

    _update(organization_id, change)
