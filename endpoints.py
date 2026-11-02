"""``/ingestion/status`` endpoints.

Exposes the ingestion-status registry over HTTP. Before returning, the
registry is reconciled against the canonical source lists (files on disk via
the file manager, URLs via the web points in the vector store) and entries
whose source no longer exists are purged (lazy delete). Purge failures are
logged and never fail the request.

The engine configuration endpoints (``/ingestion/settings*``) are served by
the CAT core route (``cat/routes/ingestion.py``, ``AuthResource.INGESTION``);
this plugin must NOT register them again or the paths would conflict.
"""
from typing import Dict, List, Optional

from fastapi import Query

from cat import (
    AuthorizedInfo,
    AuthPermission,
    AuthResource,
    check_permissions,
    endpoint,
)

from .phases import phase_catalog
from .reconcile import reconcile_agent
from .registry import (
    delete_status,
    get_status,
    list_statuses,
    status_catalog,
    status_label,
)


def _decorate_entries(
    entries: List[Dict],
    phase_labels: Dict[str, str],
    state_labels: Dict[str, str],
) -> List[Dict]:
    """Add the display-only ``phase_label`` / ``status_label`` to every entry.

    ``phase`` and ``status`` are kept untouched: they remain the machine
    fields. ``phase_label`` is the current phase's label, or ``None`` when the
    entry has no phase. A work phase that is not a registered ingestion phase
    (``downloading`` for URLs) falls back to its lifecycle label.
    """
    decorated: List[Dict] = []
    for entry in entries:
        row = dict(entry)
        phase = row.get("phase")
        phase_label_value = phase_labels.get(phase) if phase else None
        if phase and phase_label_value is None:
            phase_label_value = state_labels.get(phase)
        row["phase_label"] = phase_label_value
        row["status_label"] = status_label(row.get("status"))
        decorated.append(row)
    return decorated


@endpoint.get("/status", prefix="/ingestion", tags=["Ingestion"])
async def get_ingestion_status(
    chat_id: Optional[str] = Query(
        default=None,
        description="Conversation id to scope the statuses to a single chat.",
    ),
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.READ),
) -> List[Dict]:
    """Return the ingestion-status registry, reconciled against canonical sources.

    Agent scope by default; pass ``?chat_id=<id>`` for a conversation scope.
    Entries whose source no longer exists (file removed, URL without web
    points, or conversation gone) are purged before returning.
    """
    agent_id = info.agent_id
    if agent_id is None:
        return []

    catalog = await phase_catalog(info.cheshire_cat)
    phase_labels = {row["id"]: row["label"] for row in catalog}
    state_labels = {row["id"]: row["label"] for row in status_catalog()}

    if info.cheshire_cat is None:
        statuses = await list_statuses(agent_id, chat_id)
        return _decorate_entries(statuses, phase_labels, state_labels)

    # One reconcile implementation for both the endpoint and the recovery
    # sweep: purge the vanished sources, then return the post-purge list.
    await reconcile_agent(agent_id, chat_id=chat_id, ccat=info.cheshire_cat)
    statuses = await list_statuses(agent_id, chat_id)
    return _decorate_entries(statuses, phase_labels, state_labels)


@endpoint.get("/phases", prefix="/ingestion", tags=["Ingestion"])
async def get_ingestion_phases(
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.READ),
) -> Dict:
    """Return the phase catalog and the lifecycle-state catalog.

    ``phases`` lists every registered phase in dependency order (every
    dependency before its dependents), each row
    ``{"id", "label", "order", "depends_on"}``. ``states`` lists the
    lifecycle states as ``{"id", "label"}``. Labels are English display
    defaults; clients localize by ``id`` and never use a label as a key.
    """
    return {
        "phases": await phase_catalog(info.cheshire_cat),
        "states": status_catalog(),
    }


@endpoint.delete("/status", prefix="/ingestion", tags=["Ingestion"])
async def delete_ingestion_status(
    source: str = Query(..., description="The file name or URL whose status row to delete."),
    scope: Optional[str] = Query(
        default=None,
        description='Scope of the entry: "agent" (default) or a conversation chat_id.',
    ),
    info: AuthorizedInfo = check_permissions(AuthResource.MEMORY, AuthPermission.DELETE),
) -> Dict:
    """Delete the ingestion-status row for one source.

    Only terminal statuses (error, completed) may be deleted here. A source
    that is still being ingested (uploaded/downloading/downloaded/processing)
    MUST NOT be deleted this way — the running pipeline would keep going and a
    ``completed`` row would be re-created by the lifecycle hooks, orphaning
    the work. To abandon an in-flight source, remove the file instead (that
    deletes the file, its memory points, and the status row together).
    """
    agent_id = info.agent_id
    if agent_id is None:
        return {"deleted": False, "reason": "no_agent"}

    scope = scope or "agent"
    doc = await get_status(agent_id, scope, source)
    if doc is None:
        return {"deleted": False, "reason": "not_found"}

    status = doc.get("status")
    if status in ("uploaded", "downloading", "downloaded", "processing"):
        return {
            "deleted": False,
            "reason": "in_flight",
            "message": "source is being processed; remove the file to abandon it",
        }

    await delete_status(agent_id, scope, source)
    return {"deleted": True}