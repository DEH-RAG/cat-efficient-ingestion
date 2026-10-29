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

from .reconcile import reconcile_agent
from .registry import (
    delete_status,
    get_status,
    list_statuses,
)


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

    statuses = await list_statuses(agent_id, chat_id)

    if info.cheshire_cat is None:
        return statuses

    # One reconcile implementation for both the endpoint and the recovery
    # sweep: purge the vanished sources, then return the post-purge list.
    await reconcile_agent(agent_id, chat_id=chat_id, ccat=info.cheshire_cat)
    return await list_statuses(agent_id, chat_id)


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