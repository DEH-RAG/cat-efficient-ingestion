"""Efficient implementation of the core ingestion profile ABCs.

The core owns one abstract class per part of the ingestion (pipeline, state
store, recovery, locks, tags) and ``BaseIngestionConfiguration`` exposes them
as a profile bundle. This module supplies the efficient versions, which simply
delegate to the plugin's existing phase machine, status registry, recovery
loop, distributed locks and recursive tag resolver — no algorithm is
reimplemented here.

Upstream compatibility: on an upstream CAT checkout without
``cat.services.ingestion`` the module still imports. The five ABCs are then
replaced by local placeholders and the classes below keep working as plain
plugin implementations; ``EfficientIngestionConfiguration`` does not gain any
new abstract method from the core, so it behaves exactly as before.

Import-safe: importing this module performs no Redis, filesystem or network
access. The plugin modules are imported lazily inside the methods, so the host
can import every plugin file at activation.
"""

from contextlib import AbstractContextManager
from typing import Any

try:
    from cat.services.ingestion import (
        IngestionLockManager,
        IngestionPipeline,
        IngestionRecovery,
        IngestionStateStore,
        IngestionTagResolver,
    )
except ImportError:  # upstream CAT: the core ingestion contract package is absent
    class IngestionPipeline:  # type: ignore[no-redef]
        """Placeholder when the core ingestion package is absent."""

    class IngestionStateStore:  # type: ignore[no-redef]
        """Placeholder when the core ingestion package is absent."""

    class IngestionRecovery:  # type: ignore[no-redef]
        """Placeholder when the core ingestion package is absent."""

    class IngestionLockManager:  # type: ignore[no-redef]
        """Placeholder when the core ingestion package is absent."""

    class IngestionTagResolver:  # type: ignore[no-redef]
        """Placeholder when the core ingestion package is absent."""


def _coerce_status(status: Any) -> Any:
    """Map a core/string status onto the plugin's ``IngestionStatus`` enum.

    The plugin registry compares the status against its own enum members, so a
    value coming from the core (a different enum class with the same string
    values) must be normalized by value. An unknown value degrades to
    ``processing`` rather than raising.
    """
    from .registry import IngestionStatus

    if isinstance(status, IngestionStatus):
        return status
    value = status.value if hasattr(status, "value") else str(status)
    try:
        return IngestionStatus(value)
    except ValueError:
        return IngestionStatus.PROCESSING


def _is_url(source: str) -> bool:
    from cat.utils import is_url

    return is_url(source)


class EfficientPhaseDagPipeline(IngestionPipeline):
    """Phase-DAG pipeline: runs the plugin's one phase machine per source.

    ``execute`` hands a single source to the existing phase machine;
    ``delete`` delegates to :func:`phases.delete_file`, which already enforces
    the artifacts -> file -> status-row-last order. No deletion logic is
    reimplemented here.
    """

    def phases(self) -> list[Any]:
        """Return the built-in phases, in execution order.

        Plugin-contributed phases are resolved asynchronously by the phase
        machine (they depend on the active plugin managers); the synchronous
        core contract can only expose the built-ins.
        """
        from .phases import _BUILTIN_PHASE_CLASSES

        return [cls() for cls in _BUILTIN_PHASE_CLASSES]

    async def execute(self, source: str, cat) -> None:
        """Run the phase machine for ``source`` through ``cat``'s scope."""
        from cat.looking_glass.models import StoredSourceWithMetadata
        from cat.services.memory.models import VectorMemoryType

        from .reembed import reembed_sources

        # Normalize the caller exactly like the engine: a StrayCat is chat
        # scoped and resolves its CheshireCat; a CheshireCat is agent scoped.
        if hasattr(cat, "agent_key") and not hasattr(cat, "_id"):
            ccat = await cat.lizard.get_cheshire_cat(cat.agent_key)
            chat_id = cat.id
        else:
            ccat = cat
            chat_id = None
        if ccat is None:
            raise ValueError(
                f"Agent '{getattr(cat, 'agent_key', None)}' not found; cannot ingest."
            )

        collection_name = (
            VectorMemoryType.EPISODIC if chat_id else VectorMemoryType.DECLARATIVE
        )
        metadata = {"chat_id": chat_id} if chat_id else {}
        source_obj = StoredSourceWithMetadata(
            name=source, path=source, content=None, metadata=metadata
        )
        await reembed_sources(
            ccat,
            collection_name,
            [source_obj],
            stale_after=0.0,
            caller_cat=cat,
        )

    async def delete(self, source: str, cat, scope: str | None = None) -> bool:
        """Delete ``source`` through the existing file-deletion path."""
        from .phases import delete_file

        return await delete_file(source, cat, scope)


class EfficientStateStore(IngestionStateStore):
    """Status store backed by the plugin's ingestion registry."""

    async def get(self, agent_id: str, scope: str, source: str) -> dict[str, Any] | None:
        from .registry import get_status

        return await get_status(agent_id, scope, source)

    async def set(self, agent_id: str, scope: str, source: str, **fields) -> dict[str, Any]:
        from .registry import set_status

        status = _coerce_status(fields.pop("status", "processing"))

        type_ = fields.pop("type_", None)
        if type_ is None:
            type_ = fields.pop("type", None)
        else:
            fields.pop("type", None)
        if type_ is None:
            type_ = "url" if _is_url(source) else "file"

        # The core store accepts arbitrary extra knobs; the plugin registry has
        # no warning channel on the lifecycle write (warnings belong to the
        # phase diary), so drop it instead of inventing a diary entry.
        fields.pop("warnings", None)

        return await set_status(
            agent_id, scope, source, type_=type_, status=status, **fields
        )

    async def clear(self, agent_id: str, scope: str, source: str) -> None:
        from .registry import _delete_status_doc

        await _delete_status_doc(agent_id, scope, source)


class EfficientRecovery(IngestionRecovery):
    """Recovery backed by the plugin's startup/resume and reconcile passes."""

    async def resume_pending(self, lizard) -> int:
        """Run one recovery pass; return how many agents were swept."""
        from .resume import resume

        return await resume(lizard)

    async def reconcile(self, lizard) -> int:
        """Reconcile every agent; return how many status rows were purged."""
        from cat.db.cruds import settings as crud_settings
        from cat.exceptions import CustomNotFoundException

        from .reconcile import reconcile_agent

        total = 0
        for agent in await crud_settings.get_agents():
            agent_id = agent["agent_id"]
            try:
                ccat = await lizard.get_cheshire_cat(agent_id)
            except CustomNotFoundException:
                continue
            if ccat is None:
                continue
            purged = await reconcile_agent(agent_id, ccat=ccat)
            total += len(purged)
        return total


class EfficientLockManager(IngestionLockManager):
    """Distributed locks backed by the plugin's registry primitives."""

    def source_lock(self, agent_id: str, source: str) -> AbstractContextManager[Any]:
        """Return the plugin's per-source distributed lock."""
        from .registry import source_lock

        return source_lock(agent_id, source)

    def tenant_write_lock(self, agent_id: str) -> AbstractContextManager[Any]:
        """Return the per-agent write lock (a distributed lock on the tenant)."""
        from cat.db import crud

        from .registry import ingestion_lock_timeout_seconds

        timeout = ingestion_lock_timeout_seconds()
        return crud.distributed_lock(
            f"ingestion-tenant:{agent_id}",
            timeout=timeout,
            blocking_timeout=timeout,
        )


class EfficientTagResolver(IngestionTagResolver):
    """Recursive phase tags backed by the plugin's tag resolver."""

    def own_tag(self, phase, settings) -> str:
        """Return the detached own tag of ``phase`` from ``settings``."""
        from .phase_tag_resolver import own_tag

        return own_tag(phase.id, settings)

    async def resolve(self, cat, phases) -> dict[str, str]:
        """Resolve every phase's recursive tag exactly as the machine does."""
        from .phases import compute_tags

        return await compute_tags(cat, phases)
