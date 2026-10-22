"""Collection routing and error handling against a mocked Qdrant server."""

import traceback
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from jarvis.vector_store import VectorStore


@pytest.fixture
def server() -> MagicMock:
    client = MagicMock()
    collections = {}

    def create(collection_name: str, vectors_config: object) -> None:
        assert collection_name not in collections
        collections[collection_name] = {"dim": vectors_config.size, "points": {}}

    def info(collection_name: str) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(size=collections[collection_name]["dim"])
                )
            )
        )

    def upsert(collection_name: str, points: list) -> None:
        for point in points:
            collections[collection_name]["points"][point.id] = point

    def query(collection_name: str, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            points=[
                SimpleNamespace(id=p.id, payload=p.payload, score=0.9)
                for p in collections[collection_name]["points"].values()
            ]
        )

    def delete(collection_name: str, points_selector: object, wait: bool) -> None:
        assert wait
        for point_id in points_selector.points:
            collections[collection_name]["points"].pop(point_id, None)

    client.collection_exists.side_effect = lambda collection_name: collection_name in collections
    client.create_collection.side_effect = create
    client.get_collection.side_effect = info
    client.upsert.side_effect = upsert
    client.query_points.side_effect = query
    client.delete.side_effect = delete
    return client


def test_two_collections_isolate_identical_fragment_and_point_ids(server: MagicMock) -> None:
    with patch("jarvis.vector_store.QdrantClient", return_value=server):
        personal = VectorStore(collection_name="jarvis_fragments")
        mom = VectorStore(collection_name="jarvis_mom_fragments")
    payload = {"parent_conversation_id": "conv", "segment_id": "seg", "conversation_date": "date"}
    point_id = UUID("00000000-0000-0000-0000-000000000001")
    with patch("jarvis.vector_store.uuid.uuid4", return_value=point_id):
        personal_id = personal.upsert([1.0, 0.0], "same-fragment", payload)
        mom_id = mom.upsert([1.0, 0.0, 0.0], "same-fragment", payload)
    assert personal_id == mom_id == str(point_id)
    with patch("jarvis.vector_store.QdrantClient", return_value=server):
        personal = VectorStore(collection_name="jarvis_fragments")
        mom = VectorStore(collection_name="jarvis_mom_fragments")
    assert personal.search([1.0, 0.0]) == [("same-fragment", 0.9, personal_id)]
    assert mom.search([1.0, 0.0, 0.0]) == [("same-fragment", 0.9, mom_id)]
    personal.delete_points([personal_id])
    assert personal.search([1.0, 0.0]) == []
    assert mom.search([1.0, 0.0, 0.0]) == [("same-fragment", 0.9, mom_id)]
    assert server.upsert.call_args.kwargs["points"][0].payload == {
        "fragment_id": "same-fragment",
        **payload,
    }
    for method in (
        server.collection_exists,
        server.create_collection,
        server.get_collection,
        server.upsert,
        server.query_points,
        server.delete,
    ):
        assert method.called
        assert {call.kwargs["collection_name"] for call in method.call_args_list} <= {
            "jarvis_fragments",
            "jarvis_mom_fragments",
        }
    server.delete_collection.assert_not_called()
    server.recreate_collection.assert_not_called()


def test_existing_collection_is_never_recreated_and_dimensions_are_per_instance(
    server: MagicMock,
) -> None:
    with patch("jarvis.vector_store.QdrantClient", return_value=server):
        original = VectorStore(collection_name="custom")
        original.upsert([1.0, 0.0], "frag", {})
        reopened = VectorStore(collection_name="custom")
    server.create_collection.reset_mock()
    for operation in (lambda: reopened.upsert([1.0], "frag", {}), lambda: reopened.search([1.0])):
        with pytest.raises(ValueError, match="custom.*expects 2"):
            operation()
    server.create_collection.assert_not_called()
    server.delete_collection.assert_not_called()
    assert original.search([1.0, 0.0])


def test_collection_created_after_client_construction_loads_dimension(server: MagicMock) -> None:
    with patch("jarvis.vector_store.QdrantClient", return_value=server):
        reader = VectorStore()
        writer = VectorStore()
    writer.upsert([1.0, 0.0], "frag", {})
    with pytest.raises(ValueError, match="expects 2"):
        reader.upsert([1.0], "other", {})


def test_missing_collection_search_and_delete_never_create(server: MagicMock) -> None:
    with patch("jarvis.vector_store.QdrantClient", return_value=server):
        store = VectorStore()
    with pytest.raises(RuntimeError, match="does not exist"):
        store.search([1.0])
    store.delete_points(["missing"])
    store.delete_points([])
    server.create_collection.assert_not_called()
    server.delete.assert_not_called()


@pytest.mark.parametrize(
    "method,operation",
    [
        ("collection_exists", "search"),
        ("get_collection", "search"),
        ("query_points", "search"),
        ("upsert", "upsert"),
        ("delete", "delete"),
        ("collection_exists", "delete"),
        ("create_collection", "create"),
    ],
)
def test_operation_failures_propagate_without_secret_output(
    server: MagicMock, method: str, operation: str, caplog: pytest.LogCaptureFixture
) -> None:
    with patch("jarvis.vector_store.QdrantClient", return_value=server):
        store = VectorStore()
    if operation != "create":
        store.upsert([1.0], "frag", {})
    if method == "get_collection":
        store._expected_dim = None
    getattr(server, method).side_effect = RuntimeError("private-api-key from remote response")
    with pytest.raises(RuntimeError) as error:
        if operation == "search":
            store.search([1.0])
        elif operation == "delete":
            store.delete_points(["point"])
        else:
            store.upsert([1.0], "frag", {})
    rendered = "".join(traceback.format_exception(error.value))
    assert "private-api-key" not in rendered + caplog.text


def test_connection_failures_are_sanitized() -> None:
    key = "secret-key"
    with patch("jarvis.vector_store.QdrantClient", side_effect=RuntimeError("secret-key")):
        with pytest.raises(RuntimeError) as error:
            VectorStore(url="https://qdrant.example", api_key=key)
    assert "secret-key" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize(
    "settings",
    [
        {"url": "https://example.com", "host": "localhost"},
        {"url": "https://example.com", "port": 6333},
        {"url": "https://user:secret@example.com"},
        {"url": "ftp://example.com"},
        {"url": 0},
        {"host": "http://localhost"},
        {"host": False},
        {"port": 0},
        {"collection_name": ""},
        {"collection_name": "unsafe/collection"},
    ],
)
def test_invalid_direct_configuration_fails_before_connecting(settings: dict) -> None:
    with patch("jarvis.vector_store.QdrantClient") as client:
        with pytest.raises(ValueError):
            VectorStore(**settings)
    client.assert_not_called()
