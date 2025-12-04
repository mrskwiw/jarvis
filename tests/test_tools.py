import pytest

from tools.blogging import BloggingService
from tools.calls import CallService
from tools.email import EmailService, gmail_oauth_authorize_url, pop_setup_tool_form
from tools.registry import PermissionError, ToolRegistry


def test_blogging_service_draft_and_publish_slug():
    service = BloggingService(publish_dir="/blogs")
    draft = service.draft("Hello World", "body")
    published = service.publish(draft)
    assert published.url == "/blogs/hello-world.md"
    assert published.title == "Hello World"
    assert published.body == "body"


def test_call_service_place_and_receive():
    service = CallService(provider="test-provider")
    outbound = service.place_call("+123", "hi")
    inbound = service.receive_call("+555")
    assert outbound.status == "queued"
    assert inbound.status == "received"
    assert inbound.to == "+555"


def test_email_service_placeholders():
    service = EmailService()
    assert service.list_inbox() == []

    message = service.read_message("any")
    assert message.subject == ""
    assert message.body == ""

    reply = service.draft_reply("to@example.com", "subject", "body")
    assert reply.sender == "to@example.com"
    assert reply.subject == "subject"
    assert reply.body == "body"

    send_result = service.send_email("to@example.com", "subject", "body")
    assert send_result["provider"] == "dry-run"


def test_tool_registry_permissions_and_caching():
    registry = ToolRegistry()
    registry.register("tool", lambda: {"tool": True})

    with pytest.raises(PermissionError):
        registry.get("tool")

    load_count = {"value": 0}

    def loader():
        load_count["value"] += 1
        return {"loaded": True}

    registry.register("loader_tool", loader)
    first = registry.get("loader_tool", owner_verified=True)
    second = registry.get("loader_tool", owner_verified=True)
    assert first is second
    assert load_count["value"] == 1
    assert "loader_tool" in registry.names()

    with pytest.raises(KeyError):
        registry.get("unknown", owner_verified=True)


def test_tool_registry_describe_and_free_tier(monkeypatch):
    registry = ToolRegistry()
    registry.register("email", lambda: {"email": True})
    registry.register_schema("email", {"description": "Email tool", "input_schema": {"type": "object"}, "free_tier_only": True})
    desc = registry.describe()
    assert desc["email"]["description"] == "Email tool"
    assert desc["email"]["free_tier_only"] is True


def test_gmail_oauth_and_pop(monkeypatch):
    monkeypatch.setenv("GMAIL_CLIENT_ID", "cid")
    monkeypatch.setenv("GMAIL_CLIENT_SECRET", "secret")
    monkeypatch.setenv("GMAIL_REFRESH_TOKEN", "refresh")
    gmail = EmailService(provider="gmail")
    result = gmail.send_email("to@example.com", "hi", "body")
    assert result["provider"] == "gmail"

    pop = EmailService(provider="pop", pop_host="pop.example.com", pop_user="user")
    pop_result = pop.send_email("to@example.com", "hi", "body")
    assert pop_result["provider"] == "pop"
    url = gmail_oauth_authorize_url("cid", "https://app/callback")
    assert "client_id=cid" in url
    form = pop_setup_tool_form()
    assert "fields" in form
