from dialogue.confirmation import confirm_action, needs_confirmation
from dialogue.controller import ConversationController
from nlp.intent import Intent
from nlp.router import LLMRouter, RouterDecision


class StubRouter(LLMRouter):
    def __init__(self) -> None:
        super().__init__(default_system_prompt="system")

    def normalize_request(self, intent: Intent, message: str, tools=None) -> RouterDecision:
        payload = {"model": intent.model, "payload_message": message, "tools": tools}
        return RouterDecision(model=intent.model, payload=payload)


def test_confirmation_helpers():
    assert needs_confirmation("email", "send")
    assert not needs_confirmation("email", "read")
    assert not needs_confirmation("unknown", "anything")

    prompts = []

    def prompt_yes(text: str) -> str:
        prompts.append(text)
        return "Yes"

    assert confirm_action(prompt_yes, "email", "send")
    assert prompts[-1].startswith("Do you want to send")

    def prompt_no(_: str) -> str:
        return "no"

    assert not confirm_action(prompt_no, "call", "place")


def test_conversation_controller_history_and_response():
    controller = ConversationController(router=StubRouter(), max_history=2)
    controller.record_turn("user", "first")
    controller.record_turn("assistant", "reply")
    controller.record_turn("user", "second")

    # Oldest turn is trimmed to respect max_history
    assert len(controller.history) == 2
    assert controller.history[0].content == "reply"

    intent = Intent(label="email", complexity="simple", model="haiku", confidence=0.9)
    payload = controller.respond(intent, "send it", tools=["tool-a"])
    assert payload["payload_message"] == "send it"
    assert payload["tools"] == ["tool-a"]
    assert controller.history[-1].content == "send it"

    summary = controller.summarize_task("done")
    assert summary.startswith("Summary: done")
    assert controller.history[-1].content == summary


def test_conversation_summary_and_persistence(tmp_path):
    summary_file = tmp_path / "summary.txt"
    controller = ConversationController(router=StubRouter(), max_history=3, summary_after=3, summary_path=summary_file)
    controller.record_turn("user", "a")
    controller.record_turn("assistant", "b")
    controller.record_turn("user", "c")
    assert len(controller.history) == 1  # summarized
    assert "Summary of prior turns" in controller.history[0].content
    assert summary_file.read_text().startswith("Summary of prior turns")
