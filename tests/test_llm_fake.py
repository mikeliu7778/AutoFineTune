from autofinetune.llm.client import FakeLLMClient


def test_fake_routes_by_schema_name():
    client = FakeLLMClient(
        handlers={
            "recommend_model": lambda system, user: {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "rationale": "fits 24GB",
            }
        }
    )
    out = client.complete_json("sys", "user", "recommend_model")
    assert out["model_id"].startswith("Qwen/")
