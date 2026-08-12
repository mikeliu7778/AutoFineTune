from autofinetune.llm.client import EST_COST_USD_PER_CALL, FakeLLMClient


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
    assert client.cost_usd_est == EST_COST_USD_PER_CALL
