import json

from generative_training_audit.cli import main, run


def test_framework_light_audits_pass():
    results = run("all")
    assert all(result.passed for result in results)
    assert {result.name for result in results} == {
        "bf16_ema",
        "noise_pairing",
        "scheduler_trace",
        "gan_gradient",
    }


def test_json_output_is_machine_readable(capsys):
    assert main(["ema", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["name"] == "bf16_ema"
    assert payload[0]["passed"] is True
