from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from puppy.swarm.run_swarm import run_mock_swarm  # noqa: E402


def main() -> None:
    result = run_mock_swarm()
    printable = {
        "signals": [
            output.signal.model_dump(mode="json") for output in result["agent_outputs"]
        ],
        "text_features": [
            feature.model_dump(mode="json") for feature in result["text_features"]
        ],
        "debate_request": result["debate_request"].model_dump(mode="json"),
        "debate_transcript": result["debate_transcript"].model_dump(mode="json"),
        "debate_verdict": result["debate_verdict"].model_dump(mode="json"),
    }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
