"""Boot the real API in-process with a trained model and run each endpoint once.

    python scripts/demo_api.py artifacts/ai-detector-smoke
"""

import json
import sys
import tempfile

from fastapi.testclient import TestClient

from plagai.api.app import create_app
from plagai.config import Settings

HUMAN = ("honestly i didnt think the trip would be that bad but the bus broke down twice and we ended up "
         "sleeping in a gas station in nevada lol. my brother kept saying it was an adventure. it was not an "
         "adventure. the vending machine ate three of my dollars and the guy at the counter just shrugged. "
         "anyway we made it to vegas eventually, 14 hours late, and the hotel had given away our room.")
AI = ("Traveling by bus offers a unique opportunity to experience the landscape at a slower pace. "
      "Additionally, it is often more affordable than other modes of transportation, making it an "
      "attractive option for budget-conscious travelers. However, it is important to note that delays "
      "can occur due to mechanical issues or traffic. In conclusion, while bus travel has its challenges, "
      "it remains a valuable and accessible way to explore new destinations.")
SOURCE = ("The Great Barrier Reef is the world's largest coral reef system, composed of over 2,900 individual "
          "reefs and 900 islands stretching for over 2,300 kilometres. The reef is located in the Coral Sea, "
          "off the coast of Queensland, Australia. It can be seen from outer space and is the world's biggest "
          "single structure made by living organisms.")
ESSAY = ("For my project I studied marine ecosystems. The Great Barrier Reef is the world's largest coral reef "
         "system, composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometres. "
         "This enormous reef sits in the Coral Sea near the Queensland coast of Australia. "
         "I visited an aquarium last summer and saw clownfish for the first time.")


def main(model_dir: str) -> None:
    tmp = tempfile.mkdtemp()
    s = Settings(database_url=f"sqlite:///{tmp}/demo.db", corpus_dir=f"{tmp}/corpus", ai_model_dir=model_dir,
                 admin_token="demo", ai_max_length=256)
    c = TestClient(create_app(s))
    acct = c.post("/admin/accounts", json={"email": "demo@example.com", "credits": 100},
                  headers={"X-Admin-Token": "demo"}).json()
    auth = {"Authorization": f"Bearer {acct['api_key']}"}
    for name, text in [("human-ish", HUMAN), ("ai-ish", AI)]:
        r = c.post("/v1/ai-detect", json={"text": text, "include_segments": True}, headers=auth).json()
        print(name, json.dumps({k: r[k] for k in ("label", "ai_probability", "ai_fraction", "signals", "usage")}))
    c.post("/v1/documents", json={"text": SOURCE, "title": "Reef encyclopedia", "id": "reef"}, headers=auth)
    r = c.post("/v1/plagiarism/check", json={"text": ESSAY, "scope": "private"}, headers=auth).json()
    print("plagiarism", json.dumps({k: r[k] for k in ("similarity_percent", "label", "counts", "usage")}))
    for p in r["passages"]:
        print("  ", p["type"], p["similarity"], "|", p["text"][:90])
    print("account", c.get("/v1/account", headers=auth).json()["credits"], "credits left")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "artifacts/ai-detector")
