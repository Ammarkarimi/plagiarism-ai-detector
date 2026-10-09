"""``plagai`` command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys

COMMANDS = {
    "prepare-ai-data": ("plagai.ai.data", "Download and build the human-vs-AI dataset"),
    "train-ai": ("plagai.ai.train", "Fine-tune the AI-text classifier"),
    "fit-ensemble": ("plagai.ai.ensemble", "Fit the stacker and calibrate thresholds"),
    "train-paraphrase": ("plagai.plagiarism.train_paraphrase", "Fine-tune the paraphrase encoder"),
    "index-corpus": ("plagai.plagiarism.corpus", "Index reference documents for plagiarism checks"),
}


def _admin(argv: list[str]) -> None:
    from plagai.api import billing
    from plagai.api.db import make_session_factory
    from plagai.config import get_settings

    p = argparse.ArgumentParser(prog="plagai admin")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create-account")
    c.add_argument("email")
    c.add_argument("--name", default="")
    c.add_argument("--credits", type=int, default=0)
    k = sub.add_parser("issue-key")
    k.add_argument("account_id", type=int)
    k.add_argument("--name", default="default")
    g = sub.add_parser("add-credits")
    g.add_argument("account_id", type=int)
    g.add_argument("amount", type=int)
    g.add_argument("--reason", default="purchase")
    g.add_argument("--reference", default="")
    a = p.parse_args(argv)

    Session = make_session_factory(get_settings().database_url)
    with Session.begin() as db:
        if a.cmd == "create-account":
            acct = billing.create_account(db, a.email, a.name, a.credits)
            key, plain = billing.issue_key(db, acct.id)
            out = {"account_id": acct.id, "api_key": plain, "credits": a.credits}
        elif a.cmd == "issue-key":
            key, plain = billing.issue_key(db, a.account_id, a.name)
            out = {"key_id": key.id, "api_key": plain}
        else:
            out = {"credits": billing.grant(db, a.account_id, a.amount, a.reason, a.reference)}
    print(json.dumps(out, indent=2))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: plagai <command> [args]\n\ncommands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:<18} {desc}")
        print(f"  {'admin':<18} Accounts, API keys and credits (create-account | issue-key | add-credits)")
        print(f"  {'serve':<18} Run the HTTP API (uvicorn)")
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "admin":
        return _admin(rest)
    if cmd == "serve":
        import uvicorn
        p = argparse.ArgumentParser(prog="plagai serve")
        p.add_argument("--host", default="0.0.0.0")
        p.add_argument("--port", type=int, default=8000)
        p.add_argument("--workers", type=int, default=1)
        a = p.parse_args(rest)
        uvicorn.run("plagai.api.app:app", host=a.host, port=a.port, workers=a.workers)
        return
    if cmd not in COMMANDS:
        sys.exit(f"unknown command: {cmd}")
    import importlib
    importlib.import_module(COMMANDS[cmd][0]).main(rest)


if __name__ == "__main__":
    main()
