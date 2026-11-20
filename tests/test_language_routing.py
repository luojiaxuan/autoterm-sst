import asyncio

from framework.agent import Agent, SessionInfo
from framework.config import _language_servers, language_agent_key
from framework.router import AgentRouter


class Recorder(Agent):
    def __init__(self, pair):
        self.pair = pair
        self.opened = []

    def describe(self):
        return {"models": [{"id": "RASST"}], "language_pairs": [{"id": self.pair, "available": True}]}

    async def start(self, emit):
        pass

    async def open_session(self, session_id, config):
        self.opened.append(session_id)
        return SessionInfo(meta={"language_pair": self.pair})

    async def submit_audio(self, session_id, pcm, *, final=False):
        pass

    async def close_session(self, session_id):
        pass


def test_language_servers_parse(monkeypatch):
    monkeypatch.setenv(
        "RASST_FRAMEWORK_LANGUAGE_SERVERS",
        "English -> Chinese=http://127.0.0.1:18700|zh; English -> German=http://127.0.0.1:18612",
    )
    assert _language_servers() == {
        "English -> Chinese": ("http://127.0.0.1:18700", "zh"),
        "English -> German": ("http://127.0.0.1:18612", None),
    }


def test_session_goes_to_agent_of_its_language_pair():
    agents = {language_agent_key("RASST", p): Recorder(p) for p in ("English -> Chinese", "English -> German")}
    router = AgentRouter(agents, default_agent=language_agent_key("RASST", "English -> Chinese"))
    name, info = asyncio.run(router.open_session({"agent_type": "RASST", "language_pair": "English -> German"}))
    assert info.meta["language_pair"] == "English -> German"
    assert agents[language_agent_key("RASST", "English -> German")].opened
    assert not agents[language_agent_key("RASST", "English -> Chinese")].opened


def test_config_lists_every_language_pair_once():
    agents = {language_agent_key("RASST", p): Recorder(p) for p in ("English -> Chinese", "English -> German")}
    router = AgentRouter(agents, default_agent=language_agent_key("RASST", "English -> Chinese"))
    config = router.config()
    assert [m["id"] for m in config["models"]] == ["RASST"]
    assert {lang["id"] for lang in config["language_pairs"]} == {"English -> Chinese", "English -> German"}
