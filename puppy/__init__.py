__all__ = ["MarketEnvironment", "LLMAgent", "RunMode"]


def __getattr__(name):
    if name == "MarketEnvironment":
        from .environment import MarketEnvironment

        return MarketEnvironment
    if name == "LLMAgent":
        from .agent import LLMAgent

        return LLMAgent
    if name == "RunMode":
        from .run_type import RunMode

        return RunMode
    raise AttributeError(f"module 'puppy' has no attribute {name!r}")
