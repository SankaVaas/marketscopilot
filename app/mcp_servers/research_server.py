"""
Example MCP server exposing MarketsCopilot's retrieval and entitlement checks
as standard MCP tools, so any MCP-compatible client (Claude Desktop, other
agents, IDE assistants) can call into this platform the same way they'd call
any other connector.

MCP is a free, open protocol -- there is no commercial dependency here.

Run standalone with:
    python -m app.mcp_servers.research_server
"""
from mcp.server.fastmcp import FastMCP

from app.retrieval.hybrid import hybrid_search
from app.security.rbac import allowed_classifications, is_authorized

mcp = FastMCP("marketscopilot-research")


@mcp.tool()
def search_research(query: str, role: str = "public_reader", top_k: int = 5) -> list[dict]:
    """Search internal market research and compliance documents.

    Args:
        query: Natural-language research question.
        role: Requesting user's role, used to enforce entitlements
              (front_office_analyst, compliance_officer, public_reader).
        top_k: Max number of chunks to return.
    """
    allowed = allowed_classifications(role)
    results = hybrid_search(query, top_k=top_k, allowed_classifications=allowed)
    return [
        {
            "source_doc": r["source_doc"],
            "classification": r["classification"],
            "snippet": r["text"][:300],
        }
        for r in results
    ]


@mcp.tool()
def check_entitlement(role: str, classification: str) -> bool:
    """Check whether a given role is entitled to view a document classification."""
    return is_authorized(role, classification)


if __name__ == "__main__":
    mcp.run()
