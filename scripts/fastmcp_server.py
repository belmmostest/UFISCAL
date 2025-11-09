"""FastMCP server exposing the DGCE policy simulation tool."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, cast
from pathlib import Path
import sys

from fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgce_model.api.json_serialization import to_json_ready
from dgce_model.api.policy_package_schema import PolicyPackageModel
from dgce_model.orchestration.policy_package import EXAMPLE_PAYLOAD, run_policy_package

# Thread pool for running blocking simulations without blocking the event loop
# max_workers=5 allows up to 5 concurrent policy simulations
_simulation_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="dgce-sim")

server = FastMCP(
    name="dgce-policy-simulator",
    version="1.0.0",
    instructions=(
        "Run comprehensive DGCE corporate tax and VAT policy simulations. "
        "Call the run_policy_package tool with a payload matching the PolicyPackageModel schema to "
        "execute quick, comprehensive, sectoral, strategic, and subpolicy analyses in one shot."
    ),
)


@server.tool(name="run_policy_package", description="Execute the full DGCE policy pipeline")
async def run_policy_tool(payload: PolicyPackageModel | None = None) -> Dict[str, Any]:
    """Invoke the DGCE pipeline with optional parameter overrides.

    This function runs the blocking DGCE simulation in a thread pool to avoid
    blocking the FastMCP server's event loop, allowing multiple concurrent simulations.
    """
    payload_model = payload or PolicyPackageModel()
    payload_dict = payload_model.dict(by_alias=False, exclude_none=True)

    # Run the blocking simulation in a thread pool to prevent event loop blocking
    # This allows the FastMCP server to handle other requests while simulations run
    loop = asyncio.get_event_loop()
    raw_result = await loop.run_in_executor(_simulation_executor, run_policy_package, payload_dict)

    return cast(Dict[str, Any], to_json_ready(raw_result))


@server.resource(
    "resource://dgce/example-payload",
    title="Example DGCE Policy Payload",
    description="Sample payload demonstrating all supported fields for run_policy_package.",
)
def example_payload() -> Dict[str, Any]:
    return EXAMPLE_PAYLOAD


def run() -> None:
    """Run the FastMCP server using stdio transport by default."""

    server.run(transport="http", port=8080)


if __name__ == "__main__":
    run()
