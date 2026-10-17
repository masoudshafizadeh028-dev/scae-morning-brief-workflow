"""Worker entrypoint for the SCAE Morning Signal Report workflow."""

import asyncio

from mistralai_workflows import discover_workflows, run_worker


async def main() -> None:
    workflows = discover_workflows()
    await run_worker(workflows)


if __name__ == "__main__":
    asyncio.run(main())
