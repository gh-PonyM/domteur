import aiomqtt

from domteur.components.base import MQTTClient, start_cli_client
from domteur.config import Settings


def sync_run_client(
    cfg: Settings,
    component: type[MQTTClient],
    component_name: str | None = None,
    **client_kwargs,
):
    import asyncio
    import signal

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    shutdown_event = asyncio.Event()

    def shutdown(signum, frame):
        print("Received shutdown signal, exiting...")
        shutdown_event.set()

    async def main():
        c = aiomqtt.Client(hostname=cfg.broker_host, port=cfg.broker_port)
        await start_cli_client(
            client=c,
            mqtt_client=component,
            settings=cfg,
            name=component_name,
            shutdown_event=shutdown_event,
            **client_kwargs,
        )

    for handler in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(handler, shutdown, handler, None)

    try:
        loop.run_until_complete(main())
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        # Cancel any remaining tasks before closing
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()

        # Wait for tasks to complete cancellation
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
