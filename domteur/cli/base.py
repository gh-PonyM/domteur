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

    def shutdown():
        print("Received shutdown signal, exiting...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    async def main():
        c = aiomqtt.Client(hostname=cfg.broker_host, port=cfg.broker_port)
        await start_cli_client(
            client=c,
            mqtt_client=component,
            settings=cfg,
            name=component_name,
            **client_kwargs,
        )

    loop = asyncio.new_event_loop()
    for handler in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(handler, shutdown)
    try:
        loop.run_until_complete(main())
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        shutdown()
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
