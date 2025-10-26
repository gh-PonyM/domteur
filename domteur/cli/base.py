import asyncio

import aiomqtt
from loguru import logger

from domteur.components.base import MQTTClient
from domteur.config import Settings


def sync_run_client(
    cfg: Settings,
    mqtt_client_class: type[MQTTClient],
    queue_type: type[asyncio.Queue[aiomqtt.Message]] | None = None,
    **client_kwargs,
):
    """Function to call directly inside the click cli function"""
    import asyncio
    import signal

    def handle_sigterm(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, handle_sigterm)
    reconnect_interval = 5

    async def main():
        while True:
            try:
                async with aiomqtt.Client(
                    hostname=cfg.broker_host,
                    port=cfg.broker_port,
                    queue_type=queue_type,
                ) as client:
                    instance = mqtt_client_class(client, settings=cfg, **client_kwargs)
                    await instance.pre_start()
                    async with asyncio.TaskGroup() as tg:
                        for coro in instance.start_coros():
                            tg.create_task(coro, name=coro.__name__)
            except asyncio.CancelledError:
                # Handle cancellation gracefully
                logger.info("Client task cancelled during shutdown")
                raise
            except aiomqtt.MqttError:
                logger.info(
                    f"Connection lost; Reconnecting in {reconnect_interval} seconds ..."
                )
                await asyncio.sleep(reconnect_interval)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutdown requested (SIGINT or SIGTERM). Cleaning up...")
