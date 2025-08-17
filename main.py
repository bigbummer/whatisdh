import asyncio
from bot import VKRBot

async def main():
    app = VKRBot()
    await app.on_startup()
    await app.dp.start_polling(app.bot)

if __name__ == "__main__":
    asyncio.run(main())
