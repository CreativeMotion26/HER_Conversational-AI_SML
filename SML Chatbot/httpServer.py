import asyncio
from aiohttp import web
from modulefinder import chatBotModel

chatbot = chatBotModel()

async def handle_chat(request):
  data = await request.json()
  message = data.get("message", "")
  print(f"Received: {message}")

  output = chatbot.generate_response(message)
  print(output)

  return web.json_response({"response": output})

async def main():
  app = web.Application()
  app.router.add_post("/chat", handle_chat)

  runner = web.AppRunner(app)
  await runner.setup()
  site = web.TCPSite(runner, "0.0.0.0", 8888)
  await site.start()

  print("HTTP server running on http://localhost:8888/chat")

  # Keep running indefinitely
  while True:
    await asyncio.sleep(3600)

if __name__ == "__main__":
  asyncio.run(main())