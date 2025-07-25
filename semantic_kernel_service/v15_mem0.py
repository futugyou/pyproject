from mem0 import MemoryClient
import asyncio


async def mem0_demo(memoryClient: MemoryClient) -> str:
    messages = [
        {
            "role": "user",
            "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts.",
        },
        {
            "role": "assistant",
            "content": "Hello Alex! I see that you're a vegetarian with a nut allergy.",
        },
    ]

    memoryClient.add(messages, user_id="alex")
    query = "What can I cook for dinner tonight?"
    result = memoryClient.search(query, user_id="alex")

    return str(result)


if __name__ == "__main__":

    async def main():
        from service import memoryClient

        result = await mem0_demo(memoryClient)
        print(result)

    asyncio.run(main())
