from typing import cast

from google.adk.agents import BaseAgent
from google.adk.runners import InMemoryRunner
from google.adk.sessions.session import Session
from google.genai import types


async def run_agent(base_agent: BaseAgent, message: str):
    app_name = "my_app"
    user_id_1 = "user1"
    runner = InMemoryRunner(
        app_name=app_name,
        agent=base_agent,
    )

    async def run_prompt(session: Session, new_message: str) -> Session:
        content = types.Content(
            role="user", parts=[types.Part.from_text(text=new_message)]
        )
        print("** User says:", content.model_dump(exclude_none=True))
        async for event in runner.run_async(
            user_id=user_id_1,
            session_id=session.id,
            new_message=content,
        ):
            if not event.content or not event.content.parts:
                continue
            if event.content.parts[0].text:
                print(f"** {event.author}: {event.content.parts[0].text}")
            elif event.content.parts[0].function_call:
                print(
                    f"** {event.author}: fc /"
                    f" {event.content.parts[0].function_call.name} /"
                    f" {event.content.parts[0].function_call.args}\n"
                )
            elif event.content.parts[0].function_response:
                print(
                    f"** {event.author}: fr /"
                    f" {event.content.parts[0].function_response.name} /"
                    f" {event.content.parts[0].function_response.response}\n"
                )

        return cast(
            Session,
            await runner.session_service.get_session(
                app_name=app_name, user_id=user_id_1, session_id=session.id
            ),
        )

    session_1 = await runner.session_service.create_session(
        app_name=app_name, user_id=user_id_1
    )

    print(f"----Session to create memory: {session_1.id} ----------------------")
    session_1 = await run_prompt(session_1, message)
