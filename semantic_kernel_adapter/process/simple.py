# Copyright (c) Microsoft. All rights reserved.

import asyncio
from enum import Enum
from typing import ClassVar

from pydantic import Field

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.chat_completion_client_base import (
    ChatCompletionClientBase,
)
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel.processes.kernel_process.kernel_process_step import (
    KernelProcessStep,
)
from semantic_kernel.processes.kernel_process.kernel_process_step_context import (
    KernelProcessStepContext,
)
from semantic_kernel.processes.kernel_process.kernel_process_step_state import (
    KernelProcessStepState,
)
from semantic_kernel.processes.local_runtime.local_event import KernelProcessEvent
from semantic_kernel.processes.local_runtime.local_kernel_process import start
from semantic_kernel.processes.process_builder import ProcessBuilder


class StartStep(KernelProcessStep):
    @kernel_function
    async def print_intro_message(self):
        print("Step 1 - Start\n")


class DoSomeWorkStep(KernelProcessStep):
    @kernel_function
    async def print_intro_message(self):
        print("Step 2 - Doing Some Work...\n")


class DoMoreWorkStep(KernelProcessStep):
    @kernel_function
    async def print_intro_message(self):
        print("Step 3 - Doing Yet More Work...\n")


class LastStep(KernelProcessStep):
    @kernel_function
    async def print_intro_message(self):
        print("Step 4 - This is the Final Step...\n")


class ProcessEvents(Enum):
    StartProcess = "StartProcess"


async def step01_processes(kernel: Kernel, scripted: bool = True):
    process = ProcessBuilder(name="ChatBot")

    start_step = process.add_step(StartStep)
    do_some_work_step = process.add_step(DoSomeWorkStep)
    do_more_work_step = process.add_step(DoMoreWorkStep)
    last_step = process.add_step(LastStep)

    process.on_input_event(event_id=ProcessEvents.StartProcess).send_event_to(
        target=start_step
    )

    start_step.on_function_result(
        function_name=StartStep.print_intro_message.__name__
    ).send_event_to(target=do_some_work_step)

    do_some_work_step.on_function_result(
        function_name=DoSomeWorkStep.print_intro_message.__name__
    ).send_event_to(target=do_more_work_step)

    # For the response step, send the response back to the user input step
    do_more_work_step.on_function_result(
        function_name=DoMoreWorkStep.print_intro_message.__name__
    ).send_event_to(target=last_step)

    last_step.on_function_result(
        function_name=LastStep.print_intro_message.__name__
    ).stop_process()

    # Build the kernel process
    kernel_process = process.build()

    # Start the process
    await start(
        process=kernel_process,
        kernel=kernel,
        initial_event=KernelProcessEvent(id=ProcessEvents.StartProcess, data=None),
    )


if __name__ == "__main__":
    from ..service import build_kernel_pipeline

    kernel = build_kernel_pipeline()

    asyncio.run(step01_processes(kernel=kernel, scripted=False))
