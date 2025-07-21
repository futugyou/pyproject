from fastapi import APIRouter, Depends
from app.dependencies import get_kernel

router = APIRouter(prefix="/sk")


@router.get("/base")
async def list_earnings(kernel=Depends(get_kernel)):
    plugins_directory = "prompt_template_samples"
    funFunctions = kernel.add_plugin(
        parent_directory=str(plugins_directory), plugin_name="FunPlugin"
    )

    jokeFunction = funFunctions["Joke"]
    result = await kernel.invoke(
        jokeFunction, input="travel to dinosaur age", style="silly"
    )
    return result
