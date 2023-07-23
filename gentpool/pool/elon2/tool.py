### Define your custom tool here. Check prebuilts in gentopia.tool (:###
from gentopia.tools import *
from gentopia.tools.gradio_tools.tools import StableDiffusionTool

class ElonDrawing(BaseTool):
    name = "elon_drawing"
    description = "A tool to generate images based on text input, output is a local path where the generated image is stored."
    args_schema: Optional[Type[BaseModel]] = create_model("ElonDrawingArgs", text=(str, ...))

    def _run(self, text: AnyStr) -> Any:
        ans = StableDiffusionTool().run(text)
        return ans

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError