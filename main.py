from dotenv import load_dotenv
import logging
from typing import Literal, AsyncGenerator
from contextlib import asynccontextmanager,AsyncExitStack

from typing import List
from google.genai import types
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams
from google.adk.sessions import InMemorySessionService
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner

load_dotenv()
# logging.basicConfig(
#     level=logging.DEBUG,  
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.StreamHandler()        
#     ]
# )

logger = logging.getLogger(__name__)

available_ai_providers = Literal["gemini", "openai"]

class MCPClient:
    def __init__(self, ai_provider: str, sse_server_urls: List[str], user_id: str, app_name: str):
        self.ai_provider = ai_provider
        self.app_name = app_name
        self.user_id = user_id
        self.sse_server_urls = sse_server_urls
        self.exit_stacks=[]
        #OpenAI not implemented yet 
        if self.ai_provider.lower() not in ("gemini", "openai"):
            raise ValueError(f"Invalid AI_PROVIDER. Use one of: 'gemini', 'openai'")
        
        self.messages = []
        self.tools = []
        self.exit_stack = AsyncExitStack()

    @asynccontextmanager
    async def session(self):
        await self._load_all_tools()
        try:
            yield
        finally:
            if self.exit_stack:
                await self.exit_stack.aclose()
                logger.info("Cleaned up tool resources with exit_stack.")

    async def _connect_to_server(self, url):
        try:
            connection = SseServerParams(url=url)
            tools, stack = await MCPToolset.from_server(connection_params=connection)
            logger.info(f"Connected to {url} and loaded {len(tools)} tools.")
            return tools, stack
        except Exception as e:
            logger.warning(f"Failed to connect to {url}: {e}")
            return [], None

    async def _load_all_tools(self):
        for url in self.sse_server_urls:
            try:
                connection = SseServerParams(url=url)
                tools, tool_stack = await MCPToolset.from_server(connection_params=connection)

                if tools and tool_stack:
                    self.tools.extend(tools)
                    await self.exit_stack.enter_async_context(tool_stack)

            except Exception as e:
                logger.warning(f"Failed to connect to {url}: {e}")

    async def _prepare_agent(self):
        logger.info("Preparing agent with available tools...")
        self.agent = LlmAgent(
            model='gemini-2.0-flash',
            name='MCP_agent',
            instruction='Help user interact with available tools.',
            tools=self.tools,
        )
        logger.info(f"Agent prepared with {len(self.tools)} tools.")

        session_service = InMemorySessionService()
        session = session_service.create_session(
            state={}, app_name=self.app_name, user_id=self.user_id
        )

        runner = Runner(
            app_name=self.app_name,
            agent=self.agent,
            session_service=session_service
        )
        return runner, session

    async def process_query(self, query: str) -> AsyncGenerator[str, None]:
        logger.info(f"Processing query: {query}")
        try:
            runner, session = await self._prepare_agent()
            content = types.Content(role='user', parts=[types.Part(text=query)])
            events_async = runner.run_async(
                session_id=session.id, user_id=session.user_id, new_message=content
            )
            async for event in events_async:
                if event.content.parts[0].function_response:
                    yield event.content.parts[0].function_response.response['result'].content[0].text
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise



import asyncio
import warnings
warnings.filterwarnings("ignore")
async def main():
    client = MCPClient(
        ai_provider="gemini",
        sse_server_urls=["http://localhost:8000/sse","http://localhost:8001/sse"], 
        user_id="user-123",
        app_name="my-app"
    )

    prompt = "Get a code and get its code phrase"
    print(f"User Query: '{prompt}'")

    async with client.session():
        async for chunk in client.process_query(prompt):
            print(chunk, end="", flush=True) 

if __name__ == "__main__":
    asyncio.run(main())