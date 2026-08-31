import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console
from rich.panel import Panel

# Load environment variables
load_dotenv()

# --- GENERIC REACT SYSTEM PROMPT ---

SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tools_description}

You MUST use the following format:

Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (do NOT quote the input, pass it as raw text)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat multiple times as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {question}
Thought:"""


class ReActAgent:
    def __init__(self, tools: list = None, model_name: str = "gemini-2.5-flash"):
        """
        Initializes the generic ReAct agent.
        
        Args:
            tools (list): A list of callable tool functions.
            model_name (str): The name of the Gemini model to use.
        """
        if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" in os.environ:
            os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
            
        self.model_name = model_name
        self.client = genai.Client()
        
        # Dynamically build tools registry and prompt descriptions
        self.tools = {}
        self.tools_description = "No tools available."
        self.tool_names = "None"
        
        if tools:
            for tool_func in tools:
                name = tool_func.__name__
                self.tools[name] = tool_func
            
            self.tool_names = ", ".join(self.tools.keys())
            
            # Introspect function docstrings to generate system prompt tool descriptions
            desc_lines = []
            for name, func in self.tools.items():
                doc = func.__doc__.strip() if func.__doc__ else "No description available."
                # Clean up docstring multi-line spacing
                doc_clean = " ".join([line.strip() for line in doc.split("\n")])
                desc_lines.append(f"- {name}: {doc_clean}")
            self.tools_description = "\n".join(desc_lines)

    def run(self, question: str, max_steps: int = 8):
        console = Console()
        console.print(Panel(f"[bold yellow]Question:[/bold yellow] {question}", title="Generic ReAct Agent Initialized", border_style="blue"))
        
        # Dynamically format the system prompt with the registered tools
        prompt = SYSTEM_PROMPT.format(
            question=question,
            tools_description=self.tools_description,
            tool_names=self.tool_names
        )
        
        for step in range(1, max_steps + 1):
            console.print(f"\n[bold magenta]── Step {step} ──[/bold magenta]")
            
            # Request LLM generation, stopping immediately before 'Observation:'
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    stop_sequences=["Observation:"],
                    temperature=0.0
                )
            )
            
            llm_output = response.text
            # Print thoughts and planned action
            console.print(Panel(llm_output.strip(), title=f"Agent Thoughts & Action proposal", border_style="cyan"))
            
            # Feed current thoughts to prompt
            prompt += llm_output
            
            # Check if final answer is achieved
            if "Final Answer:" in llm_output:
                final_answer = llm_output.split("Final Answer:")[-1].strip()
                console.print(Panel(final_answer, title="[bold green]Final Answer[/bold green]", border_style="green"))
                return final_answer
                
            # Parse proposed Action & Input
            action_match = re.search(r"Action:\s*(\w+)", llm_output)
            input_match = re.search(r"Action Input:\s*(.+)", llm_output)
            
            if action_match and input_match:
                action_name = action_match.group(1).strip()
                action_input = input_match.group(1).strip().strip("'\"")
                
                # Check tool support
                if action_name in self.tools:
                    console.print(f"[yellow]Executing Tool:[/yellow] [bold cyan]{action_name}[/bold cyan] with input [bold cyan]'{action_input}'[/bold cyan]...")
                    tool_func = self.tools[action_name]
                    observation = tool_func(action_input)
                    
                    # Print results
                    console.print(Panel(observation.strip(), title="Observation Result", border_style="green"))
                    
                    # Append result back to the prompt
                    prompt += f"\nObservation: {observation}\nThought: "
                else:
                    observation = f"Error: Tool '{action_name}' is not supported. Supported: {list(self.tools.keys())}."
                    console.print(f"[bold red]{observation}[/bold red]")
                    prompt += f"\nObservation: {observation}\nThought: "
            else:
                # Fallback if agent generated text but lost formatting
                console.print("[yellow]Warning: Agent deviated from ReAct structure. Terminating early and using output as final response.[/yellow]")
                console.print(Panel(llm_output.strip(), title="Output Response", border_style="green"))
                return llm_output.strip()
                
        console.print("[bold red]Error: Limit of reasoning steps reached without finding a Final Answer.[/bold red]")
        return "Reasoning step limit reached."
