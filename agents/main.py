import sys
import os
from dotenv import load_dotenv
from agent import ReActAgent
from tools import get_github_repo_info, get_github_repo_prs, get_weather
from rich.console import Console

# Load environment variables
load_dotenv()

def main():
    console = Console()
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        console.print("[bold red]Error: No API key found. Please set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable.[/bold red]")
        sys.exit(1)
        
    # Instantiate the generic agent, passing the required tools dynamically
    agent = ReActAgent(tools=[get_github_repo_info, get_github_repo_prs, get_weather])
    
    # If a query is passed as command line argument, execute it and exit
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        try:
            agent.run(query)
        except Exception as e:
            console.print(f"[bold red]Execution error:[/bold red] {str(e)}")
        return

    # Otherwise enter interactive mode (REPL)
    console.print("\n[bold green]=========================================[/bold green]")
    console.print("[bold yellow]      Pure Python ReAct Agent CLI        [/bold yellow]")
    console.print("[bold green]=========================================[/bold green]")
    console.print("Type your questions below. Enter [bold red]'exit'[/bold red] or [bold red]'quit'[/bold red] to end the session.\n")
    
    while True:
        try:
            query = input("Ask the agent > ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit']:
                console.print("\n[bold green]Goodbye![/bold green]")
                break
                
            agent.run(query)
            console.print("-" * 50)
        except KeyboardInterrupt:
            console.print("\n[bold green]Goodbye![/bold green]")
            break
        except Exception as e:
            console.print(f"[bold red]Error during run:[/bold red] {str(e)}\n")

if __name__ == "__main__":
    main()
