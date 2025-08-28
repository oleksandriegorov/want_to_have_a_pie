import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from want_to_have_a_pie.tools.vision_tool import VisionTool
from crewai_tools import SerperDevTool
from crewai_tools import FirecrawlScrapeWebsiteTool


# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

os.environ["SERPER_API_KEY"] = "XXXXXXXXXXXX"  # serper.dev API key
os.environ["FIRECRAWL_API_KEY"] = "YYYYYYYYY"

# Initialize the Firecrawl tool without a specific URL
# This allows the agent to scrape any URL provided in the task context
scrape_tool = FirecrawlScrapeWebsiteTool(
    # No url parameter - agent will specify URLs dynamically
    api_key=os.getenv("FIRECRAWL_API_KEY"),  # Optional if set in env
    page_options={
        "onlyMainContent": True,  # Get clean content without headers/footers
        "includeHtml": False,  # Return markdown only
    },
    extractor_options={
        "mode": "llm-extraction",  # Use AI extraction for better results
        "extractionPrompt": "Extract the main content, key information, and important details from this webpage",
    },
    timeout=30000,  # 30 second timeout
)


@CrewBase
class WantToHaveAPie:
    """WantToHaveAPie crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def vision_food_estimator(self) -> Agent:
        return Agent(
            config=self.agents_config["vision_food_estimator"],  # type: ignore[index]
            verbose=True,
            tools=[VisionTool()],  # Add our custom vision tool
            llm="gpt-4.1-mini",  # Use vision-capable model
            max_iter=3,
        )

    @agent
    def recipe_hunter(self) -> Agent:
        return Agent(
            config=self.agents_config["recipe_hunter"],  # type: ignore[index]
            verbose=True,
            tools=[SerperDevTool()],  # Use CrewAI's search tool
            # llm="gpt-4o-mini",  # Use vision-capable model
            max_iter=3,
        )

    @agent
    def choose_recipe_and_ingredients(self) -> Agent:
        return Agent(
            config=self.agents_config["choose_recipe_and_ingredients"],  # type: ignore[index]
            verbose=True,
            tools=[scrape_tool],  # Add our custom vision tool
            llm="gpt-4.1-mini",  # Use vision-capable model
            max_iter=3,
        )

    @agent
    def html_formatter(self) -> Agent:
        return Agent(
            config=self.agents_config["html_formatter"],  # type: ignore[index]
            verbose=True,
            tools=[scrape_tool],  # Add search tool for finding recipe images
            llm="gpt-4.1-mini",  # Use GPT-4o-mini for HTML generation
            max_iter=3,
        )

    # @agent
    # def reporting_analyst(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['reporting_analyst'], # type: ignore[index]
    #         verbose=True
    #     )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def food_estimation_task(self) -> Task:
        return Task(
            config=self.tasks_config["food_estimation_task"],  # type: ignore[index]
        )

    @task
    def find_recipe_task(self) -> Task:
        return Task(
            config=self.tasks_config["find_recipe_task"],  # type: ignore[index]
        )

    @task
    def find_ingredients_task(self) -> Task:
        return Task(
            config=self.tasks_config["find_ingredients_task"],  # type: ignore[index],
        )

    @task
    def html_format_task(self) -> Task:
        return Task(
            config=self.tasks_config["html_format_task"],  # type: ignore[index]
            output_file="recipe_recommendations.html",
        )

    # @task
    # def reporting_task(self) -> Task:
    #    return Task(
    #        config=self.tasks_config['reporting_task'], # type: ignore[index]
    #        output_file='report.md'
    #    )

    @crew
    def crew(self) -> Crew:
        """Creates the WantToHaveAPie crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
