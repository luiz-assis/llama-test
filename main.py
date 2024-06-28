from tabnanny import verbose
from urllib import response
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process
from langsmith import expect
from sympy import true
from types import SimpleNamespace
import json


class CrewRunner:
    def run_crew(self, review):
        model = Ollama(model="llama3", base_url="http://localhost:11434")
        cr_agent = Agent(
            role="customer relationship analyst",
            goal="understand in which score the sentiment of a customer's review fits, must be in one of those: 'very positive', 'positive', 'neutral', 'negative' or 'very negative",
            backstory="you are an expert in customer relationship analysis and will help the owner of a restaurant to score the reviews done by customers and to have a better understanding of their satisfaction level.",
            verbose=True,
            allow_delegation=False,
            llm=model,
        )

        pr_agent = Agent(
            role="public relationship analyst",
            goal="generate a proper response to the customer's review; it must show gratitude or apology accordingly. It shall not promise any action to regain trust, just apologize and say we are working to make it better.",
            backstory="you are an public relationship analyst and will help a restaurant owner to write a response to their customer's reviews.",
            verbose=True,
            allow_delegation=False,
            llm=model,
        )

        bi_agent = Agent(
            role="business intelligence analyst",
            goal="analyse the review and extract the good and bad points informed by the customer. For example: 'the food was great, but the service was terrible' review would result in this categorization => good: 'food', bad: 'service'.",
            backstory="you are an business intelligence analyst and will help a restaurant owner to understand the good and bad points of the service provided by its restaurant, he wants to know what is going well and what needs to be improved.",
            verbose=True,
            allow_delegation=False,
            llm=model,
        )

        classify_review_task = Task(
            name="classify review",
            agent=cr_agent,
            description=f"classify the review: <<{review}>> as one of the following: 'very positive', 'positive', 'neutral', 'negative' or 'very negative'",
            expected_output="one of the following string: 'very positive', 'positive', 'neutral', 'negative' or 'very negative'. The output must be only one of those options, any text beyond one of the option.",
        )

        respond_to_review_task = Task(
            description=f"write a response to the customer review: <<{review}>> apologizing or acknowledging accordingly to the level of satisfaction; no more than 400 characters.",
            agent=pr_agent,
            expected_output="a short and polite response to the review, no more than 400 characters.",
        )

        classify_strong_weak_points_task = Task(
            description=f"given the review: <<{review}>> analise it and extract the good and bad points about the service. Then create a JSON object with the follow schema <<{{good:['food', 'service'], bad: ['ambient','loud music']}}>> and put the extracted points in one of the arrays (good or bad). VERY IMPORTANT! Don't return more text than that. The output must be only the JSON object.",
            agent=bi_agent,
            expected_output="a JSON object with the follow schema <<{{good:['food', 'service'], bad: ['ambient','loud music']}}>>",
        )

        crew = Crew(
            agents=[cr_agent, pr_agent, bi_agent],
            tasks=[
                classify_review_task,
                respond_to_review_task,
                classify_strong_weak_points_task,
            ],
            verbose=2,
            process=Process.sequential,
        )

        crew.kickoff()

        output = extract_first_json_object(
            classify_strong_weak_points_task.output.raw_output
        )

        anon_obj = SimpleNamespace(
            sentiment=classify_review_task.output.raw_output,
            response=respond_to_review_task.output.raw_output,
            good_points=output.good,
            bad_points=output.bad,
        )

        return json.dumps(vars(anon_obj), indent=4)


def extract_first_json_object(s):
    depth = 0
    start_index = None
    for i, char in enumerate(s):
        if char == "{":
            depth += 1
            if start_index is None:
                start_index = i
        elif char == "}":
            depth -= 1
            if depth == 0 and start_index is not None:
                try:
                    # Attempt to parse the JSON substring
                    json_obj = json.loads(
                        s[start_index : i + 1],
                        object_hook=lambda d: SimpleNamespace(**d),
                    )
                    return json_obj
                except json.JSONDecodeError:
                    # If parsing fails, reset and continue looking for a valid JSON object
                    start_index = None
    # Return None if no valid JSON object is found
    return None
