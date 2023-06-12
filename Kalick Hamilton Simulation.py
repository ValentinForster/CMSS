from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mesa.batchrunner import batch_run


class Person(Agent):
    def __init__(self, unique_id, model, sex, attractiveness):
        super().__init__(unique_id, model)
        self.matched = False
        self.sex = sex
        self.attractiveness = attractiveness

    def calculate_acceptance_probability_v1(self, partner_attractiveness):
        p1 = ((partner_attractiveness ** 3 / 1000) ** ((51 - self.model.date_number) / 50))
        return p1

    def calculate_acceptance_probability_v2(self, partner_attractiveness):
        p2_temp = (((10 - abs(self.attractiveness - partner_attractiveness)) ** 3) / 1000)
        p2 = (p2_temp ** ((51 - self.model.date_number) / 50))
        return p2

    def step(self):
        if not self.matched:
            potential_matches = [
                agent
                for agent in self.model.schedule.agents
                if isinstance(agent, Person) and not agent.matched and agent.sex != self.sex
            ]
            if potential_matches:
                match = random.choice(potential_matches)
                p1 = self.calculate_acceptance_probability_v1(match.attractiveness) #################################
                p2 = match.calculate_acceptance_probability_v1(self.attractiveness) #################################
                if random.random() < p1 and random.random() < p2:
                    self.matched = True
                    match.matched = True
                    self.model.num_matches += 1
                    self.model.attractiveness_sum += self.attractiveness + match.attractiveness
                    self.model.attractiveness_pairs.append((self.attractiveness, match.attractiveness))


class MatchingModel(Model):
    def __init__(self, N):
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.num_matches = 0
        self.date_number = 1
        self.attractiveness_sum = 0
        self.attractiveness_pairs = []

        for i in range(self.num_agents):
            attractiveness = random.randint(1, 10)
            sex = "male" if i < self.num_agents // 2 else "female"
            person = Person(i, self, sex, attractiveness)
            self.schedule.add(person)

        self.datacollector = DataCollector(
            model_reporters={"Num Matches": "num_matches"},
            agent_reporters={"Matched": "matched"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.date_number += 1

    def get_mean_attractiveness(self):
        if self.num_matches > 0:
            return self.attractiveness_sum / (2 * self.num_matches)
        else:
            return 0

    def get_cumulative_correlation(self):
        if self.num_matches > 0:
            x = np.array([pair[0] for pair in self.attractiveness_pairs])
            y = np.array([pair[1] for pair in self.attractiveness_pairs])
            correlation = np.corrcoef(x, y)[0, 1]
            return correlation
        else:
            return 0


# Set the number of agents
num_agents = 1000

# Set the maximum number of steps and initialize the model
max_steps = 50
model = MatchingModel(num_agents)

# Lists to store the percentage of agents matched, cumulative mean attractiveness, and cumulative correlation
percentage_matched = []
cumulative_mean_attractiveness = []
cumulative_correlation = []

# Run the simulation until all individuals become part of a couple or the maximum number of steps is reached
while model.num_matches < num_agents / 2 and model.date_number <= max_steps:
    model.step()
    percentage = (model.num_matches / (num_agents / 2)) * 100
    mean_attractiveness = model.get_mean_attractiveness()
    correlation = model.get_cumulative_correlation()
    percentage_matched.append(percentage)
    cumulative_mean_attractiveness.append(mean_attractiveness)
    cumulative_correlation.append(correlation)

# Accessing the data collector
data = model.datacollector.get_model_vars_dataframe()

# Print the number of matches over time
print(data["Num Matches"])

# Print the successful couples
print("Successful Couples:")
for agent in model.schedule.agents:
    if isinstance(agent, Person) and agent.matched:
        print(f"Agent {agent.unique_id} is matched.")

# Plotting the results
fig, ax1 = plt.subplots()

# Plotting the relationship between matched agents and cumulative mean attractiveness
ax1.plot(percentage_matched, cumulative_mean_attractiveness, color="b")
ax1.set_xlabel("Percentage of Agents Matched")
ax1.set_ylabel("Cumulative Mean Attractiveness", color="b")
ax1.tick_params(axis="y", labelcolor="b")

# Creating a second y-axis for the cumulative correlation
ax2 = ax1.twinx()
ax2.plot(percentage_matched, cumulative_correlation, color="r")
ax2.set_ylabel("Cumulative Intracouple Attractiveness Correlation", color="r")
ax2.tick_params(axis="y", labelcolor="r")

plt.title("Relationship between Matched Agents and Cumulative Mean Attractiveness")
plt.show()
