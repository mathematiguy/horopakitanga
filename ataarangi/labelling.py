import os
import json
import uuid
import click
import random
import logging
from ataarangi import WorldState
import matplotlib.pyplot as plt


# Function to handle description confirmation
def confirm_description(description):
    # Show the entered description and prompt for confirmation
    confirm = input(f"Confirm description '{description}' (hit enter to confirm or type to edit): ")
    if confirm == '':  # If user hits enter, confirm the description
        return description
    else:  # If user types anything, ask for a new description
        return confirm_description(input("Please enter a new description: "))  # Recursive call for new description

@click.command()
@click.option("--budget", default=0.0, help="Entropy budget")
@click.option("--log_level", default="INFO", help="Log level (default: INFO)")
def main(budget, log_level):

    # Set logger config
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    num_states = 20  # Define how many states you want to generate and record
    output_folder = 'data/graphs'  # Folder to save graphs
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists

    while budget < 8.0:
        for _ in range(num_states):
            world_state = WorldState()
            world_state.generate_rākau_until_budget(budget)

            # Generate a unique ID for the current world state
            state_id = str(uuid.uuid4())
            file_path = os.path.join(output_folder, f"{state_id}.png")

            # Draw and check if all sticks are selected
            world_state.draw()
            all_selected = all(rākau.selected for rākau in world_state.ngā_rākau)

            # Set automatic description or prompt the user based on selection
            if all_selected:
                if len(world_state.ngā_rākau) > 1:
                    description = "ngā rākau"  # Automatically set description for all selected sticks
                else:
                    description = "te rākau"
            else:
                description = input("Please enter your description for the world state: ")

            # Use the function to handle description entry and confirmation
            if not description in ['ngā rākau', 'te rākau']:
                description = confirm_description(description)

            # Show final accepted description
            print(f"Final confirmed description: {description}")

            world_state.save(file_path)

            # Append the description and ID to a JSON Lines file
            with open('data/world_states.jsonl', 'a', encoding='utf-8') as file:
                record = {
                    "id": state_id,
                    "description": description,
                    "sticks": world_state.to_dict(),
                    "entropy": world_state.calculate_entropy(),
                    "image_path": file_path
                }
                file.write(json.dumps(record) + "\n")

        budget += 0.5


if __name__ == "__main__":
    main()
