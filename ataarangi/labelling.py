from ataarangi import WorldState
import matplotlib.pyplot as plt
import random
import json
import os
import uuid


def main():
    num_states = 5  # Define how many states you want to generate and record
    output_folder = 'data/graphs'  # Folder to save graphs
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
    
    for _ in range(num_states):
        world_state = WorldState()
        world_state.generate_random_world_state()
        
        # Generate a unique ID for the current world state
        state_id = str(uuid.uuid4())
        file_path = os.path.join(output_folder, f"{state_id}.png")
        
        # Draw and save the graph
        world_state.draw(file_path)
        
        # Prompt the user for a description
        description = input("Please enter your description for the world state: ")

        # Append the description and ID to a JSON Lines file
        with open('data/world_states.jsonl', 'a') as file:
            record = {
                "id": state_id,
                "description": description,
                "sticks": [{"color": s.color, "height": s.height, "x_coordinate": s.x_coordinate} for s in world_state.sticks],
                "image_path": file_path
            }
            file.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
