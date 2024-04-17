import random
import logging
import numpy as np
import matplotlib.pyplot as plt


class Rākau:
    
    def __init__(self, color, height, location, selected=False):
        self.color = color
        self.height = height
        self.location = location
        self.selected = selected

    def toggle_selection(self):
        self.selected = not self.selected

    def __repr__(self):
        return f"Rākau(color={self.color}, height={self.height}, location={self.location}, selected={self.selected})"


class WorldState:
    
    def __init__(self):
        self.ngā_rākau = []
        self.colors = {}
        self.heights = {}
        self.locations = {}

    def add_rākau(self, rākau):
        self.ngā_rākau.append(rākau)
        self.update(rākau)

    def update(self, rākau):
        if rākau.color in self.colors:
            self.colors[rākau.color] += 1
        else:
            self.colors[rākau.color] = 1

        if rākau.height in self.heights:
            self.heights[rākau.height] += 1
        else:
            self.heights[rākau.height] = 1

        if rākau.location in self.locations:
            self.locations[rākau.location] += 1
        else:
            self.locations[rākau.location] = 1

    def to_dict(self):
        return {
            'ngā_rākau': [
                {'color': rākau.color, 'height': rākau.height, 'location': rākau.location, 'selected': rākau.selected}
                for rākau in self.ngā_rākau
            ]
        }

    @classmethod
    def from_dict(cls, data):
        instance = cls()
        for rākau_data in data.get('ngā_rākau', []):
            rākau = Rākau(
                color=rākau_data['color'],
                height=rākau_data['height'],
                location=rākau_data['location'],
                selected=rākau_data.get('selected', False)
            )
            instance.add_rākau(rākau)
        return instance

    def toggle_rākau_selection(self, index):
        if 0 <= index < len(self.ngā_rākau):
            self.ngā_rākau[index].toggle_selection()

    def toggle_random_selections(self, n):
        if n > len(self.ngā_rākau):
            raise ValueError("Number of toggles requested exceeds the number of sticks available.")
        
        selected_indices = random.sample(range(len(self.ngā_rākau)), n)  # Get n unique random indices
        for index in selected_indices:
            self.ngā_rākau[index].toggle_selection()  # Toggle the selection of each chosen stick

    def generate_rākau_until_budget(self, complexity_budget):
        while True:
            location = random.choice([x for x in range(1, 21) if not x in [rākau.location for rākau in self.ngā_rākau]])
            new_rākau = Rākau(random.choice(['red', 'blue', 'green']), random.randint(1, 10), location)
            self.add_rākau(new_rākau)
            current_complexity = self.calculate_entropy()
            logging.debug(f"Added {new_rākau}. Current complexity (entropy): {current_complexity:.2f}")
            if current_complexity > complexity_budget:
                logging.debug("Complexity budget exceeded.")
                break

    def calculate_entropy(self):
        # Calculate entropy for color, height, and x-coordinate distributions
        color_entropy = -sum((p / len(self.ngā_rākau) * np.log2(p / len(self.ngā_rākau)) for p in self.colors.values()))
        height_entropy = -sum((p / len(self.ngā_rākau) * np.log2(p / len(self.ngā_rākau)) for p in self.heights.values()))
        x_entropy = -sum((p / len(self.ngā_rākau) * np.log2(p / len(self.ngā_rākau)) for p in self.locations.values()))
        return color_entropy + height_entropy + x_entropy

    def draw(self):
        fig, ax = plt.subplots()
        for rākau in self.ngā_rākau:
            x = rākau.location
            height = rākau.height
            color = rākau.color
            edgecolor = 'red' if rākau.selected else 'black'
            linestyle = 'dotted' if rākau.selected else 'solid'
            linewidth = 2 if rākau.selected else 0
            ax.bar(x, height, width=0.8, color=color, edgecolor=edgecolor, linewidth=linewidth, linestyle=linestyle, align='center')

        ax.set_xlim(0, 21)
        ax.set_ylim(0, 11)
        ax.set_xlabel('Location')
        ax.set_ylabel('Height')
        ax.set_title('World State Configuration of ngā_rākau')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.show()