class RationalSpeechAgent:

    def __init__(self, world_states, utterances, literal_listener_matrix, prior=None):
        self.world_states = world_states
        self.utterances = utterances
        self.literal_listener_matrix = normalize(np.array(literal_listener_matrix))
        self.prior = prior if prior is not None else np.ones(len(world_states)) / len(world_states)

    def literal_listener(self, utterance_index):
        return self.literal_listener_matrix[:, utterance_index]

    def pragmatic_speaker(self, world_state_index, alpha=1.0):
        utilities = np.array([safe_log(alpha * probability) for probability in self.literal_listener_matrix[world_state_index, :]])
        return normalize(np.exp(utilities))

    def pragmatic_listener(self, utterance_index):
        speaker_matrix = np.array([self.pragmatic_speaker(ws) for ws in range(len(self.world_states))])
        return normalize(np.dot(speaker_matrix.T, self.prior)[utterance_index])


class TeacherAgent(RationalSpeechAgent):
    """
    The main difference between the teacher and the student is that a teacher has fixed beliefs,
    and also tracks the beliefs of the student
    """
    def __init__(self, world_states, utterances, literal_listener_matrix, student_model_matrix, prior=None):
        super().__init__(world_states, utterances, literal_listener_matrix, prior)
        self.student_model_matrix = normalize(np.array(student_model_matrix))  # Teacher's belief about the student's knowledge

    def update_student_model(self, world_state_index, student_utterance_index):
        # Set the probability for the observed utterance and world state to 1
        self.student_model_matrix[world_state_index, :] = 0
        self.student_model_matrix[world_state_index, student_utterance_index] = 1

        # Renormalize every row
        for utterance_index in range(len(self.utterances)):
            nonzeros = 1 * (self.student_model_matrix[:, utterance_index] > 0)
            self.student_model_matrix[:, utterance_index] = nonzeros / np.sum(nonzeros)

    def suggest_world_state(self):
        # Calculate entropy only for rows with all non-zero values
        entropy = lambda p: -np.sum([pi * np.log(pi) for pi in p if pi > 0])
        valid_indices = [index for index, row in enumerate(self.student_model_matrix) if not np.any(row == 0)]

        # If no valid rows exist, pick a state at random
        if not valid_indices:
            return np.random.randint(0, len(self.world_states))

        # Calculate entropy for each valid world state
        student_entropy_values = np.array([entropy(self.student_model_matrix[state_index, :]) if state_index in valid_indices else -np.inf for state_index in range(len(self.world_states))])
        teacher_entropy_values = np.array([entropy(self.literal_listener_matrix[state_index, :]) if state_index in valid_indices else -np.inf for state_index in range(len(self.world_states))])

        # Choose the world state with the highest entropy among the valid ones
        return np.argmin(teacher_entropy_values * student_entropy_values)


class StudentAgent(RationalSpeechAgent):
    def __init__(self, world_states, utterances, literal_listener_matrix, prior=None):
        super().__init__(world_states, utterances, literal_listener_matrix, prior)
        self.known_utterances = set()
        self.attempt_history = set()

    def update_belief(self, world_state_index, observed_utterance_index):
        # Update the probability for the observed utterance and world state
        self.literal_listener_matrix[world_state_index, :] = 0
        self.literal_listener_matrix[world_state_index, observed_utterance_index] = 1

        # Efficiently normalize the matrix
        self.normalize_matrix()

        # Update known utterances based on the updated beliefs
        self.update_known_utterances()

    def normalize_matrix(self):
        for utterance_index in range(len(self.utterances)):
            column_sum = np.sum(self.literal_listener_matrix[:, utterance_index])
            if column_sum > 0:
                self.literal_listener_matrix[:, utterance_index] /= column_sum

    def update_known_utterances(self):
        # Update known utterances by checking columns with unique non-zero entries
        for utterance_index in range(len(self.utterances)):
            if np.any(np.sum(self.literal_listener_matrix[:, utterance_index] != 0) == 1):
                self.known_utterances.add(utterance_index)

    def suggest_world_state(self):
        # Entropy-based selection with an additional constraint to avoid states already correctly answered
        entropy = lambda p: -np.sum([pi * np.log(pi) for pi in p if pi > 0])

        valid_indices = [index for index, row in enumerate(self.literal_listener_matrix)
                         if index not in self.attempt_history or not self.attempt_history[index]
                         and not np.any(row == 0)
                         and all(utterance_index in self.known_utterances for utterance_index, val in enumerate(row) if val > 0)]

        if not valid_indices:
            return np.random.randint(0, len(self.world_states))

        entropy_values = [entropy(self.literal_listener_matrix[state_index, :]) if state_index in valid_indices else -np.inf
                          for state_index in range(len(self.world_states))]

        return np.argmax(entropy_values)

world_states = ['1 rākau', '2 rākau', '3 rākau', '4 rākau', '5 rākau']
utterances = ['Te rākau', 'Ngā rākau', 'He rākau']

# Instantiate teacher and student
teacher = TeacherAgent(
    world_states=world_states,
    utterances=utterances,
    literal_listener_matrix=[
        [1.0, 0.0, 0.0],  # 1 rākau
        [0.0, 0.0, 1.0],  # 2 rākau
        [0.0, 0.0, 1.0],  # 3 rākau
        [0.0, 0.0, 1.0],  # 4 rākau
        [0.0, 1.0, 0.0],  # 5 rākau
    ],
    student_model_matrix=np.full((len(world_states), len(utterances)), 1 / len(world_states))  # Teacher's initial model of the student's knowledge
)

student = StudentAgent(
    world_states=world_states,
    utterances=utterances,
    literal_listener_matrix=np.full((len(world_states), len(utterances)), 1 / len(world_states))  # Student starts with no specific knowledge
)
