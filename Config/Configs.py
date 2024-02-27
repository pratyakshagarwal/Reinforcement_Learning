from datetime import datetime
import pandas as pd

class Config:
    def __init__(self, obs, hidden_dims, action, epochs, loss_fn='Huber', optimizer='adam', EPSILON=1.0, EPSILON_DECAY=1.005,
                 GAMMA=0.99, MAX_TRANSITIONS=100000, BATCH_SIZE=64, TARGET_UPDATE_AFTER=4, LEARN_AFTER_STEPS=1000):
        self.obs = obs
        self.hidden_dims = hidden_dims
        self.action = action
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.EPSILON = EPSILON
        self.EPSILON_DECAY = EPSILON_DECAY
        self.GAMMA = GAMMA
        self.MAX_TRANSITIONS = MAX_TRANSITIONS
        self.BATCH_SIZE = BATCH_SIZE
        self.TARGET_UPDATE_AFTER = TARGET_UPDATE_AFTER
        self.LEARN_AFTER_STEPS = LEARN_AFTER_STEPS
        
        self.configurations = []

    def save_configration(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config_data = {'Timestamp':timestamp, 'observation':self.obs, 'hidden_dims': self.hidden_dims, 'action':self.action, 'epochs':self.epochs, 'loss_fn':self.loss_fn,
                        'optimizer':self.optimizer, 'EPSIOLON':self.EPSILON, 'EPSILON_DECAY':self.EPSILON_DECAY, 'GAMMA': self.GAMMA, 'MAX_TRANSITIONS':self.MAX_TRANSITIONS,
                          'BATCH_SIZE':self.BATCH_SIZE, 'TARGET_UPDATE_AFTER': self.TARGET_UPDATE_AFTER, 'LEARN_AFTER_STEPS': self.LEARN_AFTER_STEPS}

        # Append the configuration to the list
        self.configurations.append(config_data)

    def get_saved_parameters(self):
        return self.configurations

    def save_to_dataframe_and_csv(self, filename='configurations.csv'):
        """
        Save configurations to a DataFrame and then save the DataFrame to a CSV file.

        Parameters:
        - filename (str): Name of the CSV file (default is 'configurations.csv').
        """
        df = pd.DataFrame(self.configurations)
        df.to_csv(filename, index=False)
        print(f"Configurations saved to {filename}")

    def get_config(self, parameter_value, file_name='configurations.csv'):
        df = pd.read_csv(file_name)  # Use the provided file_name parameter

        return df[parameter_value]


if __name__ == "__main__":
    # Example usage:
    config_instance = Config(obs=10, dim1=20, dim2=10, action=5, epochs=100)
    config_instance.save_configration()
    config_instance.save_to_dataframe_and_csv()
    print(config_instance.get_saved_parameters())
    print(config_instance.get_config('hidden_dim2')[0])