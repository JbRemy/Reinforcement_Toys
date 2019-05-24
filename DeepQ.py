class deep_Q(torch.nn.Module):
    def __init__(self, units, lr):
        super(deep_Q, self).__init__()
        input_size = 8
        output_size = 4
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_size, out_features=units),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=units, out_features=units),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=units, out_features=units),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=units, out_features=output_size)
        )
        self.criterion = torch.nn.MSELoss()
        self.optimizeer = torch.optim.Adam(self.layers.parameters(), lr=lr)
        
    def forward(self, x):
        x = self.layers(x)
        return x
    
    def __call__(self, observation, item=True):
        x = torch.FloatTensor(observation)
        x = self.forward(x)
        return x.tolist()
    
    def train(self, dataset, batch_size=32, n_epochs=1):
        loss_ = []
        for x, y in dataset.get_batch(batch_size, n_epochs):
            self.optimizeer.zero_grad()
            x = torch.FloatTensor(x)
            y = torch.FloatTensor(y)
            x = self.forward(x)
            loss = self.criterion(x, y)
            loss_.append(loss.item())
            loss.backward()
            self.optimizeer.step()
                      
def policy_epsilon_greedy(state, action_space, Q, epsilon, return_max=False, return_vals=False):
    vals = Q(state)
    if return_max:
        return np.max(vals)
    
    else:
        if np.random.uniform() < 1-epsilon:
            if return_vals:
                return np.argmax(vals), vals
            
            else:
                return np.argmax(vals)
        else:
            return np.random.choice([action for action in action_space if not action==np.argmax(vals)]), vals
