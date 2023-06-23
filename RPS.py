import random
import itertools
import math
import pickle
import os.path

MINCOR = -0.5
MAXCOR = 1.2
def rand(min = MINCOR, max = MAXCOR):
    return random.random()*(MAXCOR-MINCOR)+MINCOR

def sigmoid(x):
    return 1/(1+math.exp(-x))

def add_mul(x, y):
    return sum(i*j for i, j in zip(x, y))

# Model is 30 -> 10 -> 6 -> 3

class NNModel:
    def __init__(self, neurons = None):
        if neurons is None: # We assume 2-2
            neurons = [2, 2]
        self.prop = []
        for x, y in itertools.pairwise(neurons):
            self.prop.append([[rand() for i in range(x)] for j in range(y)])
    
    def eval(self, input_layer):
        self.layers = [input_layer]
        for prop_mat in self.prop:
            next_layer = [sigmoid(add_mul(coefficients, self.layers[-1])) for coefficients in prop_mat]
            self.layers.append(next_layer)
        return next_layer
    
    def learn(self, expectations, strength = 0.1):
        strength *= sum((i-j)**2 for i, j in zip(expectations, self.layers[-1]))/len(expectations)**0.5
        layers = self.layers[:]
        for prop_mat in self.prop[::-1]:
            loss = [i-j for i, j in zip(expectations, layers.pop())]
            prev_layer = layers[-1]
            expectations = prev_layer[:]
            for loss_i, coefficients in zip(loss, prop_mat):
                for i in range(len(coefficients)):
                    coefficients[i] += prev_layer[i]*loss_i*strength
                    expectations[i] += (coefficients[i]*loss_i)/len(loss)
            expectations = [min(1., max(x, 0.)) for x in expectations]
    
    def pickle(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.prop, f)
    def unpickle (self, file):
        with open(file, 'rb') as f:
            self.prop = pickle.load(f)

class RPS_bot:
    """ A Neural network model bot to play Rock-Paper-Scissor."""
    def __init__(self, name="medium", model = None):
        if model is None: model = [30,10,6,3]
        self.nnm = NNModel(model)
        self.backtrack = [0.]*model[0]
        self.filename = "./" + name + ".pkl"
        if os.path.exists(self.filename):
            self.nnm.unpickle(self.filename)
    
    def next(self):
        """ Gets the next response from the bot."""
        weights = self.nnm.eval(self.backtrack)
        reply = ["R", "P", "S"][max((wt*random.random(), i) for i, wt in enumerate(weights))[1]]
        return reply
    
    def respond(self, response, strength = 0.1):
        """ Feed it the opponent's response and make it learn."""
        expectations = {"R": [0.3, 1., 0.], "P": [0., 0.3, 1.], "S":[1., 0., 0.3]}[response]
        self.nnm.learn(expectations, strength)
        self.backtrack = {"R": [1.0, 0., 0.], "P": [0., 1.0, 0.], "S":[0., 0., 1.0]}[response] + self.backtrack[:-3]
    
    def train(self, data, strength = 1.0):
        """ Train from a stream of responses."""
        for response in data:
            self.next()
            self.respond(response, strength)
        
    def write(self):
        self.nnm.pickle(self.filename)

if __name__ == "__main__":
    medium_bot = RPS_bot()
    print("Play Rock Paper Scissors with a bot.")
    print("Enter responses as R, P or S and push enter.")
    print("Enter N to reset scoring, X to exit, Q to quit without training.")
    print("Your first response  :", end = "")
    quitting = False
    win = draw = loss = 0
    while not quitting:
        response = input()[0].upper()
        if response in "XQ":
            quitting = True
            if response == "X":
                medium_bot.write()
        elif response == "N":
            print("Resetting score to [   0|   0|   0]  :", end = "")
            win = draw = loss = 0
        elif response in "RPS":
            bot_response = medium_bot.next()
            probs = medium_bot.nnm.layers[-1]
            medium_bot.respond(response, 0.1)
            if (response, bot_response) in [("R", "S"), ("P", "R"), ("S", "P")]:
                win += 1
                vs = ">"
            elif response == bot_response:
                draw += 1
                vs = "="
            else:
                loss += 1
                vs = "<"
            print(f" {response} {vs} {bot_response} [{win:4d}|{draw:4d}|{loss:4d}]  :", end = "")
            #print(f" [{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}] :", end = "")
        elif response == "T":
            data = "".join(random.choice("RPS") for i in range(30))
            data += "R"*15 + "P"*15 + "S"*15 + "P"*15 + "R"*15 + "S"*15
            data += "RPS"*10 + "PRS"*40 + "RP"*10 + "PS"*10 + "SR"*10
            data += "RPP"*5 + "PSS"*5 + "SRR"*5 + "SSR"*5 + "PPS"*5 + "RRP"*5
            data += "".join(random.choice("RPS") for i in range(15))
            medium_bot.train(data, 0.1)
            print("Did some standard training.")
        else:
            print("Unrecognised response  :", end = "")