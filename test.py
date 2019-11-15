from tabular_model import tabular_mcts,torch_policy_value_model

# The main recursive method
# to print all possible
# strings of length k
def printAllKLengthRec(characters, prefix, n, k):
    global boards

    # Base case: k is 0,
    # print prefix
    if (k == 0) :
        boards.append(prefix)
        return prefix

    # One by one add all characters
    # from set and recursively
    # call for k equals to k-1
    for i in range(n):

        # Next character of input added
        newPrefix = prefix + characters[i]

        # k is decreased, because
        # we have added a new character
        result = printAllKLengthRec(characters, newPrefix, n, k - 1)

if __name__ == "__main__":
    global boards
    boards = []
    policy_value_model = None
    #if(path.exists("value_model.torch")):
    #    value_model = torch.load("value_model.torch")
    #if(path.exists("policy_model.torch")):
    #    policy_model = torch.load("policy_model.torch")

    # inialize the model
    mcts_model = tabular_mcts(policy_value_model = policy_value_model)

    # num games per training loop
    characters = ["0","1","2"]
    all_boards = printAllKLengthRec(characters, "", 3,9 ) 
    all_boards = boards
    all_boards = [[int(i) for i in j] for j in all_boards]

    actions= [1/9 for i in range(9)]

    experience = [[board,actions,board,0] for board in all_boards]
    for i in range(10):
        print(i)
        mcts_model.train(experience)

    result = mcts_model.call_model(all_boards[0])
    print(result)

    
    

