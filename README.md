# MCTS TICTACTOE

In the project we use the AlphaZero algorithm on the game of tictactoe  
This like applying a jackhammer to a nail, you could solve tictactoe using a lookup table,
there are only 19,683 possible states and many of those are reflections of each other.  
The reason I chose to use tictactoe was so I could have the model converge on my laptop cpu in a matter of minuites not the 5,000 TPU's for 24 hours that it took for Deepmind to achieve superhuman level of Go.  
In that respect I think it was a success, the model will not lose and will beat a naive player.  
I also hold this project dear to my heart because I was able to really learn about the workings of mcts and the alpha zero paper

## To Run
clone the repository  
Because there is a pretrained model included, you may run `python play_against_ai.py` and you can play against a model that will play competently  
if you want to train your own model for any reason, you can run `python train.py`  and watch as the agent gets better  

## Interesting notes
If you're reading this because you want to implement something similar yourself, here's what I learned:
- Being object oriented and treating the mcts tree as a real tree sounds wonderful but your primary way of interacting with it is aggeragting all the next moves from a board state, so I found it was easier to use a dictonary for each of the major things I was tracking (num visits, q values from backup, policy values from model)  
- This is talked about in reinforcment learning a fair amount but using small models works really well for this application, our task is not as complicated and independent as something like ResNet so having less parameters allows us to train faster and learn simpler policies.
- My presentation at UCI in 2019 about this topic: https://www.youtube.com/watch?v=axSM-YMa4RU&t=33s  






