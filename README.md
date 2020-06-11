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
if you want to train your own model for any reason, you may
