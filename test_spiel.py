import pyspiel
import open_spiel.python.games
# Print version and list a few games to ensure the C++ bindings are linked
print(f"OpenSpiel loaded successfully.")
print(f"Registered games: {len(pyspiel.registered_names())}")
print("Trying to load Tic-Tac-Toe...")

game = pyspiel.load_game("tic_tac_toe")
print(f"Game loaded: {game}")