from IPython import display
import matplotlib.pyplot as plt

plt.ion()

def plot(scores, mean_scores):
    plt.clf()  # Clear the current figure
    plt.title("Training")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores, label='Scores')
    plt.plot(mean_scores, label='Mean Scores')
    plt.ylim(ymin=0)
    
    # Add text annotations if there are scores
    if scores:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    
    plt.legend()
    plt.pause(0.1)
    plt.show()
    plt.savefig("assets/plot.png")

