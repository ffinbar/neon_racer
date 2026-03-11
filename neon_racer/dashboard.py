
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import os

LOG_FILE = 'training_log.jsonl'
plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 6))
fig.canvas.manager.set_window_title('Neon Racer - Training Dashboard')

ax1 = fig.add_subplot(1, 1, 1)

def animate(i):
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)

    if not os.path.exists(LOG_FILE):
        ax.text(0.5, 0.5, "Waiting for training log...", ha='center', va='center')
        return

    neat_data = []
    rl_data = []
    
    try:
        with open(LOG_FILE, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('type') == 'neat':
                        neat_data.append(entry['data'])
                    elif entry.get('type') == 'rl':
                        rl_data.append(entry['data'])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading log: {e}")
        return
    if neat_data:
        generations = [d.get('generation') for d in neat_data]
        avg_fitness = [d.get('avg_fitness') for d in neat_data]
        best_fitness = [d.get('best_fitness') for d in neat_data]
        max_fitness = [d.get('max_fitness') for d in neat_data]
        
        cumulative_avg = []
        running_sum = 0
        for i, val in enumerate(avg_fitness):
            running_sum += val
            cumulative_avg.append(running_sum / (i + 1))
        
        ax.plot(generations, max_fitness, label='All-Time Best', color='cyan', linestyle='--')
        ax.plot(generations, best_fitness, label='Gen Best', color='lime')
        ax.plot(generations, avg_fitness, label='Gen Avg', color='magenta', alpha=0.5)
        ax.plot(generations, cumulative_avg, label='All-Time Avg', color='yellow', linestyle=':')
        
        ax.set_title("NEAT Training Progress")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    elif rl_data:
        steps = [d.get('step') for d in rl_data]
        lines = []
        labels = []
        legend_ax = ax

        if 'loss' in rl_data[0]:
            losses = [d.get('loss') for d in rl_data]
            l1, = ax.plot(steps, losses, label='Loss', color='orange')
            ax.set_ylabel("Loss")
            lines.append(l1)
            labels.append(l1.get_label())
            
        if 'reward' in rl_data[0]:
            rewards = [d.get('reward') for d in rl_data]
            if 'loss' in rl_data[0]:
                ax2 = ax.twinx()
                l2, = ax2.plot(steps, rewards, label='Mean Reward', color='lime')
                ax2.set_ylabel("Mean Reward")
                lines.append(l2)
                labels.append(l2.get_label())
                legend_ax = ax2
            else:
                 l2, = ax.plot(steps, rewards, label='Mean Reward', color='lime')
                 ax.set_ylabel("Mean Reward")
                 lines.append(l2)
                 labels.append(l2.get_label())

        ax.set_title("RL Training Progress (PPO)")
        ax.set_xlabel("Time Steps")
        legend = legend_ax.legend(lines, labels, loc='upper left')
        legend.set_zorder(100)
        ax.grid(True, alpha=0.3)
        
    else:
        ax.text(0.5, 0.5, "Log file empty...", ha='center', va='center')

def main():
    print(f"Starting Dashboard. reading from {os.path.abspath(LOG_FILE)}")
    ani = animation.FuncAnimation(fig, animate, interval=100, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    main()
