import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
import argparse
from neon_racer.game_engine import GameEngine

def main():
    parser = argparse.ArgumentParser(description="Neon-Racer AI Environment")
    parser.add_argument('--track', type=str, default='default', help="Track name to load")
    parser.add_argument('--agents', nargs='+', help="List of agents to race e.g. 'manual' 'neat:best.pkl' 'rl:model.zip'")
    parser.add_argument('--no-sound', action='store_true', help="Disable sound effects")
    args = parser.parse_args()
    
    agents_config = []
    
    if args.agents:
        print(f"Starting Multi-Agent Race on track '{args.track}'...")
        for i, agent_str in enumerate(args.agents):
            parts = agent_str.split(':')
            atype = parts[0]
            path = parts[1] if len(parts) > 1 else None
            name = f"{atype.capitalize()}_{i+1}"
            if path:
                name = f"{atype.upper()}_{path}"
                
            conf = {'type': atype, 'name': name}
            if path:
                conf['path'] = path
            agents_config.append(conf)
    else:
        print(f"Starting Neon-Racer on track '{args.track}'...")
        agents_config = [{'type': 'manual', 'name': 'Player'}]
    
    engine = GameEngine(track_name=args.track, agents_config=agents_config, enable_sound=(not args.no_sound))
    
    engine.run()

if __name__ == "__main__":
    main()
