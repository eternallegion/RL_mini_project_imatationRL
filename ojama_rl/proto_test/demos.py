# demos.py
DEMO = ["OJAMA_TRIO", "ZERO_GRAVITY", "BIG_EVOLUTION_PILL", "ATTACK_ALL", "STOP"]

def demo_indices(env):
    return [env.idx[a] for a in DEMO]
