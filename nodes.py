import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

from .src.agent import Agent
from .src.envs import WorldModelEnv
from .src.game.game import Game
from .src.game.play_env import PlayEnv


OmegaConf.register_new_resolver("eval", eval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--record", action="store_true", help="Record episodes in PlayEnv.")
    parser.add_argument("--store-denoising-trajectory", action="store_true", help="Save denoising steps in info.")
    parser.add_argument("--store-original-obs", action="store_true", help="Save original obs (pre resizing) in info.")
    parser.add_argument("--mouse-multiplier", type=int, default=10, help="Multiplication factor for the mouse movement.")
    parser.add_argument("--compile", action="store_true", help="Turn on model compilation.")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate.")
    parser.add_argument("--no-header", action="store_true")
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if not args.record and (args.store_denoising_trajectory or args.store_original_obs):
        print("Warning: not in recording mode, ignoring --store* options")
    return True


def prepare_play_mode(cfg: DictConfig, args: argparse.Namespace) -> PlayEnv:
    path_hf = Path(snapshot_download(repo_id="Enigma-AI/multiverse"))

    path_ckpt = path_hf / 'agent.pt'
    spawn_dir = Path('.') / 'game/spawn'
    # Override config
    cfg.agent = OmegaConf.load("config/agent/racing.yaml")
    cfg.env = OmegaConf.load("config/env/racing.yaml")
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("----------------------------------------------------------------------")
    print(f"Using {device} for rendering.")
    if not torch.cuda.is_available() and not torch.backends.mps.is_available(): # warn in case CUDA isn't being used (not on MPS devices)
        print("If you have a CUDA GPU available and it is not being used, please follow the instructions at https://pytorch.org/get-started/locally/ to reinstall torch with CUDA support and try again.")
    print("----------------------------------------------------------------------")

    assert cfg.env.train.id == "racing"
    num_actions = cfg.env.num_actions

    # Models
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    agent.load(path_ckpt)
    
    # World model environment
    sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
    if agent.upsampler is not None:
        sl = max(sl, cfg.agent.upsampler.inner_model.num_steps_conditioning)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env = WorldModelEnv(agent.denoiser, agent.upsampler, agent.rew_end_model, spawn_dir, 1, sl, wm_env_cfg, return_denoising_trajectory=True)
    
    if device.type == "cuda" and args.compile:
        print("Compiling models...")
        wm_env.predict_next_obs = torch.compile(wm_env.predict_next_obs, mode="reduce-overhead")
        wm_env.upsample_next_obs = torch.compile(wm_env.upsample_next_obs, mode="reduce-overhead")

    play_env = PlayEnv(
        agent,
        wm_env,
        args.record,
        args.store_denoising_trajectory,
        args.store_original_obs,
    )

    return play_env


@torch.no_grad()
def main():
    args = parse_args()
    ok = check_args(args)
    if not ok:
        return

    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="trainer")

    # window size
    h, w = (cfg.env.train.size,) * 2 if isinstance(cfg.env.train.size, int) else cfg.env.train.size
    size_h, size_w = h, w
    env = prepare_play_mode(cfg, args)
    game = Game(env, (size_h, size_w), fps=args.fps, verbose=not args.no_header)
    game.run()


class PlayGame:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "play_game"
    CATEGORY = "Multiverse"

    def play_game(self):
        main()
        return ()
