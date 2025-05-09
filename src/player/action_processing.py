"""
Credits: some parts are taken and modified from the file `config.py` from https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning/
"""

from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np
import pygame
import torch

from .keymap import GAME_FORBIDDEN_COMBINATIONS, GAME_KEYMAP


@dataclass
class GameAction:
    keys: List[int]

    def __post_init__(self) -> None:
        self.keys = filter_keys_pressed_forbidden(self.keys)
        # self.process_mouse()

    @property
    def key_names(self) -> List[str]:
        return [pygame.key.name(key) for key in self.keys]


def print_game_action(action: GameAction) -> str:
    action_names = [GAME_KEYMAP[k] for k in action.keys] if len(action.keys) > 0 else []
    action_names = [x for x in action_names if not x.startswith("camera_")]
    keys = " + ".join(action_names)
    return keys


N_PLAYERS = 2
N_GAS_KAVIM = 11
N_BREX_KAVIM = 11
N_STEERING = 11
N_KEYS = 8  # number of keyboard outputs, w,s,a,d,up,down,left,right


def encode_game_action(game_action: GameAction, device: torch.device) -> torch.Tensor:
    p1_gas = torch.zeros(N_GAS_KAVIM)
    p2_gas = torch.zeros(N_GAS_KAVIM)

    p1_brex = torch.zeros(N_BREX_KAVIM)
    p2_brex = torch.zeros(N_BREX_KAVIM)

    p1_steer = torch.zeros(N_STEERING)
    p2_steer = torch.zeros(N_STEERING)

    p1_is_steer = False
    p2_is_steer = False

    for key in game_action.key_names:
        if key == "w":
            p1_gas[N_GAS_KAVIM - 1] = 1
        if key == "a":
            p1_steer[3] = 1
            p1_is_steer = True
        if key == "s":
            p1_brex[N_BREX_KAVIM - 1] = 1
        if key == "d":
            p1_steer[N_STEERING - 3] = 1
            p1_is_steer = True

        if key == "up":
            p2_gas[N_GAS_KAVIM - 1] = 1
        if key == "left":
            p2_steer[3] = 1
            p2_is_steer = True
        if key == "down":
            p2_brex[N_BREX_KAVIM - 1] = 1
        if key == "right":
            p2_steer[N_STEERING - 3] = 1
            p2_is_steer = True

    if not p1_is_steer:
        p1_steer[len(p1_steer) // 2] = 1

    if not any(p1_gas):
        p1_gas[0] = 1

    if not any(p1_brex):
        p1_brex[0] = 1

    if not p2_is_steer:
        p2_steer[len(p2_steer) // 2] = 1

    if not any(p2_gas):
        p2_gas[0] = 1

    if not any(p2_brex):
        p2_brex[0] = 1

    return torch.cat([p1_gas, p1_brex, p1_steer, p2_gas, p2_brex, p2_steer]).float().to(device)


def decode_game_action(y_preds: torch.Tensor) -> GameAction:
    y_preds = y_preds.squeeze()
    keys_pred = y_preds[0:N_KEYS]

    keys_pressed = []
    keys_pressed_onehot = np.round(keys_pred)
    if keys_pressed_onehot[0] == 1:
        keys_pressed.append("w")
    if keys_pressed_onehot[1] == 1:
        keys_pressed.append("a")
    if keys_pressed_onehot[2] == 1:
        keys_pressed.append("s")
    if keys_pressed_onehot[3] == 1:
        keys_pressed.append("d")
    if keys_pressed_onehot[4] == 1:
        keys_pressed.append("up")
    if keys_pressed_onehot[5] == 1:
        keys_pressed.append("left")
    if keys_pressed_onehot[6] == 1:
        keys_pressed.append("down")
    if keys_pressed_onehot[7] == 1:
        keys_pressed.append("right")

    keys_pressed = [pygame.key.key_code(x) for x in keys_pressed]

    return GameAction(keys_pressed)


def filter_keys_pressed_forbidden(keys_pressed: List[int], keymap: Dict[int, str] = GAME_KEYMAP,
                                  forbidden_combinations: List[Set[str]] = GAME_FORBIDDEN_COMBINATIONS) -> List[int]:
    keys = set()
    names = set()
    for key in keys_pressed:
        if key not in keymap:
            continue
        name = keymap[key]
        keys.add(key)
        names.add(name)
        for forbidden in forbidden_combinations:
            if forbidden.issubset(names):
                keys.remove(key)
                names.remove(name)
                break
    return list(filter(lambda key: key in keys, keys_pressed))
