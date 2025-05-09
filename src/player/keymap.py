import pygame


GAME_KEYMAP = {
    pygame.K_w: "p1_up",
    pygame.K_d: "p1_right",
    pygame.K_a: "p1_left",
    pygame.K_s: "p1_down",

    pygame.K_UP: "p2_up",
    pygame.K_RIGHT: "p2_right",
    pygame.K_LEFT: "p2_left",
    pygame.K_DOWN: "p2_down",
}

GAME_FORBIDDEN_COMBINATIONS = {}
