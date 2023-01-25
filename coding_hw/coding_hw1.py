from typing import List, Tuple, Dict, Optional, cast
from environments.environment_abstract import Environment, State
from environments.farm_grid_world import FarmState
from environments.n_puzzle import NPuzzleState
from heapq import heappush, heappop
import time
import numpy as np


class Node:
    def __init__(self, state: State, path_cost: float, parent_action: Optional[int], parent):
        self.state: State = state
        self.parent: Optional[Node] = parent
        self.path_cost: float = path_cost
        self.parent_action: Optional[int] = parent_action

    def __hash__(self):
        return self.state.__hash__()

    def __gt__(self, other):
        return self.path_cost < other.path_cost

    def __eq__(self, other):
        return self.state == other.state


def get_next_state_and_transition_cost(env: Environment, state: State, action: int) -> Tuple[State, float]:
    """

    :param env: Environment
    :param state: State
    :param action: Action
    :return: the next state and the transition cost
    """
    rw, states_a, _ = env.state_action_dynamics(state, action)
    state: State = states_a[0]
    transition_cost: float = -rw

    return state, transition_cost


def visualize_bfs(viz, closed_states: List[State], queue: List[Node], wait: float):
    """

    :param viz: visualizer
    :param closed_states: states in CLOSED
    :param queue: states in priority queue
    :param wait: number of seconds to wait after displaying
    :return: None
    """

    if viz is None:
        return

    grid_dim_x, grid_dim_y = viz.env.grid_shape
    for pos_i in range(grid_dim_x):
        for pos_j in range(grid_dim_y):
            viz.board.itemconfigure(viz.grid_squares[pos_i][pos_j], fill="white")

    for state_u in closed_states:
        pos_i_up, pos_j_up = state_u.agent_idx
        viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="red")

    for node in queue:
        state_u: FarmState = cast(FarmState, node.state)
        pos_i_up, pos_j_up = state_u.agent_idx
        viz.board.itemconfigure(viz.grid_squares[pos_i_up][pos_j_up], fill="grey")

    viz.window.update()
    time.sleep(wait)


def search_optimal(state_start: State, env: Environment, viz) -> Optional[List[int]]:
    """ Return an optimal path
    :param state_start: starting state
    :param env: environment
    :param viz: visualization object
    :return: a list of integers representing the actions that should be taken to reach the goal or None if no solution
    """
    current_node = Node(state_start, 0, None, None)
    opened = [current_node]
    seen = {state_start: current_node}
    while opened:
        prioritize_list(opened)
        current_node = opened.pop(0)
        if env.is_terminal(current_node.state):
            return ret_formatter(current_node)
        paths = env.get_actions(current_node.state)
        for path in paths:
            state_next, cost_next = get_next_state_and_transition_cost(env, current_node.state, path)
            child = Node(state_next, cost_next + current_node.path_cost, path, current_node)
            seen_node = seen.get(state_next)
            if seen_node is None or seen_node.__gt__(child):
                seen[state_next] = child
                opened.append(child)
    return None


def prioritize_list(sort_list: List[Node]) -> List[Node]:
    ret_list = []
    goal = np.array([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])

    least_cost = -1
    for nodes in sort_list:
        current_distribution = nodes.state.tiles.reshape((3, 3))

        j = 0
        while j <= 8:
            curr_x, curr_y = np.where(current_distribution == j)
            final_x, final_y = np.where(goal == j)
            nodes.path_cost += abs(final_x[0] - curr_x[0]) + abs(final_y[0] - curr_y[0])
            j = j + 1
        if nodes.path_cost < least_cost or least_cost == -1:
            ret_list.insert(0, nodes)
        else:
            ret_list.append(nodes)
    return sort_list


def ret_formatter(last_node: Node) -> List[int]:
    path = []
    while last_node.parent_action is not None:
        path.append(last_node.parent_action)
        last_node = last_node.parent
    return reversed(path)


def search_speed(state_start: State, env: Environment, viz) -> Optional[List[int]]:
    """ Return a path as quickly as possible
    :param state_start: starting state
    :param env: environment
    :param viz: visualization object

    :return: a list of integers representing the actions that should be taken to reach the goal or None if no solution
    """

    pass
