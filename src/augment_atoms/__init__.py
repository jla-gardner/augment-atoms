from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import ase
import ase.io
import dacite
import data2objects
import load_atoms
import numpy as np
import vesin
import yaml
from ase.calculators.calculator import Calculator

__version__ = "0.1.0"


def select_structure(
    structures: list[ase.Atoms],
    kT: float,
    beta: float,
    rand: np.random.RandomState,
) -> ase.Atoms:
    energies = np.array([s.info["energy"] / len(s) for s in structures])
    energy_probs = np.exp(-(energies - energies[0]) / kT)
    energy_probs /= energy_probs.sum()

    levels = np.array([s.info["level"] for s in structures]) + 1
    level_probs = levels / levels.sum()

    assert 0 <= beta <= 1, (
        f"beta must be between 0 and 1 (inclusive), got {beta}"
    )
    probs = (1 - beta) * energy_probs + beta * level_probs
    # set any nan to 0
    probs[np.isnan(probs)] = 0
    idx = rand.choice(len(structures), p=probs)

    return structures[idx]


def generate_structures(
    starting_structure: ase.Atoms,
    calc: Calculator,
    existing_pool: list[ase.Atoms],
    config: Config,
) -> list[ase.Atoms]:
    rand = np.random.RandomState(config.seed)

    final_pool = list(existing_pool)
    if not final_pool:
        s = label(starting_structure, calc)
        if s.pbc.any():
            s.wrap()
        s.info["id"] = unique_id()
        s.info["level"] = 0
        s.info["parent"] = None
        s.info["relax_steps"] = 0
        final_pool.append(s)

    start_time = time.time()

    kT = config.get_kT()

    rejections = {"force": 0, "similar": 0, "separation": 0}

    while len(final_pool) < config.n_per_structure + len(existing_pool):
        print(
            f"Time: {int(time.time() - start_time):>6}s,",
            f"N: {len(final_pool):>5}",
            end="\r",
            flush=True,
        )

        # select structure from pool
        parent = select_structure(final_pool, kT, config.beta, rand)

        # rattle it
        child = parent.copy()
        if config.cell_sigma is not None:
            # make a small perturbation to THE ORIGINAL CELL
            # to ensure we don't end up with overly large or small cells
            cell_change = (
                rand.randn(3, 3)
                * config.cell_sigma
                * np.linalg.norm(starting_structure.cell.array, axis=1)
            ).T
            child.set_cell(
                starting_structure.cell.array + cell_change.T,
                scale_atoms=True,
            )
            restoring_matrix = starting_structure.cell.array @ np.linalg.inv(
                child.cell.array
            )
        else:
            restoring_matrix = np.eye(3)

        # choose log uniform sigma
        log_lo, log_hi = np.log(config.sigma_range)
        sigma = np.exp(rand.uniform(log_lo, log_hi))
        dx = rand.randn(len(child.positions), 3) * sigma
        dx -= np.mean(dx, axis=0)  # subtrac c.o.m.
        child.positions += dx
        child.info["sigma"] = sigma

        child.info["parent"] = parent.info["id"]
        child.info["level"] = parent.info["level"] + 1
        child.info["id"] = unique_id()

        # relax the structure
        s = label(child, calc)
        s.info["relax_steps"] = 0
        prev_s = s.copy()

        def atoms_far_enough_part(atoms: ase.Atoms) -> bool:
            (i,) = vesin.ase_neighbor_list(
                "i", atoms, cutoff=config.min_separation
            )
            return len(i) == 0

        for i in range(1, config.max_relax_steps + 2):
            if too_similar(s.positions @ restoring_matrix, final_pool):
                if not too_similar(
                    prev_s.positions @ restoring_matrix, final_pool
                ):
                    if not max_force(prev_s) < config.max_force:
                        rejections["force"] += 1
                    elif not atoms_far_enough_part(prev_s):
                        rejections["separation"] += 1
                    else:
                        final_pool.append(prev_s)
                else:
                    rejections["similar"] += 1

                break

            if i == config.max_relax_steps + 1:
                # end of the relaxation
                if max_force(s) > config.max_force:
                    rejections["force"] += 1
                elif not atoms_far_enough_part(s):
                    rejections["separation"] += 1
                else:
                    final_pool.append(s)

                break

            ΔE = (s.info["energy"] - parent.info["energy"]) / len(s)
            prob = np.exp(-ΔE / kT)
            prob = min(0.25, prob)
            if (
                max_force(s) < config.max_force
                and rand.uniform() < prob
                and atoms_far_enough_part(s)
            ):
                final_pool.append(s)
                break

            prev_s = s.copy()
            s.info["relax_steps"] += 1
            dir = direction(s.arrays["forces"])
            factor = s.info["sigma"] / i
            s.positions += factor * dir
            s = label(s, calc)

    print()
    names = {
        "force": f"Maximum force of {config.max_force} exceeded",
        "similar": "Structure too similar to existing structures",
        "separation": "Atoms too close to each other",
    }

    if any(v > 0 for v in rejections.values()):
        print("Rejections:")
        for k, v in rejections.items():
            if v > 0:
                print(f"   {v} {names[k]}")

    return final_pool[len(existing_pool) :]


def unique_id():
    return str(uuid.uuid4())


def label(structure: ase.Atoms, calc: Calculator) -> ase.Atoms:
    structure = structure.copy()
    calc.calculate(structure, ["energy", "forces"])

    structure.arrays["forces"] = calc.results["forces"]
    structure.info["energy"] = calc.results["energy"]
    return structure


def direction(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1)
    norm[norm < 1e-6] = 1e-6
    return v / norm[:, None]


def too_similar(
    pos: np.ndarray, pool: list[ase.Atoms], threshold: float = 0.1
) -> bool:
    for p in pool:
        if np.linalg.norm(pos - p.positions, axis=1).mean() < threshold:
            return True
    return False


def max_force(s: ase.Atoms) -> float:
    return np.linalg.norm(s.arrays["forces"], axis=1).max()


@dataclass
class Config:
    n_per_structure: int
    T: float
    beta: float
    sigma_range: tuple[float, float]
    seed: int
    units: Literal["eV", "kcal/mol"]
    cell_sigma: Union[float, None]  # noqa: UP007
    max_force: float = 30
    min_separation: float = 0.5
    max_relax_steps: int = 20

    def get_kT(self) -> float:
        if self.units == "eV":
            k = 8.617333262145e-5
        elif self.units == "kcal/mol":
            k = 0.0019872041
        else:
            raise ValueError(f"Unknown units: {self.units}")
        return self.T * k


@dataclass
class DataConfig:
    input: str
    output: str


@dataclass
class ModelConfig:
    calculator: Calculator


@dataclass
class AugmentConfig:
    data: DataConfig
    model: ModelConfig
    config: Config


def parse_config(path: str) -> AugmentConfig:
    with open(path) as f:
        config_dict = yaml.safe_load(f)

    print(yaml.dump(config_dict, indent=4))

    this_module = sys.modules[__name__]
    object_dict = data2objects.from_dict(
        config_dict,
        modules=[this_module],
    )

    # Convert sigma_range from list to tuple after data2objects conversion
    if "config" in object_dict and "sigma_range" in object_dict["config"]:
        object_dict["config"]["sigma_range"] = tuple(
            object_dict["config"]["sigma_range"]
        )

    return dacite.from_dict(
        data_class=AugmentConfig,
        data=object_dict,
        config=dacite.Config(strict=True),
    )


def graph_pes_calculator(path: str) -> Calculator:
    import torch
    from graph_pes.models import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(path).to(device)
    return model.ase_calculator(skin=2)


def lennard_jones() -> Calculator:
    from ase.calculators.lj import LennardJones

    return LennardJones()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Augment atoms based on configuration"
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the configuration file"
    )
    args = parser.parse_args()

    config = parse_config(args.config_path)

    starting_structures = load_atoms.load_dataset(config.data.input)
    for i, s in enumerate(starting_structures):
        s.info["starting-structure"] = i

    if Path(config.data.output).exists():
        existing_dataset = load_atoms.load_dataset(config.data.output)
        existing_structures = {}
        for i in range(len(starting_structures)):
            mask = existing_dataset.info["starting-structure"] == i
            if mask.sum() > 0:
                existing_structures[i] = list(existing_dataset[mask])
            else:
                existing_structures[i] = []

    else:
        existing_structures = {i: [] for i in range(len(starting_structures))}

    for i, s in enumerate(starting_structures):
        print(
            f"Augmenting structure {i + 1} of {len(starting_structures)}",
            flush=True,
        )
        new_structures = generate_structures(
            s,
            config.model.calculator,
            existing_structures[i],
            config.config,
        )
        ase.io.write(config.data.output, new_structures, append=True)


if __name__ == "__main__":
    main()
