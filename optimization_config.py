"""
Configuration system for toggling Model optimizations.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class OptimizationConfig:
    """
    Configuration for model optimizations.

    This class defines all toggleable optimizations that can be
    enabled or disabled for benchmarking purposes.  The Model
    class uses full optimizations by default, while this
    configuration system allows granular performance analysis.

    Attributes
    ----------
    vectorize : bool
        Enable vectorized NumPy operations over loops
    sparse : bool
        Enable sparse matrix representations
    polyalgorithm : bool
        Enable polyalgorithm for value function
    cython : bool
        Enable Cython-optimized implementations
    derivatives : bool
        Use analytical derivatives instead of finite differences
    """

    # Model computation optimizations
    vectorize: bool = True
    sparse: bool = True
    polyalgorithm: bool = True
    cython: bool = True

    # Estimation optimizations
    derivatives: bool = True

    def copy(self) -> 'OptimizationConfig':
        """
        Create a copy of this configuration.

        Returns
        -------
        OptimizationConfig
            A new instance with the same settings
        """
        return OptimizationConfig(
            vectorize=self.vectorize,
            sparse=self.sparse,
            polyalgorithm=self.polyalgorithm,
            cython=self.cython,
            derivatives=self.derivatives
        )

    @classmethod
    def full(cls) -> 'OptimizationConfig':
        """
        Fully optimized configuration (default).

        Returns
        -------
        OptimizationConfig
            Configuration with all optimizations enabled
        """
        return cls()

    @classmethod
    def baseline(cls) -> 'OptimizationConfig':
        """
        No optimizations (reference implementation).

        Returns
        -------
        OptimizationConfig
            Configuration with all optimizations disabled
        """
        return cls(
            vectorize=False,
            sparse=False,
            polyalgorithm=False,
            cython=False,
            derivatives=False
        )

    @classmethod
    def sequential(cls) -> List[Tuple[str, 'OptimizationConfig']]:
        """
        Generate configurations for benchmarking optimization impact.

        This method returns a sequence of configurations that match the
        required optimization levels exactly: baseline, vectorize,
        polyalgorithm, cython, sparse, derivatives.

        Returns
        -------
        List[Tuple[str, OptimizationConfig]]
            List of (name, config) pairs showing optimization levels
        """
        configs = []

        # baseline: No optimizations
        configs.append(("baseline", cls(
            vectorize=False, polyalgorithm=False, cython=False,
            sparse=False, derivatives=False
        )))

        # vectorize: Only vectorization enabled
        configs.append(("vectorize", cls(
            vectorize=True, polyalgorithm=False, cython=False,
            sparse=False, derivatives=False
        )))

        # polyalgorithm: vectorize + polyalgorithm
        configs.append(("polyalgorithm", cls(
            vectorize=True, polyalgorithm=True, cython=False,
            sparse=False, derivatives=False
        )))

        # cython: vectorize + polyalgorithm + cython
        configs.append(("cython", cls(
            vectorize=True, polyalgorithm=True, cython=True,
            sparse=False, derivatives=False
        )))

        # sparse: vectorize + polyalgorithm + cython + sparse
        configs.append(("sparse", cls(
            vectorize=True, polyalgorithm=True, cython=True,
            sparse=True, derivatives=False
        )))

        # derivatives: all optimizations
        configs.append(("derivatives", cls(
            vectorize=True, polyalgorithm=True, cython=True,
            sparse=True, derivatives=True
        )))

        return configs

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns
        -------
        dict
            Dictionary representation of the configuration
        """
        return {
            'vectorize': self.vectorize,
            'sparse': self.sparse,
            'polyalgorithm': self.polyalgorithm,
            'cython': self.cython,
            'derivatives': self.derivatives
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'OptimizationConfig':
        """
        Create configuration from dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary with optimization settings

        Returns
        -------
        OptimizationConfig
            Configuration instance
        """
        return cls(**config_dict)

    def __str__(self) -> str:
        """
        String representation showing enabled optimizations.

        Returns
        -------
        str
            Human-readable description of enabled optimizations
        """
        enabled = []
        if self.vectorize:
            enabled.append("vectorize")
        if self.sparse:
            enabled.append("sparse")
        if self.polyalgorithm:
            enabled.append("polyalgorithm")
        if self.cython:
            enabled.append("cython")
        if self.derivatives:
            enabled.append("derivatives")

        if not enabled:
            return "OptimizationConfig(no optimizations)"
        return f"OptimizationConfig({', '.join(enabled)})"

    def summary(self) -> str:
        """
        Generate a detailed summary of the configuration.

        Returns
        -------
        str
            Multi-line summary of optimization settings
        """
        lines = ["Optimization Configuration:"]
        lines.append(f"  Vectorized operations: {self.vectorize}")
        lines.append(f"  Sparse matrix operations: {self.sparse}")
        lines.append(f"  Polyalgorithm value function: {self.polyalgorithm}")
        lines.append(f"  Cython acceleration: {self.cython}")
        lines.append(f"  Analytical derivatives: {self.derivatives}")
        return "\n".join(lines)
