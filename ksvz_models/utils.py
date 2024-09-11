# utils.py

import numpy as np

from numba import njit

factorial_table = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200, 1307674368000, 20922789888000, 355687428096000], dtype='int64')

@njit
def fast_factorial(n: int) -> int:
   """
   Lookup table for the factorial of \f$n!$\f$.

   Parameters
   ----------
   n : int
      Positive integer \f$n\f$ to compute the factorial of.

   Returns
   -------
   int
      The factorial \f$n!\f$.
   """
   if n > len(factorial_table):
      raise ValueError(f"Factorial of {n} is too large for the factorial table.")
   return factorial_table[n]

@njit
def sign(x):
   """
   Implementation of the sign function.

   Parameters
   ----------
   x : float
      The input value.

   Returns
   -------
   int
      The sign of the input value (-1, 0, or 1)
   """
   return (x > 0) - (x < 0)
