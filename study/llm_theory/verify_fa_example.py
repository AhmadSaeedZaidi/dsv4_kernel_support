"""Verify the Flash Attention 2 worked example in fa.md.

Compares the tiled online-softmax algorithm (with the correction factors
from Milakov & Gimelshein 2018) against the standard non-tiled attention
on the same Q_1, K, V inputs. Also prints intermediate quantities so the
worked example's numbers can be cross-checked.
"""

from __future__ import annotations

import numpy as np


def rowmax(X: np.ndarray) -> np.ndarray:
    return X.max(axis=1)


def rowsum(X: np.ndarray) -> np.ndarray:
    return X.sum(axis=1)


def standard_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    S = Q @ K.T
    P = np.exp(S - rowmax(S)[:, None])
    P = P / rowsum(P)[:, None]
    return P @ V


def flash_attention_tile(
    Q_i: np.ndarray, K_blocks: list[np.ndarray], V_blocks: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    B_r = Q_i.shape[0]
    m_i = np.full(B_r, -np.inf)
    l_i = np.zeros(B_r)
    O_i = np.zeros((B_r, Q_i.shape[1]))
    trace: list[dict] = []

    for j, (K_j, V_j) in enumerate(zip(K_blocks, V_blocks), start=1):
        S_ij = Q_i @ K_j.T
        m_ij = rowmax(S_ij)
        P_ij = np.exp(S_ij - m_ij[:, None])
        l_ij = rowsum(P_ij)
        P_ij_V_j = P_ij @ V_j

        m_new = np.maximum(m_i, m_ij)
        alpha = np.exp(m_i - m_new)
        beta = np.exp(m_ij - m_new)
        l_new = alpha * l_i + beta * l_ij
        O_new = alpha[:, None] * O_i + beta[:, None] * P_ij_V_j

        trace.append(
            dict(
                j=j,
                S_ij=S_ij,
                m_ij=m_ij,
                P_ij=P_ij,
                l_ij=l_ij,
                P_ij_V_j=P_ij_V_j,
                m_old=m_i.copy(),
                m_new=m_new,
                l_old=l_i.copy(),
                l_new=l_new,
                alpha=alpha,
                beta=beta,
                O_old=O_i.copy(),
                O_new=O_new,
            )
        )
        m_i, l_i, O_i = m_new, l_new, O_new

    O_final = O_i / l_i[:, None]
    return O_final, m_i, l_i, trace


def main() -> None:
    Q = np.array(
        [
            [2, -1],
            [-2, 5],
            [4, 1],
            [0, 2],
            [3, 0],
            [1, -3],
        ],
        dtype=float,
    )
    K = np.array(
        [
            [1, -2],
            [0, 3],
            [4, 1],
            [-1, 2],
            [5, 0],
            [2, -4],
        ],
        dtype=float,
    )
    V = np.array(
        [
            [3, 0],
            [1, 2],
            [-2, 1],
            [5, -3],
            [4, 0],
            [-1, 2],
        ],
        dtype=float,
    )
    Br, Bc = 2, 2
    Q1 = Q[0:Br]
    K_blocks = [K[0 * Bc : 1 * Bc], K[1 * Bc : 2 * Bc], K[2 * Bc : 3 * Bc]]
    V_blocks = [V[0 * Bc : 1 * Bc], V[1 * Bc : 2 * Bc], V[2 * Bc : 3 * Bc]]

    O_standard = standard_attention(Q1, K, V)
    O_flash, m_final, l_final, trace = flash_attention_tile(Q1, K_blocks, V_blocks)

    print("Q_1 =\n", Q1)
    print("K^T full =\n", K.T)
    print("V rows 1-6 =\n", V)
    print()

    for step in trace:
        j = step["j"]
        print(f"--- Inner step (i=1, j={j}) ---")
        print(f"K_{j}^T =\n{step['S_ij'].shape}  (S_{1}{j} = Q_1 K_{j}^T)")
        print(f"S_{1}{j} =\n{step['S_ij']}")
        print(f"m_{1}{j} = {step['m_ij']}")
        print(f"P_{1}{j} =\n{step['P_ij']}")
        print(f"l_{1}{j} = {step['l_ij']}")
        print(f"m_1^old = {step['m_old']}  m_1^new = {step['m_new']}")
        print(f"alpha = e^(m_old - m_new) = {step['alpha']}")
        print(f"beta  = e^(m_ij - m_new) = {step['beta']}")
        print(f"l_1^old = {step['l_old']}")
        print(f"l_1^new = {step['l_new']}")
        print(f"P_{1}{j} V_{j} =\n{step['P_ij_V_j']}")
        print(f"O_1^old =\n{step['O_old']}")
        print(f"O_1^new =\n{step['O_new']}")
        print()

    print(f"m_1^final = {m_final}")
    print(f"l_1^final = {l_final}")
    print()

    print("diag(ell_1^final) =\n", np.diag(l_final))
    print()
    print("Step 6:  O_1 = O_1^final * diag(ell_1^final)^-1")
    print("        = diag(ell_1^final)^-1 @ O_1^final   (row-scaling)")
    print()
    O_via_diag = np.diag(1.0 / l_final) @ trace[-1]["O_new"]
    print("O_1^final (raw, before step 6) =\n", trace[-1]["O_new"])
    print("O_1 (after step 6, computed element-wise) =\n", O_flash)
    print("O_1 (after step 6, computed via diag)      =\n", O_via_diag)
    print()

    print("Standard attention  softmax(Q_1 K^T) V =\n", O_standard)
    print("Flash attention O_1 (final)              =\n", O_flash)
    print(
        "Max abs diff between flash and standard:",
        float(np.max(np.abs(O_flash - O_standard))),
    )


if __name__ == "__main__":
    main()
