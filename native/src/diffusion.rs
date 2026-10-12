//! Preconditioned conjugate gradients for the screened surface Laplacian.
//! Degree weighting makes the row-normalized diffusion system symmetric.
use serde_json::{Value, json};

fn apply(
    neighbors: &[Vec<(usize, f32)>],
    degree: &[f32],
    width: usize,
    lambda: f32,
    x: &[f32],
    out: &mut [f32],
) {
    for (i, row) in neighbors.iter().enumerate() {
        let values = &x[i * width..(i + 1) * width];
        let target = &mut out[i * width..(i + 1) * width];
        for k in 0..width {
            target[k] = degree[i] * values[k];
        }
        for &(j, weight) in row {
            let other = &x[j * width..(j + 1) * width];
            for k in 0..width {
                target[k] += lambda * weight * (values[k] - other[k]);
            }
        }
    }
}

fn dots(a: &[f32], b: &[f32], width: usize) -> Vec<f64> {
    let mut result = vec![0.0; width];
    for (a, b) in a.chunks_exact(width).zip(b.chunks_exact(width)) {
        for k in 0..width {
            result[k] += f64::from(a[k]) * f64::from(b[k]);
        }
    }
    result
}

/// Solve (D + lambda L) W = D W0 with a deterministic traversal and measured residual.
pub fn solve(
    neighbors: &[Vec<(usize, f32)>],
    initial: &[f32],
    width: usize,
    height: f32,
    lambda: f32,
) -> Result<(Vec<f32>, Value), String> {
    let graph: Vec<Vec<_>> = neighbors
        .iter()
        .map(|row| {
            row.iter()
                .map(|&(j, distance)| (j, 1.0 / distance.max(0.002 * height)))
                .collect()
        })
        .collect();
    let degree: Vec<f32> = graph
        .iter()
        .map(|row| {
            if row.is_empty() {
                1.0
            } else {
                row.iter().map(|(_, w)| w).sum()
            }
        })
        .collect();
    let diagonal: Vec<f32> = graph
        .iter()
        .zip(&degree)
        .map(|(row, d)| {
            if row.is_empty() {
                *d
            } else {
                (1.0 + lambda) * d
            }
        })
        .collect();
    let mut x = initial.to_vec();
    let mut product = vec![0.0; x.len()];
    apply(&graph, &degree, width, lambda, &x, &mut product);
    let rhs: Vec<f32> = initial
        .chunks_exact(width)
        .zip(&degree)
        .flat_map(|(row, &d)| row.iter().map(move |v| d * v))
        .collect();
    let mut residual: Vec<f32> = rhs.iter().zip(&product).map(|(a, b)| a - b).collect();
    let mut preconditioned: Vec<f32> = residual
        .chunks_exact(width)
        .zip(&diagonal)
        .flat_map(|(row, &d)| row.iter().map(move |v| v / d))
        .collect();
    let mut direction = preconditioned.clone();
    let mut rho = dots(&residual, &preconditioned, width);
    let rhs_norm = dots(&rhs, &rhs, width);
    let mut active = vec![true; width];
    let mut iterations = 0;
    for iteration in 0..192 {
        let residual_norm = dots(&residual, &residual, width);
        for k in 0..width {
            active[k] = residual_norm[k] > 1e-10 * rhs_norm[k].max(1e-20);
        }
        if !active.iter().any(|v| *v) {
            break;
        }
        apply(&graph, &degree, width, lambda, &direction, &mut product);
        let curvature = dots(&direction, &product, width);
        let mut alpha = vec![0.0_f32; width];
        for k in 0..width {
            if active[k] {
                if !curvature[k].is_finite() || curvature[k] <= 0.0 {
                    return Err("Skin diffusion lost positive curvature".into());
                }
                alpha[k] = (rho[k] / curvature[k]) as f32;
            }
        }
        for (i, &diagonal_value) in diagonal.iter().enumerate() {
            for (k, &step) in alpha.iter().enumerate() {
                let index = i * width + k;
                x[index] += step * direction[index];
                residual[index] -= step * product[index];
                preconditioned[index] = residual[index] / diagonal_value;
            }
        }
        let next_rho = dots(&residual, &preconditioned, width);
        for i in 0..graph.len() {
            for k in 0..width {
                let index = i * width + k;
                let beta = if active[k] {
                    (next_rho[k] / rho[k].max(1e-30)) as f32
                } else {
                    0.0
                };
                direction[index] = preconditioned[index] + beta * direction[index];
            }
        }
        rho = next_rho;
        iterations = iteration + 1;
    }
    apply(&graph, &degree, width, lambda, &x, &mut product);
    let mut maximum = 0.0_f32;
    let mut minimum = 0.0_f32;
    for i in 0..graph.len() {
        let row = &mut x[i * width..(i + 1) * width];
        for k in 0..width {
            maximum = maximum.max((rhs[i * width + k] - product[i * width + k]).abs() / degree[i]);
            minimum = minimum.min(row[k]);
            if !row[k].is_finite() {
                return Err("Nonfinite skin diffusion result".into());
            }
            row[k] = row[k].max(0.0);
        }
        let total: f32 = row.iter().sum();
        if total <= 0.0 {
            return Err("Skin diffusion produced zero support".into());
        }
        for value in row {
            *value /= total;
        }
    }
    Ok((
        x,
        json!({"solver":"preconditioned-conjugate-gradient", "lambda":lambda, "iterations":iterations, "max_normalized_system_residual":maximum, "minimum_before_nonnegative_projection":minimum}),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_analytic_screened_three_vertex_path_and_keeps_isolated_vertex() {
        let graph = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(1, 1.0)],
            vec![],
        ];
        let initial = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.25, 0.75];
        let (weights, report) = solve(&graph, &initial, 2, 1.0, 1.0).unwrap();
        let expected = [
            7.0 / 12.0,
            5.0 / 12.0,
            1.0 / 6.0,
            5.0 / 6.0,
            1.0 / 12.0,
            11.0 / 12.0,
            0.25,
            0.75,
        ];
        for (actual, expected) in weights.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        assert!(report["max_normalized_system_residual"].as_f64().unwrap() < 1e-5);
    }
}
