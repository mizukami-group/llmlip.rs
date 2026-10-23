//! MACE-specific primitive kernels for fixed-weight inference.

use e3nn_core::{CoreError, Result};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
fn mace_product_f32_parallel_chunk_rows(num_nodes: usize) -> usize {
    if num_nodes >= 64 {
        4
    } else {
        1
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MaceScalarSymmetricContraction<'a> {
    pub num_elements: usize,
    pub num_features: usize,
    pub num_ell: usize,
    pub u1_params: usize,
    pub u2_params: usize,
    pub u3_params: usize,
    pub u1: &'a [f64],
    pub u2: &'a [f64],
    pub u3: &'a [f64],
    pub weights_u2: &'a [f64],
    pub weights_u1: &'a [f64],
    pub weights_u3: &'a [f64],
    pub entries: Option<&'a MaceScalarSymmetricContractionEntries>,
}

#[derive(Clone, Debug)]
pub struct MaceScalarSymmetricContractionEntries {
    u1: Vec<ScalarU1Entry>,
    u2: Vec<ScalarU2Entry>,
    u3: Vec<ScalarU3Entry>,
}

#[derive(Clone, Copy, Debug)]
pub struct MaceVectorSymmetricContraction<'a> {
    pub num_elements: usize,
    pub num_features: usize,
    pub num_equivariance: usize,
    pub num_ell: usize,
    pub u1_params: usize,
    pub u2_params: usize,
    pub u3_params: usize,
    pub u1: &'a [f64],
    pub u2: &'a [f64],
    pub u3: &'a [f64],
    pub weights_u2: &'a [f64],
    pub weights_u1: &'a [f64],
    pub weights_u3: &'a [f64],
    pub entries: Option<&'a MaceVectorSymmetricContractionEntries>,
}

#[derive(Clone, Copy, Debug)]
pub struct MaceScalarVectorSymmetricContraction<'a> {
    pub scalar: MaceScalarSymmetricContraction<'a>,
    pub vector: MaceVectorSymmetricContraction<'a>,
}

#[derive(Clone, Debug)]
pub struct MaceVectorSymmetricContractionEntries {
    u1: Vec<Vec<VectorU1Entry>>,
    u2: Vec<Vec<VectorU2Entry>>,
    u3: Vec<Vec<VectorU3Entry>>,
}

#[derive(Clone, Debug)]
pub struct MaceScalarSingleElementProductPlan {
    num_features: usize,
    num_ell: usize,
    u1_entries: Vec<ScalarU1Entry>,
    u2_entries: Vec<ScalarU2Entry>,
    u2_grouped_entries: Vec<ScalarU2WeightedEntry>,
    u3_entries: Vec<ScalarU3Entry>,
    u3_grouped_entries: Vec<ScalarU3WeightedEntry>,
    weighted_u1: Vec<f64>,
    weighted_u2: Vec<f64>,
    weighted_u3: Vec<f64>,
    weighted_u2_grouped: Vec<f64>,
    weighted_u3_grouped: Vec<f64>,
    weighted_u1_f32: Vec<f32>,
    weighted_u2_grouped_f32: Vec<f32>,
    weighted_u3_grouped_f32: Vec<f32>,
    u1_len: usize,
    u2_len: usize,
    u2_grouped_len: usize,
    u3_len: usize,
    u3_grouped_len: usize,
}

#[derive(Clone, Debug)]
pub struct MaceVectorSingleElementProductPlan {
    num_features: usize,
    num_ell: usize,
    num_equivariance: usize,
    weighted_u1: Vec<Vec<VectorU1WeightedEntry>>,
    weighted_u2: Vec<Vec<VectorU2WeightedEntry>>,
    weighted_u3: Vec<Vec<VectorU3WeightedEntry>>,
    weighted_u1_f32_flat: Vec<VectorU1WeightedEntryF32>,
    weighted_u2_f32_flat: Vec<VectorU2WeightedEntryF32>,
    weighted_u3_f32_flat: Vec<VectorU3WeightedEntryF32>,
    weighted_u1_f32_ranges: Vec<(usize, usize)>,
    weighted_u2_f32_ranges: Vec<(usize, usize)>,
    weighted_u3_f32_ranges: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct MaceScalarVectorSingleElementProductPlan {
    scalar: MaceScalarSingleElementProductPlan,
    vector: MaceVectorSingleElementProductPlan,
}

impl MaceScalarSingleElementProductPlan {
    pub fn forward(&self, x: &[f64], out: &mut [f64]) -> Result<()> {
        let num_nodes = self.check_forward_shapes(
            x.len(),
            out.len(),
            "mace scalar prepared product forward out",
        )?;
        out.fill(0.0);
        self.forward_rows(x, out, num_nodes, self.num_features, 0);
        Ok(())
    }

    pub fn vjp_x(&self, x: &[f64], grad_out: &[f64], grad_x: &mut [f64]) -> Result<()> {
        let num_nodes = self.check_forward_shapes(
            x.len(),
            grad_out.len(),
            "mace scalar prepared product vjp grad_out",
        )?;
        self.check_grad_x_len(
            x.len(),
            grad_x.len(),
            "mace scalar prepared product vjp grad_x",
        )?;
        self.vjp_rows(x, grad_out, grad_x, num_nodes, self.num_features, 0);
        Ok(())
    }

    pub fn forward_f32(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        let num_nodes = self.check_forward_shapes(
            x.len(),
            out.len(),
            "mace scalar prepared product forward_f32 out",
        )?;
        out.fill(0.0);
        self.forward_rows_f32(x, out, num_nodes, self.num_features, 0);
        Ok(())
    }

    pub fn vjp_x_f32(&self, x: &[f32], grad_out: &[f32], grad_x: &mut [f32]) -> Result<()> {
        let num_nodes = self.check_forward_shapes(
            x.len(),
            grad_out.len(),
            "mace scalar prepared product vjp_x_f32 grad_out",
        )?;
        self.check_grad_x_len(
            x.len(),
            grad_x.len(),
            "mace scalar prepared product vjp_x_f32 grad_x",
        )?;
        self.vjp_rows_f32(x, grad_out, grad_x, num_nodes, self.num_features, 0);
        Ok(())
    }

    fn check_forward_shapes(
        &self,
        x_len: usize,
        out_len: usize,
        out_op: &'static str,
    ) -> Result<usize> {
        if self.num_ell != 16 {
            return Err(CoreError::InvalidTensorProduct(
                "mace scalar prepared product plan currently requires num_ell == 16".to_owned(),
            ));
        }
        let per_node_x = self.num_features * self.num_ell;
        if !x_len.is_multiple_of(per_node_x) {
            return Err(CoreError::ShapeMismatch {
                op: "mace scalar prepared product x",
                expected: per_node_x,
                actual: x_len,
            });
        }
        let num_nodes = x_len / per_node_x;
        let expected = num_nodes * self.num_features;
        if out_len != expected {
            return Err(CoreError::ShapeMismatch {
                op: out_op,
                expected,
                actual: out_len,
            });
        }
        Ok(num_nodes)
    }

    fn check_grad_x_len(&self, expected: usize, actual: usize, op: &'static str) -> Result<()> {
        if expected == actual {
            Ok(())
        } else {
            Err(CoreError::ShapeMismatch {
                op,
                expected,
                actual,
            })
        }
    }

    #[inline(always)]
    fn weighted_slices(&self, feature: usize) -> (&[f64], &[f64], &[f64]) {
        let u1_base = feature * self.u1_len;
        let u2_base = feature * self.u2_len;
        let u3_base = feature * self.u3_len;
        (
            &self.weighted_u1[u1_base..u1_base + self.u1_len],
            &self.weighted_u2[u2_base..u2_base + self.u2_len],
            &self.weighted_u3[u3_base..u3_base + self.u3_len],
        )
    }

    #[inline(always)]
    fn weighted_slices_f32(&self, feature: usize) -> (&[f32], &[f32], &[f32]) {
        let u1_base = feature * self.u1_len;
        let u2_base = feature * self.u2_grouped_len;
        let u3_base = feature * self.u3_grouped_len;
        (
            &self.weighted_u1_f32[u1_base..u1_base + self.u1_len],
            &self.weighted_u2_grouped_f32[u2_base..u2_base + self.u2_grouped_len],
            &self.weighted_u3_grouped_f32[u3_base..u3_base + self.u3_grouped_len],
        )
    }

    #[inline(always)]
    fn weighted_grouped_slices(&self, feature: usize) -> (&[f64], &[f64], &[f64]) {
        let u1_base = feature * self.u1_len;
        let u2_base = feature * self.u2_grouped_len;
        let u3_base = feature * self.u3_grouped_len;
        (
            &self.weighted_u1[u1_base..u1_base + self.u1_len],
            &self.weighted_u2_grouped[u2_base..u2_base + self.u2_grouped_len],
            &self.weighted_u3_grouped[u3_base..u3_base + self.u3_grouped_len],
        )
    }

    fn forward_rows(
        &self,
        x: &[f64],
        out: &mut [f64],
        num_nodes: usize,
        per_node_out: usize,
        out_offset: usize,
    ) {
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() >= 20 && num_nodes <= 8 && self.num_features >= 128 {
            for node in 0..num_nodes {
                let out_row = &mut out[node * per_node_out..(node + 1) * per_node_out];
                self.forward_one_row_feature_parallel(x, node, out_row, out_offset);
            }
            return;
        }
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && num_nodes > 1 {
            out.par_chunks_mut(per_node_out)
                .enumerate()
                .for_each(|(node, out_row)| {
                    self.forward_one_row(x, node, out_row, out_offset);
                });
            return;
        }
        for node in 0..num_nodes {
            let out_row = &mut out[node * per_node_out..(node + 1) * per_node_out];
            self.forward_one_row(x, node, out_row, out_offset);
        }
    }

    #[cfg(feature = "parallel")]
    fn forward_one_row_feature_parallel(
        &self,
        x: &[f64],
        node: usize,
        out_row: &mut [f64],
        out_offset: usize,
    ) {
        const FEATURE_CHUNK: usize = 32;
        out_row[out_offset..out_offset + self.num_features]
            .par_chunks_mut(FEATURE_CHUNK)
            .enumerate()
            .for_each(|(chunk, out_chunk)| {
                let feature_start = chunk * FEATURE_CHUNK;
                for (local_feature, out_value) in out_chunk.iter_mut().enumerate() {
                    let feature = feature_start + local_feature;
                    let (weighted_u1, weighted_u2, weighted_u3) = self.weighted_slices(feature);
                    let x_base = (node * self.num_features + feature) * 16;
                    let x_row = &x[x_base..x_base + 16];
                    *out_value = evaluate_scalar_weighted_monomials_ell16(
                        x_row,
                        &self.u1_entries,
                        weighted_u1,
                        &self.u2_entries,
                        weighted_u2,
                        &self.u3_entries,
                        weighted_u3,
                    );
                }
            });
    }

    fn forward_one_row(&self, x: &[f64], node: usize, out_row: &mut [f64], out_offset: usize) {
        for feature in 0..self.num_features {
            let (weighted_u1, weighted_u2, weighted_u3) = self.weighted_slices(feature);
            let x_base = (node * self.num_features + feature) * 16;
            let x_row = &x[x_base..x_base + 16];
            out_row[out_offset + feature] = evaluate_scalar_weighted_monomials_ell16(
                x_row,
                &self.u1_entries,
                weighted_u1,
                &self.u2_entries,
                weighted_u2,
                &self.u3_entries,
                weighted_u3,
            );
        }
    }

    fn forward_rows_f32(
        &self,
        x: &[f32],
        out: &mut [f32],
        num_nodes: usize,
        per_node_out: usize,
        out_offset: usize,
    ) {
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && num_nodes > 1 {
            let chunk_rows = mace_product_f32_parallel_chunk_rows(num_nodes);
            out.par_chunks_mut(per_node_out * chunk_rows)
                .enumerate()
                .for_each(|(chunk, out_chunk)| {
                    let node_start = chunk * chunk_rows;
                    for (local_node, out_row) in out_chunk.chunks_mut(per_node_out).enumerate() {
                        self.forward_one_row_f32(x, node_start + local_node, out_row, out_offset);
                    }
                });
            return;
        }
        for node in 0..num_nodes {
            let out_row = &mut out[node * per_node_out..(node + 1) * per_node_out];
            self.forward_one_row_f32(x, node, out_row, out_offset);
        }
    }

    fn forward_one_row_f32(&self, x: &[f32], node: usize, out_row: &mut [f32], out_offset: usize) {
        for feature in 0..self.num_features {
            let (weighted_u1, weighted_u2, weighted_u3) = self.weighted_slices_f32(feature);
            let x_base = (node * self.num_features + feature) * 16;
            let x_row = &x[x_base..x_base + 16];
            out_row[out_offset + feature] = evaluate_scalar_weighted_monomials_ell16_f32_prepared(
                x_row,
                &self.u1_entries,
                weighted_u1,
                &self.u2_grouped_entries,
                weighted_u2,
                &self.u3_grouped_entries,
                weighted_u3,
            );
        }
    }

    fn vjp_rows(
        &self,
        x: &[f64],
        grad_out: &[f64],
        grad_x: &mut [f64],
        num_nodes: usize,
        per_node_out: usize,
        out_offset: usize,
    ) {
        let per_node_x = self.num_features * 16;
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && num_nodes > 1 {
            grad_x
                .par_chunks_mut(per_node_x)
                .enumerate()
                .for_each(|(node, grad_x_row)| {
                    self.vjp_one_row(x, grad_out, node, grad_x_row, per_node_out, out_offset);
                });
            return;
        }
        for node in 0..num_nodes {
            let grad_x_row = &mut grad_x[node * per_node_x..(node + 1) * per_node_x];
            self.vjp_one_row(x, grad_out, node, grad_x_row, per_node_out, out_offset);
        }
    }

    fn vjp_rows_f32(
        &self,
        x: &[f32],
        grad_out: &[f32],
        grad_x: &mut [f32],
        num_nodes: usize,
        per_node_out: usize,
        out_offset: usize,
    ) {
        let per_node_x = self.num_features * 16;
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && num_nodes > 1 {
            let chunk_rows = mace_product_f32_parallel_chunk_rows(num_nodes);
            grad_x
                .par_chunks_mut(per_node_x * chunk_rows)
                .enumerate()
                .for_each(|(chunk, grad_x_chunk)| {
                    let node_start = chunk * chunk_rows;
                    for (local_node, grad_x_row) in grad_x_chunk.chunks_mut(per_node_x).enumerate()
                    {
                        self.vjp_one_row_f32(
                            x,
                            grad_out,
                            node_start + local_node,
                            grad_x_row,
                            per_node_out,
                            out_offset,
                        );
                    }
                });
            return;
        }
        for node in 0..num_nodes {
            let grad_x_row = &mut grad_x[node * per_node_x..(node + 1) * per_node_x];
            self.vjp_one_row_f32(x, grad_out, node, grad_x_row, per_node_out, out_offset);
        }
    }

    fn vjp_one_row(
        &self,
        x: &[f64],
        grad_out: &[f64],
        node: usize,
        grad_x_row: &mut [f64],
        per_node_out: usize,
        out_offset: usize,
    ) {
        for feature in 0..self.num_features {
            let bar_output = grad_out[node * per_node_out + out_offset + feature];
            if bar_output == 0.0 {
                continue;
            }
            let (weighted_u1, weighted_u2, weighted_u3) = self.weighted_grouped_slices(feature);
            let x_base = (node * self.num_features + feature) * 16;
            let x_row = &x[x_base..x_base + 16];
            let row_offset = feature * 16;
            accumulate_scalar_weighted_monomials_ell16_prepared_into(
                bar_output,
                x_row,
                &self.u1_entries,
                weighted_u1,
                &self.u2_grouped_entries,
                weighted_u2,
                &self.u3_grouped_entries,
                weighted_u3,
                &mut grad_x_row[row_offset..row_offset + 16],
            );
        }
    }

    fn vjp_one_row_f32(
        &self,
        x: &[f32],
        grad_out: &[f32],
        node: usize,
        grad_x_row: &mut [f32],
        per_node_out: usize,
        out_offset: usize,
    ) {
        for feature in 0..self.num_features {
            let bar_output = grad_out[node * per_node_out + out_offset + feature];
            if bar_output == 0.0 {
                continue;
            }
            let (weighted_u1, weighted_u2, weighted_u3) = self.weighted_slices_f32(feature);
            let x_base = (node * self.num_features + feature) * 16;
            let x_row = &x[x_base..x_base + 16];
            let row_offset = feature * 16;
            accumulate_scalar_weighted_monomials_ell16_f32_prepared_into(
                bar_output,
                x_row,
                &self.u1_entries,
                weighted_u1,
                &self.u2_grouped_entries,
                weighted_u2,
                &self.u3_grouped_entries,
                weighted_u3,
                &mut grad_x_row[row_offset..row_offset + 16],
            );
        }
    }
}

impl MaceVectorSingleElementProductPlan {
    pub fn forward(&self, x: &[f64], out: &mut [f64]) -> Result<()> {
        if self.num_ell != 16 {
            return Err(CoreError::InvalidTensorProduct(
                "mace vector prepared product plan currently requires num_ell == 16".to_owned(),
            ));
        }
        let per_node_x = self.num_features * self.num_ell;
        if !x.len().is_multiple_of(per_node_x) {
            return Err(CoreError::ShapeMismatch {
                op: "mace vector prepared product x",
                expected: per_node_x,
                actual: x.len(),
            });
        }
        let num_nodes = x.len() / per_node_x;
        let per_node_out = self.num_features * self.num_equivariance;
        let expected_out = num_nodes * per_node_out;
        if out.len() != expected_out {
            return Err(CoreError::ShapeMismatch {
                op: "mace vector prepared product forward out",
                expected: expected_out,
                actual: out.len(),
            });
        }
        out.fill(0.0);
        self.forward_rows(x, out, num_nodes, per_node_out, 0);
        Ok(())
    }

    #[inline(always)]
    fn weighted_slices_f32(
        &self,
        index: usize,
    ) -> (
        &[VectorU1WeightedEntryF32],
        &[VectorU2WeightedEntryF32],
        &[VectorU3WeightedEntryF32],
    ) {
        let (u1_start, u1_end) = self.weighted_u1_f32_ranges[index];
        let (u2_start, u2_end) = self.weighted_u2_f32_ranges[index];
        let (u3_start, u3_end) = self.weighted_u3_f32_ranges[index];
        (
            &self.weighted_u1_f32_flat[u1_start..u1_end],
            &self.weighted_u2_f32_flat[u2_start..u2_end],
            &self.weighted_u3_f32_flat[u3_start..u3_end],
        )
    }

    fn forward_rows(
        &self,
        x: &[f64],
        out: &mut [f64],
        num_nodes: usize,
        per_node_out: usize,
        out_offset: usize,
    ) {
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() >= 20 && num_nodes <= 8 && self.num_features >= 128 {
            for node in 0..num_nodes {
                let out_row = &mut out[node * per_node_out..(node + 1) * per_node_out];
                self.forward_one_row_feature_parallel(x, node, out_row, out_offset);
            }
            return;
        }
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && num_nodes > 1 {
            out.par_chunks_mut(per_node_out)
                .enumerate()
                .for_each(|(node, out_row)| {
                    self.forward_one_row(x, node, out_row, out_offset);
                });
            return;
        }
        for node in 0..num_nodes {
            let out_row = &mut out[node * per_node_out..(node + 1) * per_node_out];
            self.forward_one_row(x, node, out_row, out_offset);
        }
    }

    #[cfg(feature = "parallel")]
    fn forward_one_row_feature_parallel(
        &self,
        x: &[f64],
        node: usize,
        out_row: &mut [f64],
        out_offset: usize,
    ) {
        const FEATURE_CHUNK: usize = 16;
        let chunk_width = FEATURE_CHUNK * self.num_equivariance;
        out_row[out_offset..out_offset + self.num_features * self.num_equivariance]
            .par_chunks_mut(chunk_width)
            .enumerate()
            .for_each(|(chunk, out_chunk)| {
                let feature_start = chunk * FEATURE_CHUNK;
                for (local_feature, out_values) in
                    out_chunk.chunks_mut(self.num_equivariance).enumerate()
                {
                    let feature = feature_start + local_feature;
                    let x_base = (node * self.num_features + feature) * 16;
                    let x_row = &x[x_base..x_base + 16];
                    let index_base = feature * self.num_equivariance;
                    for (equiv, out_value) in out_values.iter_mut().enumerate() {
                        let index = index_base + equiv;
                        *out_value = evaluate_vector_weighted_monomials_ell16(
                            x_row,
                            &self.weighted_u1[index],
                            &self.weighted_u2[index],
                            &self.weighted_u3[index],
                        );
                    }
                }
            });
    }

    fn forward_one_row(&self, x: &[f64], node: usize, out_row: &mut [f64], out_offset: usize) {
        debug_assert_eq!(self.num_ell, 16);
        for feature in 0..self.num_features {
            let x_base = (node * self.num_features + feature) * 16;
            let x_row = &x[x_base..x_base + 16];
            let out_base = out_offset + feature * self.num_equivariance;
            for equiv in 0..self.num_equivariance {
                let index = feature * self.num_equivariance + equiv;
                out_row[out_base + equiv] = evaluate_vector_weighted_monomials_ell16(
                    x_row,
                    &self.weighted_u1[index],
                    &self.weighted_u2[index],
                    &self.weighted_u3[index],
                );
            }
        }
    }

    fn vjp_one_row(
        &self,
        x: &[f64],
        grad_out: &[f64],
        node: usize,
        grad_x_row: &mut [f64],
        per_node_out: usize,
        out_offset: usize,
    ) {
        debug_assert_eq!(self.num_ell, 16);
        for feature in 0..self.num_features {
            let x_base = (node * self.num_features + feature) * 16;
            let x_row = &x[x_base..x_base + 16];
            let grad_out_base = node * per_node_out + out_offset + feature * self.num_equivariance;
            let mut grad_values = [0.0; 16];
            for equiv in 0..self.num_equivariance {
                let bar_output = grad_out[grad_out_base + equiv];
                if bar_output == 0.0 {
                    continue;
                }
                let index = feature * self.num_equivariance + equiv;
                accumulate_vector_weighted_monomials_ell16(
                    bar_output,
                    x_row,
                    &self.weighted_u1[index],
                    &self.weighted_u2[index],
                    &self.weighted_u3[index],
                    &mut grad_values,
                );
            }
            let row_offset = feature * 16;
            add_ell16_grad_to_row(&mut grad_x_row[row_offset..row_offset + 16], &grad_values);
        }
    }
}

impl MaceScalarVectorSingleElementProductPlan {
    pub fn forward(&self, x: &[f64], out: &mut [f64]) -> Result<()> {
        let num_nodes = self.check_shapes(
            x.len(),
            out.len(),
            "mace scalar+vector prepared product forward out",
        )?;
        out.fill(0.0);
        let per_node_scalar = self.scalar.num_features;
        let per_node_out = self.per_node_out();
        self.scalar.forward_rows(x, out, num_nodes, per_node_out, 0);
        self.vector
            .forward_rows(x, out, num_nodes, per_node_out, per_node_scalar);
        Ok(())
    }

    pub fn forward_f32(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        let num_nodes = self.check_shapes(
            x.len(),
            out.len(),
            "mace scalar+vector prepared product forward_f32 out",
        )?;
        out.fill(0.0);
        let per_node_scalar = self.scalar.num_features;
        let per_node_out = self.per_node_out();
        self.forward_rows_f32(x, out, num_nodes, per_node_out, per_node_scalar);
        Ok(())
    }

    pub fn vjp_x(&self, x: &[f64], grad_out: &[f64], grad_x: &mut [f64]) -> Result<()> {
        let num_nodes = self.check_shapes(
            x.len(),
            grad_out.len(),
            "mace scalar+vector prepared product vjp grad_out",
        )?;
        self.scalar.check_grad_x_len(
            x.len(),
            grad_x.len(),
            "mace scalar+vector prepared product vjp grad_x",
        )?;
        let per_node_x = self.scalar.num_features * 16;
        let per_node_scalar = self.scalar.num_features;
        let per_node_out = self.per_node_out();
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && num_nodes > 1 {
            grad_x
                .par_chunks_mut(per_node_x)
                .enumerate()
                .for_each(|(node, grad_x_row)| {
                    self.scalar
                        .vjp_one_row(x, grad_out, node, grad_x_row, per_node_out, 0);
                    self.vector.vjp_one_row(
                        x,
                        grad_out,
                        node,
                        grad_x_row,
                        per_node_out,
                        per_node_scalar,
                    );
                });
            return Ok(());
        }
        for node in 0..num_nodes {
            let grad_x_row = &mut grad_x[node * per_node_x..(node + 1) * per_node_x];
            self.scalar
                .vjp_one_row(x, grad_out, node, grad_x_row, per_node_out, 0);
            self.vector
                .vjp_one_row(x, grad_out, node, grad_x_row, per_node_out, per_node_scalar);
        }
        Ok(())
    }

    pub fn vjp_x_f32(&self, x: &[f32], grad_out: &[f32], grad_x: &mut [f32]) -> Result<()> {
        let num_nodes = self.check_shapes(
            x.len(),
            grad_out.len(),
            "mace scalar+vector prepared product vjp_x_f32 grad_out",
        )?;
        self.scalar.check_grad_x_len(
            x.len(),
            grad_x.len(),
            "mace scalar+vector prepared product vjp_x_f32 grad_x",
        )?;
        let per_node_x = self.scalar.num_features * 16;
        let per_node_scalar = self.scalar.num_features;
        let per_node_out = self.per_node_out();
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && num_nodes > 1 {
            let chunk_rows = mace_product_f32_parallel_chunk_rows(num_nodes);
            grad_x
                .par_chunks_mut(per_node_x * chunk_rows)
                .enumerate()
                .for_each(|(chunk, grad_x_chunk)| {
                    let node_start = chunk * chunk_rows;
                    for (local_node, grad_x_row) in grad_x_chunk.chunks_mut(per_node_x).enumerate()
                    {
                        let node = node_start + local_node;
                        self.vjp_one_row_f32(
                            x,
                            grad_out,
                            node,
                            grad_x_row,
                            per_node_out,
                            per_node_scalar,
                        );
                    }
                });
            return Ok(());
        }
        for node in 0..num_nodes {
            let grad_x_row = &mut grad_x[node * per_node_x..(node + 1) * per_node_x];
            self.vjp_one_row_f32(x, grad_out, node, grad_x_row, per_node_out, per_node_scalar);
        }
        Ok(())
    }

    fn forward_rows_f32(
        &self,
        x: &[f32],
        out: &mut [f32],
        num_nodes: usize,
        per_node_out: usize,
        per_node_scalar: usize,
    ) {
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && num_nodes > 1 {
            let chunk_rows = mace_product_f32_parallel_chunk_rows(num_nodes);
            out.par_chunks_mut(per_node_out * chunk_rows)
                .enumerate()
                .for_each(|(chunk, out_chunk)| {
                    let node_start = chunk * chunk_rows;
                    for (local_node, out_row) in out_chunk.chunks_mut(per_node_out).enumerate() {
                        self.forward_one_row_f32(
                            x,
                            node_start + local_node,
                            out_row,
                            per_node_scalar,
                        );
                    }
                });
            return;
        }
        for node in 0..num_nodes {
            let out_row = &mut out[node * per_node_out..(node + 1) * per_node_out];
            self.forward_one_row_f32(x, node, out_row, per_node_scalar);
        }
    }

    fn forward_one_row_f32(
        &self,
        x: &[f32],
        node: usize,
        out_row: &mut [f32],
        per_node_scalar: usize,
    ) {
        debug_assert_eq!(self.scalar.num_ell, 16);
        debug_assert_eq!(self.vector.num_ell, 16);
        for feature in 0..self.scalar.num_features {
            let x_base = (node * self.scalar.num_features + feature) * 16;
            let x_row = &x[x_base..x_base + 16];
            let (weighted_u1, weighted_u2, weighted_u3) = self.scalar.weighted_slices_f32(feature);
            out_row[feature] = evaluate_scalar_weighted_monomials_ell16_f32_prepared(
                x_row,
                &self.scalar.u1_entries,
                weighted_u1,
                &self.scalar.u2_grouped_entries,
                weighted_u2,
                &self.scalar.u3_grouped_entries,
                weighted_u3,
            );
            let vector_out_base = per_node_scalar + feature * self.vector.num_equivariance;
            let index_base = feature * self.vector.num_equivariance;
            if self.vector.num_equivariance == 3 {
                let (weighted_u1, weighted_u2, weighted_u3) =
                    self.vector.weighted_slices_f32(index_base);
                out_row[vector_out_base] = evaluate_vector_weighted_monomials_ell16_f32(
                    x_row,
                    weighted_u1,
                    weighted_u2,
                    weighted_u3,
                );
                let (weighted_u1, weighted_u2, weighted_u3) =
                    self.vector.weighted_slices_f32(index_base + 1);
                out_row[vector_out_base + 1] = evaluate_vector_weighted_monomials_ell16_f32(
                    x_row,
                    weighted_u1,
                    weighted_u2,
                    weighted_u3,
                );
                let (weighted_u1, weighted_u2, weighted_u3) =
                    self.vector.weighted_slices_f32(index_base + 2);
                out_row[vector_out_base + 2] = evaluate_vector_weighted_monomials_ell16_f32(
                    x_row,
                    weighted_u1,
                    weighted_u2,
                    weighted_u3,
                );
            } else {
                for equiv in 0..self.vector.num_equivariance {
                    let index = index_base + equiv;
                    let (weighted_u1, weighted_u2, weighted_u3) =
                        self.vector.weighted_slices_f32(index);
                    out_row[vector_out_base + equiv] = evaluate_vector_weighted_monomials_ell16_f32(
                        x_row,
                        weighted_u1,
                        weighted_u2,
                        weighted_u3,
                    );
                }
            }
        }
    }

    fn vjp_one_row_f32(
        &self,
        x: &[f32],
        grad_out: &[f32],
        node: usize,
        grad_x_row: &mut [f32],
        per_node_out: usize,
        per_node_scalar: usize,
    ) {
        debug_assert_eq!(self.scalar.num_ell, 16);
        debug_assert_eq!(self.vector.num_ell, 16);
        for feature in 0..self.scalar.num_features {
            let x_base = (node * self.scalar.num_features + feature) * 16;
            let x_row = &x[x_base..x_base + 16];
            let row_offset = feature * 16;
            let mut grad_values = [0.0_f32; 16];
            let scalar_bar = grad_out[node * per_node_out + feature];
            if scalar_bar != 0.0 {
                let (weighted_u1, weighted_u2, weighted_u3) =
                    self.scalar.weighted_slices_f32(feature);
                accumulate_scalar_weighted_monomials_ell16_f32_prepared_into(
                    scalar_bar,
                    x_row,
                    &self.scalar.u1_entries,
                    weighted_u1,
                    &self.scalar.u2_grouped_entries,
                    weighted_u2,
                    &self.scalar.u3_grouped_entries,
                    weighted_u3,
                    &mut grad_values,
                );
            }
            let vector_bar_base =
                node * per_node_out + per_node_scalar + feature * self.vector.num_equivariance;
            let index_base = feature * self.vector.num_equivariance;
            if self.vector.num_equivariance == 3 {
                let bar_output = grad_out[vector_bar_base];
                if bar_output != 0.0 {
                    let (weighted_u1, weighted_u2, weighted_u3) =
                        self.vector.weighted_slices_f32(index_base);
                    accumulate_vector_weighted_monomials_ell16_f32_into(
                        bar_output,
                        x_row,
                        weighted_u1,
                        weighted_u2,
                        weighted_u3,
                        &mut grad_values,
                    );
                }
                let bar_output = grad_out[vector_bar_base + 1];
                if bar_output != 0.0 {
                    let (weighted_u1, weighted_u2, weighted_u3) =
                        self.vector.weighted_slices_f32(index_base + 1);
                    accumulate_vector_weighted_monomials_ell16_f32_into(
                        bar_output,
                        x_row,
                        weighted_u1,
                        weighted_u2,
                        weighted_u3,
                        &mut grad_values,
                    );
                }
                let bar_output = grad_out[vector_bar_base + 2];
                if bar_output != 0.0 {
                    let (weighted_u1, weighted_u2, weighted_u3) =
                        self.vector.weighted_slices_f32(index_base + 2);
                    accumulate_vector_weighted_monomials_ell16_f32_into(
                        bar_output,
                        x_row,
                        weighted_u1,
                        weighted_u2,
                        weighted_u3,
                        &mut grad_values,
                    );
                }
            } else {
                for equiv in 0..self.vector.num_equivariance {
                    let bar_output = grad_out[vector_bar_base + equiv];
                    if bar_output == 0.0 {
                        continue;
                    }
                    let index = index_base + equiv;
                    let (weighted_u1, weighted_u2, weighted_u3) =
                        self.vector.weighted_slices_f32(index);
                    accumulate_vector_weighted_monomials_ell16_f32_into(
                        bar_output,
                        x_row,
                        weighted_u1,
                        weighted_u2,
                        weighted_u3,
                        &mut grad_values,
                    );
                }
            }
            add_ell16_grad_to_row_f32(&mut grad_x_row[row_offset..row_offset + 16], &grad_values);
        }
    }

    fn check_shapes(&self, x_len: usize, out_len: usize, out_op: &'static str) -> Result<usize> {
        if self.scalar.num_ell != 16 || self.vector.num_ell != 16 {
            return Err(CoreError::InvalidTensorProduct(
                "mace scalar+vector prepared product plan currently requires num_ell == 16"
                    .to_owned(),
            ));
        }
        if self.scalar.num_features != self.vector.num_features {
            return Err(CoreError::InvalidTensorProduct(
                "mace scalar+vector prepared product feature mismatch".to_owned(),
            ));
        }
        let per_node_x = self.scalar.num_features * 16;
        if !x_len.is_multiple_of(per_node_x) {
            return Err(CoreError::ShapeMismatch {
                op: "mace scalar+vector prepared product x",
                expected: per_node_x,
                actual: x_len,
            });
        }
        let num_nodes = x_len / per_node_x;
        let expected = num_nodes * self.per_node_out();
        if out_len != expected {
            return Err(CoreError::ShapeMismatch {
                op: out_op,
                expected,
                actual: out_len,
            });
        }
        Ok(num_nodes)
    }

    fn per_node_out(&self) -> usize {
        self.scalar.num_features + self.vector.num_features * self.vector.num_equivariance
    }
}

#[derive(Clone, Copy, Debug)]
struct ScalarU1Entry {
    ell: usize,
    param: usize,
    value: f64,
}

#[derive(Clone, Copy, Debug)]
struct ScalarU2Entry {
    left: usize,
    right: usize,
    param: usize,
    value: f64,
}

#[derive(Clone, Copy, Debug)]
struct ScalarU2WeightedEntry {
    left: usize,
    right: usize,
}

#[derive(Clone, Debug)]
struct ScalarU2PairGroup {
    left: usize,
    right: usize,
    terms: Vec<(usize, f64)>,
}

#[derive(Clone, Copy, Debug)]
struct ScalarU3Entry {
    left: usize,
    right: usize,
    inner: usize,
    param: usize,
    value: f64,
}

#[derive(Clone, Copy, Debug)]
struct ScalarU3WeightedEntry {
    left: usize,
    right: usize,
    inner: usize,
}

#[derive(Clone, Debug)]
struct ScalarU3TripleGroup {
    left: usize,
    right: usize,
    inner: usize,
    terms: Vec<(usize, f64)>,
}

#[derive(Clone, Copy, Debug)]
struct VectorU1Entry {
    ell: usize,
    param: usize,
    value: f64,
}

#[derive(Clone, Copy, Debug)]
struct VectorU2Entry {
    left: usize,
    right: usize,
    param: usize,
    value: f64,
}

#[derive(Clone, Copy, Debug)]
struct VectorU3Entry {
    left: usize,
    right: usize,
    inner: usize,
    param: usize,
    value: f64,
}

#[derive(Clone, Debug)]
struct VectorU1WeightedEntry {
    ell: usize,
    value: f64,
}

#[derive(Clone, Debug)]
struct VectorU1WeightedEntryF32 {
    ell: usize,
    value: f32,
}

#[derive(Clone, Debug)]
struct VectorU2PairGroup {
    left: usize,
    right: usize,
    terms: Vec<(usize, f64)>,
}

#[derive(Clone, Debug)]
struct VectorU2WeightedEntry {
    left: usize,
    right: usize,
    value: f64,
}

#[derive(Clone, Debug)]
struct VectorU2WeightedEntryF32 {
    left: usize,
    right: usize,
    value: f32,
}

#[derive(Clone, Debug)]
struct VectorU3TripleGroup {
    left: usize,
    right: usize,
    inner: usize,
    terms: Vec<(usize, f64)>,
}

#[derive(Clone, Debug)]
struct VectorU3WeightedEntry {
    left: usize,
    right: usize,
    inner: usize,
    value: f64,
}

#[derive(Clone, Debug)]
struct VectorU3WeightedEntryF32 {
    left: usize,
    right: usize,
    inner: usize,
    value: f32,
}

#[cfg(feature = "parallel")]
#[derive(Clone, Debug)]
struct ScalarWeightedEntriesFlat {
    u1: Vec<f64>,
    u2: Vec<f64>,
    u3: Vec<f64>,
    u1_len: usize,
    u2_len: usize,
    u3_len: usize,
    num_features: usize,
}

#[cfg(feature = "parallel")]
impl ScalarWeightedEntriesFlat {
    #[inline(always)]
    fn slices(&self, dense_element: usize, feature: usize) -> (&[f64], &[f64], &[f64]) {
        let row = dense_element * self.num_features + feature;
        let u1_base = row * self.u1_len;
        let u2_base = row * self.u2_len;
        let u3_base = row * self.u3_len;
        (
            &self.u1[u1_base..u1_base + self.u1_len],
            &self.u2[u2_base..u2_base + self.u2_len],
            &self.u3[u3_base..u3_base + self.u3_len],
        )
    }
}

impl<'a> MaceScalarSymmetricContraction<'a> {
    pub fn validate(&self) -> Result<()> {
        self.check_len(
            "mace_symmetric_contraction u1",
            self.num_ell * self.u1_params,
            self.u1.len(),
        )?;
        self.check_len(
            "mace_symmetric_contraction u2",
            self.num_ell * self.num_ell * self.u2_params,
            self.u2.len(),
        )?;
        self.check_len(
            "mace_symmetric_contraction u3",
            self.num_ell * self.num_ell * self.num_ell * self.u3_params,
            self.u3.len(),
        )?;
        self.check_len(
            "mace_symmetric_contraction weights_u1",
            self.num_elements * self.u1_params * self.num_features,
            self.weights_u1.len(),
        )?;
        self.check_len(
            "mace_symmetric_contraction weights_u2",
            self.num_elements * self.u2_params * self.num_features,
            self.weights_u2.len(),
        )?;
        self.check_len(
            "mace_symmetric_contraction weights_u3",
            self.num_elements * self.u3_params * self.num_features,
            self.weights_u3.len(),
        )?;
        validate_nonzero("mace_symmetric_contraction num_elements", self.num_elements)?;
        validate_nonzero("mace_symmetric_contraction num_features", self.num_features)?;
        validate_nonzero("mace_symmetric_contraction num_ell", self.num_ell)?;
        validate_nonzero("mace_symmetric_contraction u1_params", self.u1_params)?;
        validate_nonzero("mace_symmetric_contraction u2_params", self.u2_params)?;
        validate_nonzero("mace_symmetric_contraction u3_params", self.u3_params)?;
        Ok(())
    }

    pub fn entries(&self) -> Result<MaceScalarSymmetricContractionEntries> {
        self.validate()?;
        Ok(self.nonzero_entries())
    }

    pub fn prepare_single_element_product_plan(
        &self,
    ) -> Result<MaceScalarSingleElementProductPlan> {
        self.validate()?;
        if self.num_elements != 1 {
            return Err(CoreError::InvalidTensorProduct(
                "mace scalar single-element product plan requires num_elements == 1".to_owned(),
            ));
        }
        let owned_entries;
        let entries = if let Some(entries) = self.entries {
            entries
        } else {
            owned_entries = self.nonzero_entries();
            &owned_entries
        };
        let u1_entries = entries.u1.clone();
        let u2_entries = entries.u2.clone();
        let u3_entries = entries.u3.clone();
        let u2_groups = scalar_u2_entries_by_pair(&entries.u2);
        let u2_grouped_entries = u2_groups
            .iter()
            .map(|group| ScalarU2WeightedEntry {
                left: group.left,
                right: group.right,
            })
            .collect::<Vec<_>>();
        let u3_groups = scalar_u3_entries_by_triple(&entries.u3);
        let u3_grouped_entries = u3_groups
            .iter()
            .map(|group| ScalarU3WeightedEntry {
                left: group.left,
                right: group.right,
                inner: group.inner,
            })
            .collect::<Vec<_>>();
        let u1_len = u1_entries.len();
        let u2_len = u2_entries.len();
        let u3_len = u3_entries.len();
        let u2_grouped_len = u2_groups.len();
        let u3_grouped_len = u3_groups.len();
        let mut weighted_u1 = vec![0.0; self.num_features * u1_len];
        let mut weighted_u2 = vec![0.0; self.num_features * u2_len];
        let mut weighted_u3 = vec![0.0; self.num_features * u3_len];
        let mut weighted_u2_grouped = vec![0.0; self.num_features * u2_grouped_len];
        let mut weighted_u3_grouped = vec![0.0; self.num_features * u3_grouped_len];
        let mut weighted_u2_grouped_f32 = vec![0.0; self.num_features * u2_grouped_len];
        let mut weighted_u3_grouped_f32 = vec![0.0; self.num_features * u3_grouped_len];
        let mut w1 = vec![0.0; self.u1_params];
        let mut w2 = vec![0.0; self.u2_params];
        let mut w3 = vec![0.0; self.u3_params];
        for feature in 0..self.num_features {
            self.feature_weights_for_element_into(0, feature, &mut w1, &mut w2, &mut w3);
            let u1_base = feature * u1_len;
            let u2_base = feature * u2_len;
            let u3_base = feature * u3_len;
            weight_scalar_entries(
                &w1,
                &w2,
                &w3,
                &u1_entries,
                &u2_entries,
                &u3_entries,
                &mut weighted_u1[u1_base..u1_base + u1_len],
                &mut weighted_u2[u2_base..u2_base + u2_len],
                &mut weighted_u3[u3_base..u3_base + u3_len],
            );
            let u2_grouped_base = feature * u2_grouped_len;
            weight_scalar_u2_groups(
                &w2,
                &u2_groups,
                &mut weighted_u2_grouped[u2_grouped_base..u2_grouped_base + u2_grouped_len],
            );
            weight_scalar_u2_groups_f32(
                &w2,
                &u2_groups,
                &mut weighted_u2_grouped_f32[u2_grouped_base..u2_grouped_base + u2_grouped_len],
            );
            let u3_grouped_base = feature * u3_grouped_len;
            weight_scalar_u3_groups(
                &w3,
                &u3_groups,
                &mut weighted_u3_grouped[u3_grouped_base..u3_grouped_base + u3_grouped_len],
            );
            weight_scalar_u3_groups_f32(
                &w3,
                &u3_groups,
                &mut weighted_u3_grouped_f32[u3_grouped_base..u3_grouped_base + u3_grouped_len],
            );
        }
        let weighted_u1_f32: Vec<f32> = weighted_u1.iter().map(|&value| value as f32).collect();
        Ok(MaceScalarSingleElementProductPlan {
            num_features: self.num_features,
            num_ell: self.num_ell,
            u1_entries,
            u2_entries,
            u2_grouped_entries,
            u3_entries,
            u3_grouped_entries,
            weighted_u1,
            weighted_u2,
            weighted_u3,
            weighted_u2_grouped,
            weighted_u3_grouped,
            weighted_u1_f32,
            weighted_u2_grouped_f32,
            weighted_u3_grouped_f32,
            u1_len,
            u2_len,
            u2_grouped_len,
            u3_len,
            u3_grouped_len,
        })
    }

    pub fn forward(&self, x: &[f64], node_attrs: &[f64], out: &mut [f64]) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes(
            x,
            node_attrs,
            out.len(),
            "mace_symmetric_contraction_forward out",
        )?;
        out.fill(0.0);
        let mut w1 = vec![0.0; self.u1_params];
        let mut w2 = vec![0.0; self.u2_params];
        let mut w3 = vec![0.0; self.u3_params];
        let owned_entries;
        let entries = if let Some(entries) = self.entries {
            entries
        } else {
            owned_entries = self.nonzero_entries();
            &owned_entries
        };
        let u1_entries = &entries.u1;
        let u2_entries = &entries.u2;
        let u3_entries = &entries.u3;
        if let Some(active_elements) =
            active_unit_one_hot_elements(node_attrs, num_nodes, self.num_elements)
        {
            #[cfg(target_arch = "x86_64")]
            if self.num_ell == 16 && mace_f64_avx2_enabled() {
                let node_x_len = self.num_features * 16;
                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() > 1 && num_nodes >= 32 {
                    out.par_chunks_mut(self.num_features)
                        .enumerate()
                        .with_min_len(2)
                        .for_each(|(node, out_row)| {
                            mace_scalar_contraction_forward_node_avx2_f64_safe(
                                self,
                                &x[node * node_x_len..(node + 1) * node_x_len],
                                active_elements[node],
                                out_row,
                                u1_entries,
                                u2_entries,
                                u3_entries,
                            );
                        });
                    return Ok(());
                }
                for (node, out_row) in out.chunks_mut(self.num_features).enumerate() {
                    mace_scalar_contraction_forward_node_avx2_f64_safe(
                        self,
                        &x[node * node_x_len..(node + 1) * node_x_len],
                        active_elements[node],
                        out_row,
                        u1_entries,
                        u2_entries,
                        u3_entries,
                    );
                }
                return Ok(());
            }
            let mut nodes_by_element = vec![Vec::new(); self.num_elements];
            for (node, &element) in active_elements.iter().enumerate() {
                nodes_by_element[element].push(node);
            }
            let active_node_groups = nodes_by_element
                .into_iter()
                .enumerate()
                .filter(|(_, nodes)| !nodes.is_empty())
                .collect::<Vec<_>>();
            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() > 1 && self.num_ell == 16 && num_nodes > 1 {
                let (active_element_list, element_to_dense) =
                    active_element_dense_map(&active_elements, self.num_elements);
                let weighted_entries = self.weighted_entries_by_active_element_feature(
                    &active_element_list,
                    u1_entries,
                    u2_entries,
                    u3_entries,
                );
                out.par_chunks_mut(self.num_features)
                    .enumerate()
                    .for_each(|(node, out_row)| {
                        let dense_element = element_to_dense[active_elements[node]];
                        debug_assert_ne!(dense_element, usize::MAX);
                        for (feature, output) in out_row.iter_mut().enumerate() {
                            let (weighted_u1, weighted_u2, weighted_u3) =
                                weighted_entries.slices(dense_element, feature);
                            let x_base = self.x_index(node, feature, 0);
                            let x_row = &x[x_base..x_base + 16];
                            *output = evaluate_scalar_weighted_monomials_ell16(
                                x_row,
                                u1_entries,
                                weighted_u1,
                                u2_entries,
                                weighted_u2,
                                u3_entries,
                                weighted_u3,
                            );
                        }
                    });
                return Ok(());
            }
            let mut weighted_u1 = vec![0.0; u1_entries.len()];
            let mut weighted_u2 = vec![0.0; u2_entries.len()];
            let mut weighted_u3 = vec![0.0; u3_entries.len()];
            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() == 1 && self.num_ell == 16 {
                for feature in 0..self.num_features {
                    for (element, nodes) in &active_node_groups {
                        self.feature_weights_for_element_into(
                            *element, feature, &mut w1, &mut w2, &mut w3,
                        );
                        weight_scalar_entries(
                            &w1,
                            &w2,
                            &w3,
                            u1_entries,
                            u2_entries,
                            u3_entries,
                            &mut weighted_u1,
                            &mut weighted_u2,
                            &mut weighted_u3,
                        );
                        for &node in nodes {
                            let x_base = self.x_index(node, feature, 0);
                            let x_row = &x[x_base..x_base + 16];
                            out[node * self.num_features + feature] =
                                evaluate_scalar_weighted_monomials_ell16(
                                    x_row,
                                    u1_entries,
                                    &weighted_u1,
                                    u2_entries,
                                    &weighted_u2,
                                    u3_entries,
                                    &weighted_u3,
                                );
                        }
                    }
                }
                return Ok(());
            }
            for feature in 0..self.num_features {
                for (element, nodes) in &active_node_groups {
                    self.feature_weights_for_element_into(
                        *element, feature, &mut w1, &mut w2, &mut w3,
                    );
                    weight_scalar_entries(
                        &w1,
                        &w2,
                        &w3,
                        u1_entries,
                        u2_entries,
                        u3_entries,
                        &mut weighted_u1,
                        &mut weighted_u2,
                        &mut weighted_u3,
                    );
                    for &node in nodes {
                        out[node * self.num_features + feature] = self
                            .forward_node_feature_with_weighted_entries(
                                x,
                                node,
                                feature,
                                u1_entries,
                                &weighted_u1,
                                u2_entries,
                                &weighted_u2,
                                u3_entries,
                                &weighted_u3,
                            );
                    }
                }
            }
            return Ok(());
        }
        for node in 0..num_nodes {
            for feature in 0..self.num_features {
                out[node * self.num_features + feature] = self.forward_node_feature(
                    x, node_attrs, node, feature, &mut w1, &mut w2, &mut w3, u1_entries,
                    u2_entries, u3_entries,
                );
            }
        }
        Ok(())
    }

    pub fn forward_f32(&self, x: &[f32], node_attrs: &[f32], out: &mut [f32]) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes_f32(
            x,
            node_attrs,
            out.len(),
            "mace_symmetric_contraction_forward_f32 out",
        )?;
        out.fill(0.0);
        let mut w1 = vec![0.0_f32; self.u1_params];
        let mut w2 = vec![0.0_f32; self.u2_params];
        let mut w3 = vec![0.0_f32; self.u3_params];
        let owned_entries;
        let entries = if let Some(entries) = self.entries {
            entries
        } else {
            owned_entries = self.nonzero_entries();
            &owned_entries
        };
        let u1_entries = &entries.u1;
        let u2_entries = &entries.u2;
        let u3_entries = &entries.u3;
        if let Some(active_elements) =
            active_unit_one_hot_elements_f32(node_attrs, num_nodes, self.num_elements)
        {
            let mut nodes_by_element = vec![Vec::new(); self.num_elements];
            for (node, &element) in active_elements.iter().enumerate() {
                nodes_by_element[element].push(node);
            }
            for feature in 0..self.num_features {
                for (element, nodes) in nodes_by_element.iter().enumerate() {
                    if nodes.is_empty() {
                        continue;
                    }
                    self.feature_weights_for_element_into_f32(
                        element, feature, &mut w1, &mut w2, &mut w3,
                    );
                    for &node in nodes {
                        out[node * self.num_features + feature] = self
                            .forward_node_feature_with_weights_f32(
                                x, node, feature, &w1, &w2, &w3, u1_entries, u2_entries, u3_entries,
                            );
                    }
                }
            }
            return Ok(());
        }
        for node in 0..num_nodes {
            for feature in 0..self.num_features {
                self.feature_weights_into_f32(node_attrs, node, feature, &mut w1, &mut w2, &mut w3);
                out[node * self.num_features + feature] = self
                    .forward_node_feature_with_weights_f32(
                        x, node, feature, &w1, &w2, &w3, u1_entries, u2_entries, u3_entries,
                    );
            }
        }
        Ok(())
    }

    pub fn vjp_x_f32(
        &self,
        x: &[f32],
        node_attrs: &[f32],
        grad_out: &[f32],
        grad_x: &mut [f32],
    ) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes_f32(
            x,
            node_attrs,
            grad_out.len(),
            "mace_symmetric_contraction_vjp_x_f32 grad_out",
        )?;
        self.check_len(
            "mace_symmetric_contraction_vjp_x_f32 grad_x",
            x.len(),
            grad_x.len(),
        )?;
        let mut w1 = vec![0.0_f32; self.u1_params];
        let mut w2 = vec![0.0_f32; self.u2_params];
        let mut w3 = vec![0.0_f32; self.u3_params];
        let owned_entries;
        let entries = if let Some(entries) = self.entries {
            entries
        } else {
            owned_entries = self.nonzero_entries();
            &owned_entries
        };
        let u1_entries = &entries.u1;
        let u2_entries = &entries.u2;
        let u3_entries = &entries.u3;
        if let Some(active_elements) =
            active_unit_one_hot_elements_f32(node_attrs, num_nodes, self.num_elements)
        {
            let mut nodes_by_element = vec![Vec::new(); self.num_elements];
            for (node, &element) in active_elements.iter().enumerate() {
                nodes_by_element[element].push(node);
            }
            for feature in 0..self.num_features {
                for (element, nodes) in nodes_by_element.iter().enumerate() {
                    if nodes.is_empty()
                        || !nodes
                            .iter()
                            .any(|&node| grad_out[node * self.num_features + feature] != 0.0)
                    {
                        continue;
                    }
                    self.feature_weights_for_element_into_f32(
                        element, feature, &mut w1, &mut w2, &mut w3,
                    );
                    for &node in nodes {
                        self.vjp_node_feature_with_weights_f32(
                            x, grad_out, grad_x, node, feature, &w1, &w2, &w3, u1_entries,
                            u2_entries, u3_entries,
                        );
                    }
                }
            }
            return Ok(());
        }
        for node in 0..num_nodes {
            for feature in 0..self.num_features {
                if grad_out[node * self.num_features + feature] == 0.0 {
                    continue;
                }
                self.feature_weights_into_f32(node_attrs, node, feature, &mut w1, &mut w2, &mut w3);
                self.vjp_node_feature_with_weights_f32(
                    x, grad_out, grad_x, node, feature, &w1, &w2, &w3, u1_entries, u2_entries,
                    u3_entries,
                );
            }
        }
        Ok(())
    }

    pub fn vjp_x(
        &self,
        x: &[f64],
        node_attrs: &[f64],
        grad_out: &[f64],
        grad_x: &mut [f64],
    ) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes(
            x,
            node_attrs,
            grad_out.len(),
            "mace_symmetric_contraction_vjp_x grad_out",
        )?;
        self.check_len(
            "mace_symmetric_contraction_vjp_x grad_x",
            x.len(),
            grad_x.len(),
        )?;
        let mut w1 = vec![0.0; self.u1_params];
        let mut w2 = vec![0.0; self.u2_params];
        let mut w3 = vec![0.0; self.u3_params];
        let mut grad_values = vec![0.0; self.num_ell];
        let owned_entries;
        let entries = if let Some(entries) = self.entries {
            entries
        } else {
            owned_entries = self.nonzero_entries();
            &owned_entries
        };
        let u1_entries = &entries.u1;
        let u2_entries = &entries.u2;
        let u3_entries = &entries.u3;
        if let Some(active_elements) =
            active_unit_one_hot_elements(node_attrs, num_nodes, self.num_elements)
        {
            #[cfg(target_arch = "x86_64")]
            if self.num_ell == 16 && mace_f64_avx2_enabled() {
                let node_x_len = self.num_features * 16;
                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() > 1 && num_nodes >= 32 {
                    grad_x
                        .par_chunks_mut(node_x_len)
                        .enumerate()
                        .with_min_len(2)
                        .for_each(|(node, grad_x_node)| {
                            mace_scalar_contraction_vjp_node_avx2_f64_safe(
                                self,
                                &x[node * node_x_len..(node + 1) * node_x_len],
                                &grad_out[node * self.num_features..(node + 1) * self.num_features],
                                active_elements[node],
                                grad_x_node,
                                u1_entries,
                                u2_entries,
                                u3_entries,
                            );
                        });
                    return Ok(());
                }
                for (node, grad_x_node) in grad_x.chunks_mut(node_x_len).enumerate() {
                    mace_scalar_contraction_vjp_node_avx2_f64_safe(
                        self,
                        &x[node * node_x_len..(node + 1) * node_x_len],
                        &grad_out[node * self.num_features..(node + 1) * self.num_features],
                        active_elements[node],
                        grad_x_node,
                        u1_entries,
                        u2_entries,
                        u3_entries,
                    );
                }
                return Ok(());
            }
            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() > 1 && self.num_ell == 16 && num_nodes > 1 {
                let (active_element_list, element_to_dense) =
                    active_element_dense_map(&active_elements, self.num_elements);
                let weighted_entries = self.weighted_entries_by_active_element_feature(
                    &active_element_list,
                    u1_entries,
                    u2_entries,
                    u3_entries,
                );
                grad_x
                    .par_chunks_mut(self.num_features * 16)
                    .enumerate()
                    .for_each(|(node, grad_x_node)| {
                        let dense_element = element_to_dense[active_elements[node]];
                        debug_assert_ne!(dense_element, usize::MAX);
                        for feature in 0..self.num_features {
                            let (weighted_u1, weighted_u2, weighted_u3) =
                                weighted_entries.slices(dense_element, feature);
                            if let Some(grad) = self.vjp_node_feature_grad_with_weighted_ell16(
                                x,
                                grad_out,
                                node,
                                feature,
                                u1_entries,
                                weighted_u1,
                                u2_entries,
                                weighted_u2,
                                u3_entries,
                                weighted_u3,
                            ) {
                                let offset = feature * 16;
                                for (target, value) in
                                    grad_x_node[offset..offset + 16].iter_mut().zip(grad)
                                {
                                    *target += value;
                                }
                            }
                        }
                    });
                return Ok(());
            }
            let mut nodes_by_element = vec![Vec::new(); self.num_elements];
            for (node, &element) in active_elements.iter().enumerate() {
                nodes_by_element[element].push(node);
            }
            let active_node_groups = nodes_by_element
                .into_iter()
                .enumerate()
                .filter(|(_, nodes)| !nodes.is_empty())
                .collect::<Vec<_>>();
            #[cfg(feature = "parallel")]
            if self.num_ell == 16
                && num_nodes > 1
                && self.num_features >= 128
                && rayon::current_num_threads() > 1
            {
                let features = (0..self.num_features).collect::<Vec<_>>();
                let updates_by_chunk = features
                    .par_chunks(8)
                    .map(|feature_chunk| {
                        let mut local_w1 = vec![0.0; self.u1_params];
                        let mut local_w2 = vec![0.0; self.u2_params];
                        let mut local_w3 = vec![0.0; self.u3_params];
                        let mut updates = Vec::with_capacity(feature_chunk.len() * num_nodes);
                        for &feature in feature_chunk {
                            for (element, nodes) in &active_node_groups {
                                self.feature_weights_for_element_into(
                                    *element,
                                    feature,
                                    &mut local_w1,
                                    &mut local_w2,
                                    &mut local_w3,
                                );
                                for &node in nodes {
                                    if let Some(grad) = self
                                        .vjp_node_feature_grad_with_weights_ell16(
                                            x, grad_out, node, feature, &local_w1, &local_w2,
                                            &local_w3, u1_entries, u2_entries, u3_entries,
                                        )
                                    {
                                        updates.push((self.x_index(node, feature, 0), grad));
                                    }
                                }
                            }
                        }
                        updates
                    })
                    .collect::<Vec<_>>();
                for updates in updates_by_chunk {
                    for (offset, grad) in updates {
                        for (target, value) in grad_x[offset..offset + 16].iter_mut().zip(grad) {
                            *target += value;
                        }
                    }
                }
                return Ok(());
            }
            // Fixed `l_max=3` MACE rows avoid dynamic scratch in the common
            // force VJP path even when Rayon is not enabled.
            #[cfg(feature = "parallel")]
            if self.num_ell == 16 && rayon::current_num_threads() == 1 {
                let u1_by_param = scalar_u1_entries_by_param(u1_entries, self.u1_params);
                let u2_by_param = scalar_u2_entries_by_param(u2_entries, self.u2_params);
                let u3_by_param = scalar_u3_entries_by_param(u3_entries, self.u3_params);
                for feature in 0..self.num_features {
                    for (element, nodes) in &active_node_groups {
                        if !nodes
                            .iter()
                            .any(|&node| grad_out[node * self.num_features + feature] != 0.0)
                        {
                            continue;
                        }
                        self.feature_weights_for_element_into(
                            *element, feature, &mut w1, &mut w2, &mut w3,
                        );
                        for &node in nodes {
                            let bar_output = grad_out[node * self.num_features + feature];
                            if bar_output == 0.0 {
                                continue;
                            }
                            let x_base = self.x_index(node, feature, 0);
                            let x_row = &x[x_base..x_base + 16];
                            let mut grad = [0.0_f64; 16];
                            accumulate_scalar_grouped_entries_ell16(
                                bar_output,
                                x_row,
                                &u1_by_param,
                                &w1,
                                &u2_by_param,
                                &w2,
                                &u3_by_param,
                                &w3,
                                &mut grad,
                            );
                            add_ell16_grad_to_x(grad_x, x_base, &grad);
                        }
                    }
                }
                return Ok(());
            }
            if self.num_ell == 16 {
                for feature in 0..self.num_features {
                    for (element, nodes) in &active_node_groups {
                        if !nodes
                            .iter()
                            .any(|&node| grad_out[node * self.num_features + feature] != 0.0)
                        {
                            continue;
                        }
                        self.feature_weights_for_element_into(
                            *element, feature, &mut w1, &mut w2, &mut w3,
                        );
                        for &node in nodes {
                            if let Some(grad) = self.vjp_node_feature_grad_with_weights_ell16(
                                x, grad_out, node, feature, &w1, &w2, &w3, u1_entries, u2_entries,
                                u3_entries,
                            ) {
                                let offset = self.x_index(node, feature, 0);
                                for (target, value) in
                                    grad_x[offset..offset + 16].iter_mut().zip(grad)
                                {
                                    *target += value;
                                }
                            }
                        }
                    }
                }
                return Ok(());
            }
            for feature in 0..self.num_features {
                for (element, nodes) in &active_node_groups {
                    if !nodes
                        .iter()
                        .any(|&node| grad_out[node * self.num_features + feature] != 0.0)
                    {
                        continue;
                    }
                    self.feature_weights_for_element_into(
                        *element, feature, &mut w1, &mut w2, &mut w3,
                    );
                    for &node in nodes {
                        self.vjp_node_feature_with_weights(
                            x,
                            grad_out,
                            grad_x,
                            node,
                            feature,
                            &w1,
                            &w2,
                            &w3,
                            &mut grad_values,
                            u1_entries,
                            u2_entries,
                            u3_entries,
                        );
                    }
                }
            }
            return Ok(());
        }
        for node in 0..num_nodes {
            for feature in 0..self.num_features {
                if grad_out[node * self.num_features + feature] == 0.0 {
                    continue;
                }
                self.vjp_node_feature(
                    x,
                    node_attrs,
                    grad_out,
                    grad_x,
                    node,
                    feature,
                    &mut w1,
                    &mut w2,
                    &mut w3,
                    &mut grad_values,
                    u1_entries,
                    u2_entries,
                    u3_entries,
                );
            }
        }
        Ok(())
    }

    pub fn jvp_x(
        &self,
        x: &[f64],
        x_dot: &[f64],
        node_attrs: &[f64],
        out_dot: &mut [f64],
    ) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes(
            x,
            node_attrs,
            out_dot.len(),
            "mace_symmetric_contraction_jvp_x out_dot",
        )?;
        self.check_len(
            "mace_symmetric_contraction_jvp_x x_dot",
            x.len(),
            x_dot.len(),
        )?;
        out_dot.fill(0.0);

        let mut w1 = vec![0.0; self.u1_params];
        let mut w2 = vec![0.0; self.u2_params];
        let mut w3 = vec![0.0; self.u3_params];
        let owned_entries;
        let entries = if let Some(entries) = self.entries {
            entries
        } else {
            owned_entries = self.nonzero_entries();
            &owned_entries
        };
        let u1_entries = &entries.u1;
        let u2_entries = &entries.u2;
        let u3_entries = &entries.u3;
        if let Some(active_elements) =
            active_unit_one_hot_elements(node_attrs, num_nodes, self.num_elements)
        {
            let mut nodes_by_element = vec![Vec::new(); self.num_elements];
            for (node, &element) in active_elements.iter().enumerate() {
                nodes_by_element[element].push(node);
            }
            let active_node_groups = nodes_by_element
                .into_iter()
                .enumerate()
                .filter(|(_, nodes)| !nodes.is_empty())
                .collect::<Vec<_>>();
            #[cfg(feature = "parallel")]
            {
                let active_element_list = active_node_groups
                    .iter()
                    .map(|(element, _)| *element)
                    .collect::<Vec<_>>();
                let weighted_entries = self.weighted_entries_by_active_element_feature(
                    &active_element_list,
                    u1_entries,
                    u2_entries,
                    u3_entries,
                );
                for feature in 0..self.num_features {
                    for (dense_element, (_, nodes)) in active_node_groups.iter().enumerate() {
                        if !nodes.iter().any(|&node| {
                            let x_base = self.x_index(node, feature, 0);
                            !all_values_zero(&x_dot[x_base..x_base + self.num_ell])
                        }) {
                            continue;
                        }
                        let (weighted_u1, weighted_u2, weighted_u3) =
                            weighted_entries.slices(dense_element, feature);
                        for &node in nodes {
                            let x_base = self.x_index(node, feature, 0);
                            if all_values_zero(&x_dot[x_base..x_base + self.num_ell]) {
                                continue;
                            }
                            out_dot[node * self.num_features + feature] = self
                                .jvp_node_feature_with_weighted_entries(
                                    x,
                                    x_dot,
                                    node,
                                    feature,
                                    u1_entries,
                                    weighted_u1,
                                    u2_entries,
                                    weighted_u2,
                                    u3_entries,
                                    weighted_u3,
                                );
                        }
                    }
                }
                return Ok(());
            }
            #[cfg(not(feature = "parallel"))]
            for feature in 0..self.num_features {
                for (element, nodes) in &active_node_groups {
                    if !nodes.iter().any(|&node| {
                        let x_base = self.x_index(node, feature, 0);
                        !all_values_zero(&x_dot[x_base..x_base + self.num_ell])
                    }) {
                        continue;
                    }
                    self.feature_weights_for_element_into(
                        *element, feature, &mut w1, &mut w2, &mut w3,
                    );
                    for &node in nodes {
                        let x_base = self.x_index(node, feature, 0);
                        if all_values_zero(&x_dot[x_base..x_base + self.num_ell]) {
                            continue;
                        }
                        out_dot[node * self.num_features + feature] = self
                            .jvp_node_feature_with_weights(
                                x, x_dot, node, feature, &w1, &w2, &w3, u1_entries, u2_entries,
                                u3_entries,
                            );
                    }
                }
            }
            #[cfg(not(feature = "parallel"))]
            return Ok(());
        }
        for node in 0..num_nodes {
            for feature in 0..self.num_features {
                let x_base = self.x_index(node, feature, 0);
                if all_values_zero(&x_dot[x_base..x_base + self.num_ell]) {
                    continue;
                }
                self.feature_weights_into(node_attrs, node, feature, &mut w1, &mut w2, &mut w3);
                out_dot[node * self.num_features + feature] = self.jvp_node_feature_with_weights(
                    x, x_dot, node, feature, &w1, &w2, &w3, u1_entries, u2_entries, u3_entries,
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn vjp_x_linearize(
        &self,
        x: &[f64],
        x_dot: &[f64],
        node_attrs: &[f64],
        grad_out: &[f64],
        grad_out_dot: &[f64],
        grad_x_dot: &mut [f64],
    ) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes(
            x,
            node_attrs,
            grad_out.len(),
            "mace_symmetric_contraction_vjp_x_linearize grad_out",
        )?;
        self.check_len(
            "mace_symmetric_contraction_vjp_x_linearize x_dot",
            x.len(),
            x_dot.len(),
        )?;
        self.check_len(
            "mace_symmetric_contraction_vjp_x_linearize grad_out_dot",
            grad_out.len(),
            grad_out_dot.len(),
        )?;
        self.check_len(
            "mace_symmetric_contraction_vjp_x_linearize grad_x_dot",
            x.len(),
            grad_x_dot.len(),
        )?;

        let mut w1 = vec![0.0; self.u1_params];
        let mut w2 = vec![0.0; self.u2_params];
        let mut w3 = vec![0.0; self.u3_params];
        let owned_entries;
        let entries = if let Some(entries) = self.entries {
            entries
        } else {
            owned_entries = self.nonzero_entries();
            &owned_entries
        };
        let u1_entries = &entries.u1;
        let u2_entries = &entries.u2;
        let u3_entries = &entries.u3;
        if let Some(active_elements) =
            active_unit_one_hot_elements(node_attrs, num_nodes, self.num_elements)
        {
            let mut nodes_by_element = vec![Vec::new(); self.num_elements];
            for (node, &element) in active_elements.iter().enumerate() {
                nodes_by_element[element].push(node);
            }
            let active_node_groups = nodes_by_element
                .into_iter()
                .enumerate()
                .filter(|(_, nodes)| !nodes.is_empty())
                .collect::<Vec<_>>();
            if self.num_ell == 16 {
                #[cfg(feature = "parallel")]
                {
                    let active_element_list = active_node_groups
                        .iter()
                        .map(|(element, _)| *element)
                        .collect::<Vec<_>>();
                    let mut element_to_dense = vec![usize::MAX; self.num_elements];
                    for (dense_element, &element) in active_element_list.iter().enumerate() {
                        element_to_dense[element] = dense_element;
                    }
                    let weighted_entries = self.weighted_entries_by_active_element_feature(
                        &active_element_list,
                        u1_entries,
                        u2_entries,
                        u3_entries,
                    );
                    if num_nodes >= 32
                        && self.num_features >= 64
                        && rayon::current_thread_index().is_none()
                    {
                        let node_x_len = self.num_features * 16;
                        grad_x_dot.par_chunks_mut(node_x_len).enumerate().for_each(
                            |(node, grad_x_dot_node)| {
                                let dense_element = element_to_dense[active_elements[node]];
                                debug_assert_ne!(dense_element, usize::MAX);
                                let x_node = &x[node * node_x_len..(node + 1) * node_x_len];
                                let x_dot_node = &x_dot[node * node_x_len..(node + 1) * node_x_len];
                                let grad_out_node = &grad_out
                                    [node * self.num_features..(node + 1) * self.num_features];
                                let grad_out_dot_node = &grad_out_dot
                                    [node * self.num_features..(node + 1) * self.num_features];
                                for feature in 0..self.num_features {
                                    let row_start = feature * 16;
                                    let x_row = &x_node[row_start..row_start + 16];
                                    let x_dot_row = &x_dot_node[row_start..row_start + 16];
                                    let bar_output = grad_out_node[feature];
                                    let bar_output_dot = grad_out_dot_node[feature];
                                    if !scalar_linearized_vjp_row_can_contribute(
                                        bar_output,
                                        bar_output_dot,
                                        x_dot_row,
                                    ) {
                                        continue;
                                    }
                                    let (weighted_u1, weighted_u2, weighted_u3) =
                                        weighted_entries.slices(dense_element, feature);
                                    Self::vjp_node_feature_linearize_weighted_entries_ell16_row(
                                        x_row,
                                        x_dot_row,
                                        bar_output,
                                        bar_output_dot,
                                        &mut grad_x_dot_node[row_start..row_start + 16],
                                        u1_entries,
                                        weighted_u1,
                                        u2_entries,
                                        weighted_u2,
                                        u3_entries,
                                        weighted_u3,
                                    );
                                }
                            },
                        );
                        return Ok(());
                    }
                    self.vjp_x_linearize_active_node_groups_with_weighted_entries_ell16(
                        x,
                        x_dot,
                        grad_out,
                        grad_out_dot,
                        grad_x_dot,
                        &active_node_groups,
                        &weighted_entries,
                        u1_entries,
                        u2_entries,
                        u3_entries,
                    );
                    return Ok(());
                }
                #[cfg(not(feature = "parallel"))]
                for feature in 0..self.num_features {
                    for (element, nodes) in &active_node_groups {
                        if !nodes.iter().any(|&node| {
                            let out_index = node * self.num_features + feature;
                            let x_base = self.x_index(node, feature, 0);
                            scalar_linearized_vjp_row_can_contribute(
                                grad_out[out_index],
                                grad_out_dot[out_index],
                                &x_dot[x_base..x_base + 16],
                            )
                        }) {
                            continue;
                        }
                        self.feature_weights_for_element_into(
                            *element, feature, &mut w1, &mut w2, &mut w3,
                        );
                        for &node in nodes {
                            self.vjp_node_feature_linearize_with_weights_ell16(
                                x,
                                x_dot,
                                grad_out,
                                grad_out_dot,
                                grad_x_dot,
                                node,
                                feature,
                                &w1,
                                &w2,
                                &w3,
                                u1_entries,
                                u2_entries,
                                u3_entries,
                            );
                        }
                    }
                }
                #[cfg(not(feature = "parallel"))]
                return Ok(());
            }
            for feature in 0..self.num_features {
                for (element, nodes) in &active_node_groups {
                    if !nodes.iter().any(|&node| {
                        let out_index = node * self.num_features + feature;
                        let x_base = self.x_index(node, feature, 0);
                        scalar_linearized_vjp_row_can_contribute(
                            grad_out[out_index],
                            grad_out_dot[out_index],
                            &x_dot[x_base..x_base + self.num_ell],
                        )
                    }) {
                        continue;
                    }
                    self.feature_weights_for_element_into(
                        *element, feature, &mut w1, &mut w2, &mut w3,
                    );
                    for &node in nodes {
                        let out_index = node * self.num_features + feature;
                        let x_base = self.x_index(node, feature, 0);
                        if !scalar_linearized_vjp_row_can_contribute(
                            grad_out[out_index],
                            grad_out_dot[out_index],
                            &x_dot[x_base..x_base + self.num_ell],
                        ) {
                            continue;
                        }
                        self.vjp_node_feature_linearize_with_weights(
                            x,
                            x_dot,
                            grad_out,
                            grad_out_dot,
                            grad_x_dot,
                            node,
                            feature,
                            &w1,
                            &w2,
                            &w3,
                            u1_entries,
                            u2_entries,
                            u3_entries,
                        );
                    }
                }
            }
            return Ok(());
        }
        for node in 0..num_nodes {
            for feature in 0..self.num_features {
                let out_index = node * self.num_features + feature;
                let x_base = self.x_index(node, feature, 0);
                if !scalar_linearized_vjp_row_can_contribute(
                    grad_out[out_index],
                    grad_out_dot[out_index],
                    &x_dot[x_base..x_base + self.num_ell],
                ) {
                    continue;
                }
                self.feature_weights_into(node_attrs, node, feature, &mut w1, &mut w2, &mut w3);
                if self.num_ell == 16 {
                    self.vjp_node_feature_linearize_with_weights_ell16(
                        x,
                        x_dot,
                        grad_out,
                        grad_out_dot,
                        grad_x_dot,
                        node,
                        feature,
                        &w1,
                        &w2,
                        &w3,
                        u1_entries,
                        u2_entries,
                        u3_entries,
                    );
                } else {
                    self.vjp_node_feature_linearize_with_weights(
                        x,
                        x_dot,
                        grad_out,
                        grad_out_dot,
                        grad_x_dot,
                        node,
                        feature,
                        &w1,
                        &w2,
                        &w3,
                        u1_entries,
                        u2_entries,
                        u3_entries,
                    );
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "parallel")]
    fn vjp_x_linearize_active_node_groups_with_weighted_entries_ell16(
        &self,
        x: &[f64],
        x_dot: &[f64],
        grad_out: &[f64],
        grad_out_dot: &[f64],
        grad_x_dot: &mut [f64],
        active_node_groups: &[(usize, Vec<usize>)],
        weighted_entries: &ScalarWeightedEntriesFlat,
        u1_entries: &[ScalarU1Entry],
        u2_entries: &[ScalarU2Entry],
        u3_entries: &[ScalarU3Entry],
    ) {
        for feature in 0..self.num_features {
            for (dense_element, (_, nodes)) in active_node_groups.iter().enumerate() {
                if !nodes.iter().any(|&node| {
                    let out_index = node * self.num_features + feature;
                    let x_base = self.x_index(node, feature, 0);
                    scalar_linearized_vjp_row_can_contribute(
                        grad_out[out_index],
                        grad_out_dot[out_index],
                        &x_dot[x_base..x_base + 16],
                    )
                }) {
                    continue;
                }
                let (weighted_u1, weighted_u2, weighted_u3) =
                    weighted_entries.slices(dense_element, feature);
                for &node in nodes {
                    self.vjp_node_feature_linearize_with_weighted_entries_ell16(
                        x,
                        x_dot,
                        grad_out,
                        grad_out_dot,
                        grad_x_dot,
                        node,
                        feature,
                        u1_entries,
                        weighted_u1,
                        u2_entries,
                        weighted_u2,
                        u3_entries,
                        weighted_u3,
                    );
                }
            }
        }
    }

    pub fn jacobian_x_dense(
        &self,
        x: &[f64],
        node_attrs: &[f64],
        jacobian_x: &mut [f64],
    ) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_x_and_node_attrs(
            x,
            node_attrs,
            "mace_symmetric_contraction_jacobian_x_dense x",
        )?;
        self.check_len(
            "mace_symmetric_contraction_jacobian_x_dense jacobian_x",
            x.len(),
            jacobian_x.len(),
        )?;
        jacobian_x.fill(0.0);
        let owned_entries;
        let entries = if let Some(entries) = self.entries {
            entries
        } else {
            owned_entries = self.nonzero_entries();
            &owned_entries
        };
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && num_nodes >= 16 {
            let input_row_len = self.num_features * self.num_ell;
            jacobian_x
                .par_chunks_mut(input_row_len)
                .enumerate()
                .for_each(|(node, jacobian_row)| {
                    let mut w1 = vec![0.0; self.u1_params];
                    let mut w2 = vec![0.0; self.u2_params];
                    let mut w3 = vec![0.0; self.u3_params];
                    for feature in 0..self.num_features {
                        self.feature_weights_into(
                            node_attrs, node, feature, &mut w1, &mut w2, &mut w3,
                        );
                        let x_base = self.x_index(node, feature, 0);
                        let x_row = &x[x_base..x_base + self.num_ell];
                        let jacobian_feature =
                            &mut jacobian_row[feature * self.num_ell..(feature + 1) * self.num_ell];
                        for entry in &entries.u1 {
                            accumulate_scalar_local_jacobian_term(
                                entry.value * w1[entry.param],
                                &[entry.ell],
                                x_row,
                                jacobian_feature,
                            );
                        }
                        for entry in &entries.u2 {
                            accumulate_scalar_local_jacobian_term(
                                entry.value * w2[entry.param],
                                &[entry.left, entry.right],
                                x_row,
                                jacobian_feature,
                            );
                        }
                        for entry in &entries.u3 {
                            accumulate_scalar_local_jacobian_term(
                                entry.value * w3[entry.param],
                                &[entry.left, entry.right, entry.inner],
                                x_row,
                                jacobian_feature,
                            );
                        }
                    }
                });
            return Ok(());
        }

        let mut w1 = vec![0.0; self.u1_params];
        let mut w2 = vec![0.0; self.u2_params];
        let mut w3 = vec![0.0; self.u3_params];
        for node in 0..num_nodes {
            for feature in 0..self.num_features {
                self.feature_weights_into(node_attrs, node, feature, &mut w1, &mut w2, &mut w3);
                let x_base = self.x_index(node, feature, 0);
                let x_row = &x[x_base..x_base + self.num_ell];
                let jacobian_feature = &mut jacobian_x[x_base..x_base + self.num_ell];
                for entry in &entries.u1 {
                    accumulate_scalar_local_jacobian_term(
                        entry.value * w1[entry.param],
                        &[entry.ell],
                        x_row,
                        jacobian_feature,
                    );
                }
                for entry in &entries.u2 {
                    accumulate_scalar_local_jacobian_term(
                        entry.value * w2[entry.param],
                        &[entry.left, entry.right],
                        x_row,
                        jacobian_feature,
                    );
                }
                for entry in &entries.u3 {
                    accumulate_scalar_local_jacobian_term(
                        entry.value * w3[entry.param],
                        &[entry.left, entry.right, entry.inner],
                        x_row,
                        jacobian_feature,
                    );
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn forward_node_feature(
        &self,
        x: &[f64],
        node_attrs: &[f64],
        node: usize,
        feature: usize,
        w1: &mut [f64],
        w2: &mut [f64],
        w3: &mut [f64],
        u1_entries: &[ScalarU1Entry],
        u2_entries: &[ScalarU2Entry],
        u3_entries: &[ScalarU3Entry],
    ) -> f64 {
        self.feature_weights_into(node_attrs, node, feature, w1, w2, w3);
        self.forward_node_feature_with_weights(
            x, node, feature, w1, w2, w3, u1_entries, u2_entries, u3_entries,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn forward_node_feature_with_weights(
        &self,
        x: &[f64],
        node: usize,
        feature: usize,
        w1: &[f64],
        w2: &[f64],
        w3: &[f64],
        u1_entries: &[ScalarU1Entry],
        u2_entries: &[ScalarU2Entry],
        u3_entries: &[ScalarU3Entry],
    ) -> f64 {
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];
        let mut output = 0.0;
        let mut comp = 0.0;
        for entry in u1_entries {
            kahan_add(
                &mut output,
                &mut comp,
                entry.value * w1[entry.param] * x_row[entry.ell],
            );
        }
        for entry in u2_entries {
            kahan_add(
                &mut output,
                &mut comp,
                entry.value * w2[entry.param] * x_row[entry.left] * x_row[entry.right],
            );
        }
        for entry in u3_entries {
            kahan_add(
                &mut output,
                &mut comp,
                entry.value
                    * w3[entry.param]
                    * x_row[entry.left]
                    * x_row[entry.right]
                    * x_row[entry.inner],
            );
        }
        output
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn forward_node_feature_with_weighted_entries(
        &self,
        x: &[f64],
        node: usize,
        feature: usize,
        u1_entries: &[ScalarU1Entry],
        weighted_u1: &[f64],
        u2_entries: &[ScalarU2Entry],
        weighted_u2: &[f64],
        u3_entries: &[ScalarU3Entry],
        weighted_u3: &[f64],
    ) -> f64 {
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];
        let mut output = 0.0;
        let mut comp = 0.0;
        for (entry, &weighted_value) in u1_entries.iter().zip(weighted_u1) {
            kahan_add(&mut output, &mut comp, weighted_value * x_row[entry.ell]);
        }
        for (entry, &weighted_value) in u2_entries.iter().zip(weighted_u2) {
            kahan_add(
                &mut output,
                &mut comp,
                weighted_value * x_row[entry.left] * x_row[entry.right],
            );
        }
        for (entry, &weighted_value) in u3_entries.iter().zip(weighted_u3) {
            kahan_add(
                &mut output,
                &mut comp,
                weighted_value * x_row[entry.left] * x_row[entry.right] * x_row[entry.inner],
            );
        }
        output
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn jvp_node_feature_with_weights(
        &self,
        x: &[f64],
        x_dot: &[f64],
        node: usize,
        feature: usize,
        w1: &[f64],
        w2: &[f64],
        w3: &[f64],
        u1_entries: &[ScalarU1Entry],
        u2_entries: &[ScalarU2Entry],
        u3_entries: &[ScalarU3Entry],
    ) -> f64 {
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];
        let x_dot_row = &x_dot[x_base..x_base + self.num_ell];
        let mut output = 0.0;
        for entry in u1_entries {
            output += entry.value * w1[entry.param] * x_dot_row[entry.ell];
        }
        for entry in u2_entries {
            output += entry.value
                * w2[entry.param]
                * (x_dot_row[entry.left] * x_row[entry.right]
                    + x_row[entry.left] * x_dot_row[entry.right]);
        }
        for entry in u3_entries {
            output += entry.value
                * w3[entry.param]
                * (x_dot_row[entry.left] * x_row[entry.right] * x_row[entry.inner]
                    + x_row[entry.left] * x_dot_row[entry.right] * x_row[entry.inner]
                    + x_row[entry.left] * x_row[entry.right] * x_dot_row[entry.inner]);
        }
        output
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn jvp_node_feature_with_weighted_entries(
        &self,
        x: &[f64],
        x_dot: &[f64],
        node: usize,
        feature: usize,
        u1_entries: &[ScalarU1Entry],
        weighted_u1: &[f64],
        u2_entries: &[ScalarU2Entry],
        weighted_u2: &[f64],
        u3_entries: &[ScalarU3Entry],
        weighted_u3: &[f64],
    ) -> f64 {
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];
        let x_dot_row = &x_dot[x_base..x_base + self.num_ell];
        let mut output = 0.0;
        for (entry, &weighted_value) in u1_entries.iter().zip(weighted_u1) {
            output += weighted_value * x_dot_row[entry.ell];
        }
        for (entry, &weighted_value) in u2_entries.iter().zip(weighted_u2) {
            output += weighted_value
                * (x_dot_row[entry.left] * x_row[entry.right]
                    + x_row[entry.left] * x_dot_row[entry.right]);
        }
        for (entry, &weighted_value) in u3_entries.iter().zip(weighted_u3) {
            output += weighted_value
                * (x_dot_row[entry.left] * x_row[entry.right] * x_row[entry.inner]
                    + x_row[entry.left] * x_dot_row[entry.right] * x_row[entry.inner]
                    + x_row[entry.left] * x_row[entry.right] * x_dot_row[entry.inner]);
        }
        output
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vjp_node_feature(
        &self,
        x: &[f64],
        node_attrs: &[f64],
        grad_out: &[f64],
        grad_x: &mut [f64],
        node: usize,
        feature: usize,
        w1: &mut [f64],
        w2: &mut [f64],
        w3: &mut [f64],
        grad_values: &mut [f64],
        u1_entries: &[ScalarU1Entry],
        u2_entries: &[ScalarU2Entry],
        u3_entries: &[ScalarU3Entry],
    ) {
        self.feature_weights_into(node_attrs, node, feature, w1, w2, w3);
        self.vjp_node_feature_with_weights(
            x,
            grad_out,
            grad_x,
            node,
            feature,
            w1,
            w2,
            w3,
            grad_values,
            u1_entries,
            u2_entries,
            u3_entries,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vjp_node_feature_linearize_with_weights(
        &self,
        x: &[f64],
        x_dot: &[f64],
        grad_out: &[f64],
        grad_out_dot: &[f64],
        grad_x_dot: &mut [f64],
        node: usize,
        feature: usize,
        w1: &[f64],
        w2: &[f64],
        w3: &[f64],
        u1_entries: &[ScalarU1Entry],
        u2_entries: &[ScalarU2Entry],
        u3_entries: &[ScalarU3Entry],
    ) {
        let out_index = node * self.num_features + feature;
        let bar_output = grad_out[out_index];
        let bar_output_dot = grad_out_dot[out_index];
        if bar_output == 0.0 && bar_output_dot == 0.0 {
            return;
        }
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];
        let x_dot_row = &x_dot[x_base..x_base + self.num_ell];
        let grad_x_dot_row = &mut grad_x_dot[x_base..x_base + self.num_ell];

        for entry in u1_entries {
            grad_x_dot_row[entry.ell] += bar_output_dot * entry.value * w1[entry.param];
        }
        for entry in u2_entries {
            accumulate_monomial_vjp_linearize(
                &[entry.left, entry.right],
                entry.value * w2[entry.param],
                bar_output,
                bar_output_dot,
                x_row,
                x_dot_row,
                grad_x_dot_row,
            );
        }
        for entry in u3_entries {
            accumulate_monomial_vjp_linearize(
                &[entry.left, entry.right, entry.inner],
                entry.value * w3[entry.param],
                bar_output,
                bar_output_dot,
                x_row,
                x_dot_row,
                grad_x_dot_row,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vjp_node_feature_linearize_with_weights_ell16(
        &self,
        x: &[f64],
        x_dot: &[f64],
        grad_out: &[f64],
        grad_out_dot: &[f64],
        grad_x_dot: &mut [f64],
        node: usize,
        feature: usize,
        w1: &[f64],
        w2: &[f64],
        w3: &[f64],
        u1_entries: &[ScalarU1Entry],
        u2_entries: &[ScalarU2Entry],
        u3_entries: &[ScalarU3Entry],
    ) {
        let out_index = node * self.num_features + feature;
        let bar_output = grad_out[out_index];
        let bar_output_dot = grad_out_dot[out_index];
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + 16];
        let x_dot_row = &x_dot[x_base..x_base + 16];
        if !scalar_linearized_vjp_row_can_contribute(bar_output, bar_output_dot, x_dot_row) {
            return;
        }
        let mut grad_values = [0.0_f64; 16];

        if bar_output_dot != 0.0 {
            for entry in u1_entries {
                grad_values[entry.ell] += bar_output_dot * entry.value * w1[entry.param];
            }
        }
        for entry in u2_entries {
            let coeff = entry.value * w2[entry.param];
            if coeff == 0.0 {
                continue;
            }
            let left_term =
                bar_output_dot * x_row[entry.right] + bar_output * x_dot_row[entry.right];
            let right_term =
                bar_output_dot * x_row[entry.left] + bar_output * x_dot_row[entry.left];
            grad_values[entry.left] += coeff * left_term;
            grad_values[entry.right] += coeff * right_term;
        }
        for entry in u3_entries {
            let coeff = entry.value * w3[entry.param];
            if coeff == 0.0 {
                continue;
            }
            let indices = [entry.left, entry.right, entry.inner];
            for position in 0..3 {
                let mut except_value = 1.0;
                for (other_position, &index) in indices.iter().enumerate() {
                    if other_position != position {
                        except_value *= x_row[index];
                    }
                }
                let mut except_dot = 0.0;
                for tangent_position in 0..3 {
                    if tangent_position == position {
                        continue;
                    }
                    let mut term = x_dot_row[indices[tangent_position]];
                    for (other_position, &index) in indices.iter().enumerate() {
                        if other_position != position && other_position != tangent_position {
                            term *= x_row[index];
                        }
                    }
                    except_dot += term;
                }
                grad_values[indices[position]] +=
                    coeff * (bar_output_dot * except_value + bar_output * except_dot);
            }
        }
        add_ell16_grad_to_x(grad_x_dot, x_base, &grad_values);
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn vjp_node_feature_linearize_with_weighted_entries_ell16(
        &self,
        x: &[f64],
        x_dot: &[f64],
        grad_out: &[f64],
        grad_out_dot: &[f64],
        grad_x_dot: &mut [f64],
        node: usize,
        feature: usize,
        u1_entries: &[ScalarU1Entry],
        weighted_u1: &[f64],
        u2_entries: &[ScalarU2Entry],
        weighted_u2: &[f64],
        u3_entries: &[ScalarU3Entry],
        weighted_u3: &[f64],
    ) {
        let out_index = node * self.num_features + feature;
        let bar_output = grad_out[out_index];
        let bar_output_dot = grad_out_dot[out_index];
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + 16];
        let x_dot_row = &x_dot[x_base..x_base + 16];
        let grad_x_dot_row = &mut grad_x_dot[x_base..x_base + 16];
        Self::vjp_node_feature_linearize_weighted_entries_ell16_row(
            x_row,
            x_dot_row,
            bar_output,
            bar_output_dot,
            grad_x_dot_row,
            u1_entries,
            weighted_u1,
            u2_entries,
            weighted_u2,
            u3_entries,
            weighted_u3,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn vjp_node_feature_linearize_weighted_entries_ell16_row(
        x_row: &[f64],
        x_dot_row: &[f64],
        bar_output: f64,
        bar_output_dot: f64,
        grad_x_dot_row: &mut [f64],
        u1_entries: &[ScalarU1Entry],
        weighted_u1: &[f64],
        u2_entries: &[ScalarU2Entry],
        weighted_u2: &[f64],
        u3_entries: &[ScalarU3Entry],
        weighted_u3: &[f64],
    ) {
        debug_assert_eq!(x_row.len(), 16);
        debug_assert_eq!(x_dot_row.len(), 16);
        debug_assert_eq!(grad_x_dot_row.len(), 16);
        if !scalar_linearized_vjp_row_can_contribute(bar_output, bar_output_dot, x_dot_row) {
            return;
        }
        let mut grad_values = [0.0_f64; 16];

        if all_values_zero(x_dot_row) {
            debug_assert_ne!(bar_output_dot, 0.0);
            for (entry, &weighted_value) in u1_entries.iter().zip(weighted_u1) {
                grad_values[entry.ell] += bar_output_dot * weighted_value;
            }
            for (entry, &weighted_value) in u2_entries.iter().zip(weighted_u2) {
                if weighted_value == 0.0 {
                    continue;
                }
                let derivative = bar_output_dot * weighted_value;
                if entry.left == entry.right {
                    grad_values[entry.left] += 2.0 * derivative * x_row[entry.left];
                } else {
                    grad_values[entry.left] += derivative * x_row[entry.right];
                    grad_values[entry.right] += derivative * x_row[entry.left];
                }
            }
            for (entry, &weighted_value) in u3_entries.iter().zip(weighted_u3) {
                if weighted_value == 0.0 {
                    continue;
                }
                let derivative = bar_output_dot * weighted_value;
                let x_left = x_row[entry.left];
                let x_right = x_row[entry.right];
                let x_inner = x_row[entry.inner];
                if entry.left == entry.right {
                    if entry.right == entry.inner {
                        grad_values[entry.left] += 3.0 * derivative * x_left * x_left;
                    } else {
                        grad_values[entry.left] += 2.0 * derivative * x_left * x_inner;
                        grad_values[entry.inner] += derivative * x_left * x_left;
                    }
                } else if entry.right == entry.inner {
                    grad_values[entry.left] += derivative * x_right * x_right;
                    grad_values[entry.right] += 2.0 * derivative * x_left * x_right;
                } else {
                    grad_values[entry.left] += derivative * x_right * x_inner;
                    grad_values[entry.right] += derivative * x_left * x_inner;
                    grad_values[entry.inner] += derivative * x_left * x_right;
                }
            }
            add_ell16_grad_to_row(grad_x_dot_row, &grad_values);
            return;
        }

        if bar_output_dot != 0.0 {
            for (entry, &weighted_value) in u1_entries.iter().zip(weighted_u1) {
                grad_values[entry.ell] += bar_output_dot * weighted_value;
            }
        }
        for (entry, &weighted_value) in u2_entries.iter().zip(weighted_u2) {
            if weighted_value == 0.0 {
                continue;
            }
            let left_term =
                bar_output_dot * x_row[entry.right] + bar_output * x_dot_row[entry.right];
            let right_term =
                bar_output_dot * x_row[entry.left] + bar_output * x_dot_row[entry.left];
            grad_values[entry.left] += weighted_value * left_term;
            grad_values[entry.right] += weighted_value * right_term;
        }
        for (entry, &weighted_value) in u3_entries.iter().zip(weighted_u3) {
            if weighted_value == 0.0 {
                continue;
            }
            let left = entry.left;
            let right = entry.right;
            let inner = entry.inner;
            let x_left = x_row[left];
            let x_right = x_row[right];
            let x_inner = x_row[inner];
            let dx_left = x_dot_row[left];
            let dx_right = x_dot_row[right];
            let dx_inner = x_dot_row[inner];
            grad_values[left] += weighted_value
                * (bar_output_dot * x_right * x_inner
                    + bar_output * (dx_right * x_inner + x_right * dx_inner));
            grad_values[right] += weighted_value
                * (bar_output_dot * x_left * x_inner
                    + bar_output * (dx_left * x_inner + x_left * dx_inner));
            grad_values[inner] += weighted_value
                * (bar_output_dot * x_left * x_right
                    + bar_output * (dx_left * x_right + x_left * dx_right));
        }
        add_ell16_grad_to_row(grad_x_dot_row, &grad_values);
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vjp_node_feature_with_weights(
        &self,
        x: &[f64],
        grad_out: &[f64],
        grad_x: &mut [f64],
        node: usize,
        feature: usize,
        w1: &[f64],
        w2: &[f64],
        w3: &[f64],
        grad_values: &mut [f64],
        u1_entries: &[ScalarU1Entry],
        u2_entries: &[ScalarU2Entry],
        u3_entries: &[ScalarU3Entry],
    ) {
        let bar_output = grad_out[node * self.num_features + feature];
        if bar_output == 0.0 {
            return;
        }
        debug_assert_eq!(grad_values.len(), self.num_ell);
        grad_values.fill(0.0);
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];

        for entry in u1_entries {
            grad_values[entry.ell] += bar_output * entry.value * w1[entry.param];
        }
        for entry in u2_entries {
            let derivative = bar_output * entry.value * w2[entry.param];
            grad_values[entry.left] += derivative * x_row[entry.right];
            grad_values[entry.right] += derivative * x_row[entry.left];
        }
        for entry in u3_entries {
            let derivative = bar_output * entry.value * w3[entry.param];
            let x_left = x_row[entry.left];
            let x_right = x_row[entry.right];
            let x_inner = x_row[entry.inner];
            if entry.left == entry.right {
                if entry.right == entry.inner {
                    grad_values[entry.left] += 3.0 * derivative * x_left * x_left;
                } else {
                    grad_values[entry.left] += 2.0 * derivative * x_left * x_inner;
                    grad_values[entry.inner] += derivative * x_left * x_left;
                }
            } else if entry.right == entry.inner {
                grad_values[entry.left] += derivative * x_right * x_right;
                grad_values[entry.right] += 2.0 * derivative * x_left * x_right;
            } else {
                grad_values[entry.left] += derivative * x_right * x_inner;
                grad_values[entry.right] += derivative * x_left * x_inner;
                grad_values[entry.inner] += derivative * x_left * x_right;
            }
        }
        for (target, &value) in grad_x[x_base..x_base + self.num_ell]
            .iter_mut()
            .zip(grad_values.iter())
        {
            *target += value;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vjp_node_feature_grad_with_weights_ell16(
        &self,
        x: &[f64],
        grad_out: &[f64],
        node: usize,
        feature: usize,
        w1: &[f64],
        w2: &[f64],
        w3: &[f64],
        u1_entries: &[ScalarU1Entry],
        u2_entries: &[ScalarU2Entry],
        u3_entries: &[ScalarU3Entry],
    ) -> Option<[f64; 16]> {
        let bar_output = grad_out[node * self.num_features + feature];
        if bar_output == 0.0 {
            return None;
        }
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + 16];
        let mut grad_values = [0.0; 16];
        let mut comp = [0.0; 16];

        for entry in u1_entries {
            kahan_add(
                &mut grad_values[entry.ell],
                &mut comp[entry.ell],
                bar_output * entry.value * w1[entry.param],
            );
        }
        for entry in u2_entries {
            let derivative = bar_output * entry.value * w2[entry.param];
            kahan_add(
                &mut grad_values[entry.left],
                &mut comp[entry.left],
                derivative * x_row[entry.right],
            );
            kahan_add(
                &mut grad_values[entry.right],
                &mut comp[entry.right],
                derivative * x_row[entry.left],
            );
        }
        for entry in u3_entries {
            let derivative = bar_output * entry.value * w3[entry.param];
            let x_left = x_row[entry.left];
            let x_right = x_row[entry.right];
            let x_inner = x_row[entry.inner];
            if entry.left == entry.right {
                if entry.right == entry.inner {
                    kahan_add(
                        &mut grad_values[entry.left],
                        &mut comp[entry.left],
                        3.0 * derivative * x_left * x_left,
                    );
                } else {
                    kahan_add(
                        &mut grad_values[entry.left],
                        &mut comp[entry.left],
                        2.0 * derivative * x_left * x_inner,
                    );
                    kahan_add(
                        &mut grad_values[entry.inner],
                        &mut comp[entry.inner],
                        derivative * x_left * x_left,
                    );
                }
            } else if entry.right == entry.inner {
                kahan_add(
                    &mut grad_values[entry.left],
                    &mut comp[entry.left],
                    derivative * x_right * x_right,
                );
                kahan_add(
                    &mut grad_values[entry.right],
                    &mut comp[entry.right],
                    2.0 * derivative * x_left * x_right,
                );
            } else {
                kahan_add(
                    &mut grad_values[entry.left],
                    &mut comp[entry.left],
                    derivative * x_right * x_inner,
                );
                kahan_add(
                    &mut grad_values[entry.right],
                    &mut comp[entry.right],
                    derivative * x_left * x_inner,
                );
                kahan_add(
                    &mut grad_values[entry.inner],
                    &mut comp[entry.inner],
                    derivative * x_left * x_right,
                );
            }
        }
        Some(grad_values)
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn vjp_node_feature_grad_with_weighted_ell16(
        &self,
        x: &[f64],
        grad_out: &[f64],
        node: usize,
        feature: usize,
        u1_entries: &[ScalarU1Entry],
        weighted_u1: &[f64],
        u2_entries: &[ScalarU2Entry],
        weighted_u2: &[f64],
        u3_entries: &[ScalarU3Entry],
        weighted_u3: &[f64],
    ) -> Option<[f64; 16]> {
        let bar_output = grad_out[node * self.num_features + feature];
        if bar_output == 0.0 {
            return None;
        }
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + 16];
        let mut grad_values = [0.0; 16];
        let mut comp = [0.0; 16];

        for (entry, &weighted_value) in u1_entries.iter().zip(weighted_u1) {
            kahan_add(
                &mut grad_values[entry.ell],
                &mut comp[entry.ell],
                bar_output * weighted_value,
            );
        }
        for (entry, &weighted_value) in u2_entries.iter().zip(weighted_u2) {
            let derivative = bar_output * weighted_value;
            kahan_add(
                &mut grad_values[entry.left],
                &mut comp[entry.left],
                derivative * x_row[entry.right],
            );
            kahan_add(
                &mut grad_values[entry.right],
                &mut comp[entry.right],
                derivative * x_row[entry.left],
            );
        }
        for (entry, &weighted_value) in u3_entries.iter().zip(weighted_u3) {
            let derivative = bar_output * weighted_value;
            let x_left = x_row[entry.left];
            let x_right = x_row[entry.right];
            let x_inner = x_row[entry.inner];
            if entry.left == entry.right {
                if entry.right == entry.inner {
                    kahan_add(
                        &mut grad_values[entry.left],
                        &mut comp[entry.left],
                        3.0 * derivative * x_left * x_left,
                    );
                } else {
                    kahan_add(
                        &mut grad_values[entry.left],
                        &mut comp[entry.left],
                        2.0 * derivative * x_left * x_inner,
                    );
                    kahan_add(
                        &mut grad_values[entry.inner],
                        &mut comp[entry.inner],
                        derivative * x_left * x_left,
                    );
                }
            } else if entry.right == entry.inner {
                kahan_add(
                    &mut grad_values[entry.left],
                    &mut comp[entry.left],
                    derivative * x_right * x_right,
                );
                kahan_add(
                    &mut grad_values[entry.right],
                    &mut comp[entry.right],
                    2.0 * derivative * x_left * x_right,
                );
            } else {
                kahan_add(
                    &mut grad_values[entry.left],
                    &mut comp[entry.left],
                    derivative * x_right * x_inner,
                );
                kahan_add(
                    &mut grad_values[entry.right],
                    &mut comp[entry.right],
                    derivative * x_left * x_inner,
                );
                kahan_add(
                    &mut grad_values[entry.inner],
                    &mut comp[entry.inner],
                    derivative * x_left * x_right,
                );
            }
        }
        Some(grad_values)
    }

    #[cfg(feature = "parallel")]
    fn weighted_entries_by_active_element_feature(
        &self,
        active_elements: &[usize],
        u1_entries: &[ScalarU1Entry],
        u2_entries: &[ScalarU2Entry],
        u3_entries: &[ScalarU3Entry],
    ) -> ScalarWeightedEntriesFlat {
        let rows = active_elements.len() * self.num_features;
        let mut weighted = ScalarWeightedEntriesFlat {
            u1: vec![0.0; rows * u1_entries.len()],
            u2: vec![0.0; rows * u2_entries.len()],
            u3: vec![0.0; rows * u3_entries.len()],
            u1_len: u1_entries.len(),
            u2_len: u2_entries.len(),
            u3_len: u3_entries.len(),
            num_features: self.num_features,
        };
        let mut w1 = vec![0.0; self.u1_params];
        let mut w2 = vec![0.0; self.u2_params];
        let mut w3 = vec![0.0; self.u3_params];
        for (dense_element, &element) in active_elements.iter().enumerate() {
            for feature in 0..self.num_features {
                self.feature_weights_for_element_into(element, feature, &mut w1, &mut w2, &mut w3);
                let row = dense_element * self.num_features + feature;
                let u1_base = row * u1_entries.len();
                let u2_base = row * u2_entries.len();
                let u3_base = row * u3_entries.len();
                weight_scalar_entries(
                    &w1,
                    &w2,
                    &w3,
                    u1_entries,
                    u2_entries,
                    u3_entries,
                    &mut weighted.u1[u1_base..u1_base + u1_entries.len()],
                    &mut weighted.u2[u2_base..u2_base + u2_entries.len()],
                    &mut weighted.u3[u3_base..u3_base + u3_entries.len()],
                );
            }
        }
        weighted
    }

    fn nonzero_entries(&self) -> MaceScalarSymmetricContractionEntries {
        MaceScalarSymmetricContractionEntries {
            u1: self.u1_nonzero_entries(),
            u2: self.u2_nonzero_entries(),
            u3: self.u3_nonzero_entries(),
        }
    }

    fn u1_nonzero_entries(&self) -> Vec<ScalarU1Entry> {
        let mut entries = Vec::new();
        for ell in 0..self.num_ell {
            for param in 0..self.u1_params {
                let value = self.u1[ell * self.u1_params + param];
                if value != 0.0 {
                    entries.push(ScalarU1Entry { ell, param, value });
                }
            }
        }
        entries
    }

    fn u2_nonzero_entries(&self) -> Vec<ScalarU2Entry> {
        let mut entries = Vec::new();
        for left in 0..self.num_ell {
            for right in 0..self.num_ell {
                for param in 0..self.u2_params {
                    let value = self.u2[((left * self.num_ell + right) * self.u2_params) + param];
                    if value != 0.0 {
                        entries.push(ScalarU2Entry {
                            left,
                            right,
                            param,
                            value,
                        });
                    }
                }
            }
        }
        entries
    }

    fn u3_nonzero_entries(&self) -> Vec<ScalarU3Entry> {
        let mut entries = Vec::new();
        for left in 0..self.num_ell {
            for right in 0..self.num_ell {
                for inner in 0..self.num_ell {
                    for param in 0..self.u3_params {
                        let value = self.u3[(((left * self.num_ell + right) * self.num_ell
                            + inner)
                            * self.u3_params)
                            + param];
                        if value != 0.0 {
                            let (left, right, inner) = sorted_triple(left, right, inner);
                            entries.push(ScalarU3Entry {
                                left,
                                right,
                                inner,
                                param,
                                value,
                            });
                        }
                    }
                }
            }
        }
        combine_scalar_u3_entries(entries)
    }

    #[inline(always)]
    fn feature_weights_for_element_into(
        &self,
        element: usize,
        feature: usize,
        w1: &mut [f64],
        w2: &mut [f64],
        w3: &mut [f64],
    ) {
        debug_assert_eq!(w1.len(), self.u1_params);
        debug_assert_eq!(w2.len(), self.u2_params);
        debug_assert_eq!(w3.len(), self.u3_params);
        for (param, value) in w1.iter_mut().enumerate() {
            *value = self.weights_u1[self.weight_index(element, param, feature, self.u1_params)];
        }
        for (param, value) in w2.iter_mut().enumerate() {
            *value = self.weights_u2[self.weight_index(element, param, feature, self.u2_params)];
        }
        for (param, value) in w3.iter_mut().enumerate() {
            *value = self.weights_u3[self.weight_index(element, param, feature, self.u3_params)];
        }
    }

    fn feature_weights_for_element_into_f32(
        &self,
        element: usize,
        feature: usize,
        w1: &mut [f32],
        w2: &mut [f32],
        w3: &mut [f32],
    ) {
        debug_assert_eq!(w1.len(), self.u1_params);
        debug_assert_eq!(w2.len(), self.u2_params);
        debug_assert_eq!(w3.len(), self.u3_params);
        for (param, value) in w1.iter_mut().enumerate() {
            *value =
                self.weights_u1[self.weight_index(element, param, feature, self.u1_params)] as f32;
        }
        for (param, value) in w2.iter_mut().enumerate() {
            *value =
                self.weights_u2[self.weight_index(element, param, feature, self.u2_params)] as f32;
        }
        for (param, value) in w3.iter_mut().enumerate() {
            *value =
                self.weights_u3[self.weight_index(element, param, feature, self.u3_params)] as f32;
        }
    }

    fn feature_weights_into(
        &self,
        node_attrs: &[f64],
        node: usize,
        feature: usize,
        w1: &mut [f64],
        w2: &mut [f64],
        w3: &mut [f64],
    ) {
        debug_assert_eq!(w1.len(), self.u1_params);
        debug_assert_eq!(w2.len(), self.u2_params);
        debug_assert_eq!(w3.len(), self.u3_params);
        w1.fill(0.0);
        w2.fill(0.0);
        w3.fill(0.0);
        let attr_row = &node_attrs[node * self.num_elements..(node + 1) * self.num_elements];
        if let Some((element, coeff)) = active_one_hot_attr(attr_row) {
            for (param, value) in w1.iter_mut().enumerate() {
                *value = coeff
                    * self.weights_u1[self.weight_index(element, param, feature, self.u1_params)];
            }
            for (param, value) in w2.iter_mut().enumerate() {
                *value = coeff
                    * self.weights_u2[self.weight_index(element, param, feature, self.u2_params)];
            }
            for (param, value) in w3.iter_mut().enumerate() {
                *value = coeff
                    * self.weights_u3[self.weight_index(element, param, feature, self.u3_params)];
            }
            return;
        }
        for (element, &coeff) in attr_row.iter().enumerate() {
            if coeff == 0.0 {
                continue;
            }
            for (param, value) in w1.iter_mut().enumerate() {
                *value += coeff
                    * self.weights_u1[self.weight_index(element, param, feature, self.u1_params)];
            }
            for (param, value) in w2.iter_mut().enumerate() {
                *value += coeff
                    * self.weights_u2[self.weight_index(element, param, feature, self.u2_params)];
            }
            for (param, value) in w3.iter_mut().enumerate() {
                *value += coeff
                    * self.weights_u3[self.weight_index(element, param, feature, self.u3_params)];
            }
        }
    }

    fn feature_weights_into_f32(
        &self,
        node_attrs: &[f32],
        node: usize,
        feature: usize,
        w1: &mut [f32],
        w2: &mut [f32],
        w3: &mut [f32],
    ) {
        debug_assert_eq!(w1.len(), self.u1_params);
        debug_assert_eq!(w2.len(), self.u2_params);
        debug_assert_eq!(w3.len(), self.u3_params);
        w1.fill(0.0);
        w2.fill(0.0);
        w3.fill(0.0);
        let attr_row = &node_attrs[node * self.num_elements..(node + 1) * self.num_elements];
        if let Some((element, coeff)) = active_one_hot_attr_f32(attr_row) {
            for (param, value) in w1.iter_mut().enumerate() {
                *value = coeff
                    * self.weights_u1[self.weight_index(element, param, feature, self.u1_params)]
                        as f32;
            }
            for (param, value) in w2.iter_mut().enumerate() {
                *value = coeff
                    * self.weights_u2[self.weight_index(element, param, feature, self.u2_params)]
                        as f32;
            }
            for (param, value) in w3.iter_mut().enumerate() {
                *value = coeff
                    * self.weights_u3[self.weight_index(element, param, feature, self.u3_params)]
                        as f32;
            }
            return;
        }
        for (element, &coeff) in attr_row.iter().enumerate() {
            if coeff == 0.0 {
                continue;
            }
            for (param, value) in w1.iter_mut().enumerate() {
                *value += coeff
                    * self.weights_u1[self.weight_index(element, param, feature, self.u1_params)]
                        as f32;
            }
            for (param, value) in w2.iter_mut().enumerate() {
                *value += coeff
                    * self.weights_u2[self.weight_index(element, param, feature, self.u2_params)]
                        as f32;
            }
            for (param, value) in w3.iter_mut().enumerate() {
                *value += coeff
                    * self.weights_u3[self.weight_index(element, param, feature, self.u3_params)]
                        as f32;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn forward_node_feature_with_weights_f32(
        &self,
        x: &[f32],
        node: usize,
        feature: usize,
        w1: &[f32],
        w2: &[f32],
        w3: &[f32],
        u1_entries: &[ScalarU1Entry],
        u2_entries: &[ScalarU2Entry],
        u3_entries: &[ScalarU3Entry],
    ) -> f32 {
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];
        let mut output = 0.0_f32;
        for entry in u1_entries {
            output += entry.value as f32 * w1[entry.param] * x_row[entry.ell];
        }
        for entry in u2_entries {
            output += entry.value as f32 * w2[entry.param] * x_row[entry.left] * x_row[entry.right];
        }
        for entry in u3_entries {
            output += entry.value as f32
                * w3[entry.param]
                * x_row[entry.left]
                * x_row[entry.right]
                * x_row[entry.inner];
        }
        output
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vjp_node_feature_with_weights_f32(
        &self,
        x: &[f32],
        grad_out: &[f32],
        grad_x: &mut [f32],
        node: usize,
        feature: usize,
        w1: &[f32],
        w2: &[f32],
        w3: &[f32],
        u1_entries: &[ScalarU1Entry],
        u2_entries: &[ScalarU2Entry],
        u3_entries: &[ScalarU3Entry],
    ) {
        let bar_output = grad_out[node * self.num_features + feature];
        if bar_output == 0.0 {
            return;
        }
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];
        let grad_row = &mut grad_x[x_base..x_base + self.num_ell];
        for entry in u1_entries {
            grad_row[entry.ell] += bar_output * entry.value as f32 * w1[entry.param];
        }
        for entry in u2_entries {
            let derivative = bar_output * entry.value as f32 * w2[entry.param];
            grad_row[entry.left] += derivative * x_row[entry.right];
            grad_row[entry.right] += derivative * x_row[entry.left];
        }
        for entry in u3_entries {
            let derivative = bar_output * entry.value as f32 * w3[entry.param];
            let x_left = x_row[entry.left];
            let x_right = x_row[entry.right];
            let x_inner = x_row[entry.inner];
            if entry.left == entry.right {
                if entry.right == entry.inner {
                    grad_row[entry.left] += 3.0 * derivative * x_left * x_left;
                } else {
                    grad_row[entry.left] += 2.0 * derivative * x_left * x_inner;
                    grad_row[entry.inner] += derivative * x_left * x_left;
                }
            } else if entry.right == entry.inner {
                grad_row[entry.left] += derivative * x_right * x_right;
                grad_row[entry.right] += 2.0 * derivative * x_left * x_right;
            } else {
                grad_row[entry.left] += derivative * x_right * x_inner;
                grad_row[entry.right] += derivative * x_left * x_inner;
                grad_row[entry.inner] += derivative * x_left * x_right;
            }
        }
    }

    fn check_runtime_shapes(
        &self,
        x: &[f64],
        node_attrs: &[f64],
        out_len: usize,
        out_op: &'static str,
    ) -> Result<usize> {
        let per_node_x = self.num_features * self.num_ell;
        if !x.len().is_multiple_of(per_node_x) {
            return Err(CoreError::ShapeMismatch {
                op: "mace_symmetric_contraction x",
                expected: per_node_x,
                actual: x.len(),
            });
        }
        let num_nodes = x.len() / per_node_x;
        self.check_len(
            "mace_symmetric_contraction node_attrs",
            num_nodes * self.num_elements,
            node_attrs.len(),
        )?;
        self.check_len(out_op, num_nodes * self.num_features, out_len)?;
        Ok(num_nodes)
    }

    fn check_x_and_node_attrs(
        &self,
        x: &[f64],
        node_attrs: &[f64],
        x_op: &'static str,
    ) -> Result<usize> {
        let per_node_x = self.num_features * self.num_ell;
        if !x.len().is_multiple_of(per_node_x) {
            return Err(CoreError::ShapeMismatch {
                op: x_op,
                expected: per_node_x,
                actual: x.len(),
            });
        }
        let num_nodes = x.len() / per_node_x;
        self.check_len(
            "mace_symmetric_contraction node_attrs",
            num_nodes * self.num_elements,
            node_attrs.len(),
        )?;
        Ok(num_nodes)
    }

    fn check_runtime_shapes_f32(
        &self,
        x: &[f32],
        node_attrs: &[f32],
        out_len: usize,
        out_op: &'static str,
    ) -> Result<usize> {
        let per_node_x = self.num_features * self.num_ell;
        if !x.len().is_multiple_of(per_node_x) {
            return Err(CoreError::ShapeMismatch {
                op: "mace_symmetric_contraction_f32 x",
                expected: per_node_x,
                actual: x.len(),
            });
        }
        let num_nodes = x.len() / per_node_x;
        self.check_len(
            "mace_symmetric_contraction_f32 node_attrs",
            num_nodes * self.num_elements,
            node_attrs.len(),
        )?;
        self.check_len(out_op, num_nodes * self.num_features, out_len)?;
        Ok(num_nodes)
    }

    fn check_len(&self, op: &'static str, expected: usize, actual: usize) -> Result<()> {
        if expected == actual {
            Ok(())
        } else {
            Err(CoreError::ShapeMismatch {
                op,
                expected,
                actual,
            })
        }
    }

    fn x_index(&self, node: usize, feature: usize, ell: usize) -> usize {
        (node * self.num_features + feature) * self.num_ell + ell
    }

    fn weight_index(&self, element: usize, param: usize, feature: usize, params: usize) -> usize {
        (element * params + param) * self.num_features + feature
    }
}

impl<'a> MaceScalarVectorSymmetricContraction<'a> {
    pub fn validate(&self) -> Result<()> {
        self.scalar.validate()?;
        self.vector.validate()?;
        if self.scalar.num_elements != self.vector.num_elements
            || self.scalar.num_features != self.vector.num_features
            || self.scalar.num_ell != self.vector.num_ell
        {
            return Err(CoreError::InvalidTensorProduct(
                "mace scalar+vector symmetric contraction metadata mismatch".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn prepare_single_element_product_plan(
        &self,
    ) -> Result<MaceScalarVectorSingleElementProductPlan> {
        self.validate()?;
        Ok(MaceScalarVectorSingleElementProductPlan {
            scalar: self.scalar.prepare_single_element_product_plan()?,
            vector: self.vector.prepare_single_element_product_plan()?,
        })
    }

    pub fn forward(&self, x: &[f64], node_attrs: &[f64], out: &mut [f64]) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes(
            x,
            node_attrs,
            out.len(),
            "mace_scalar_vector_symmetric_contraction_forward out",
        )?;
        let scalar_len = num_nodes * self.scalar.num_features;
        let vector_len = num_nodes * self.vector.num_features * self.vector.num_equivariance;
        let mut scalar_out = vec![0.0; scalar_len];
        let mut vector_out = vec![0.0; vector_len];
        self.scalar.forward(x, node_attrs, &mut scalar_out)?;
        self.vector.forward(x, node_attrs, &mut vector_out)?;
        let per_node_scalar = self.scalar.num_features;
        let per_node_vector = self.vector.num_features * self.vector.num_equivariance;
        let per_node_out = per_node_scalar + per_node_vector;
        for node in 0..num_nodes {
            let scalar_src = node * per_node_scalar;
            let vector_src = node * per_node_vector;
            let out_dst = node * per_node_out;
            out[out_dst..out_dst + per_node_scalar]
                .copy_from_slice(&scalar_out[scalar_src..scalar_src + per_node_scalar]);
            out[out_dst + per_node_scalar..out_dst + per_node_out]
                .copy_from_slice(&vector_out[vector_src..vector_src + per_node_vector]);
        }
        Ok(())
    }

    pub fn forward_f32(&self, x: &[f32], node_attrs: &[f32], out: &mut [f32]) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes_f32(
            x,
            node_attrs,
            out.len(),
            "mace_scalar_vector_symmetric_contraction_forward_f32 out",
        )?;
        let scalar_len = num_nodes * self.scalar.num_features;
        let vector_len = num_nodes * self.vector.num_features * self.vector.num_equivariance;
        let mut scalar_out = vec![0.0_f32; scalar_len];
        let mut vector_out = vec![0.0_f32; vector_len];
        self.scalar.forward_f32(x, node_attrs, &mut scalar_out)?;
        self.vector.forward_f32(x, node_attrs, &mut vector_out)?;
        let per_node_scalar = self.scalar.num_features;
        let per_node_vector = self.vector.num_features * self.vector.num_equivariance;
        let per_node_out = per_node_scalar + per_node_vector;
        for node in 0..num_nodes {
            let scalar_src = node * per_node_scalar;
            let vector_src = node * per_node_vector;
            let out_dst = node * per_node_out;
            out[out_dst..out_dst + per_node_scalar]
                .copy_from_slice(&scalar_out[scalar_src..scalar_src + per_node_scalar]);
            out[out_dst + per_node_scalar..out_dst + per_node_out]
                .copy_from_slice(&vector_out[vector_src..vector_src + per_node_vector]);
        }
        Ok(())
    }

    pub fn vjp_x_f32(
        &self,
        x: &[f32],
        node_attrs: &[f32],
        grad_out: &[f32],
        grad_x: &mut [f32],
    ) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes_f32(
            x,
            node_attrs,
            grad_out.len(),
            "mace_scalar_vector_symmetric_contraction_vjp_x_f32 grad_out",
        )?;
        self.scalar.check_len(
            "mace_scalar_vector_symmetric_contraction_vjp_x_f32 grad_x",
            x.len(),
            grad_x.len(),
        )?;
        let per_node_scalar = self.scalar.num_features;
        let per_node_vector = self.vector.num_features * self.vector.num_equivariance;
        let per_node_out = per_node_scalar + per_node_vector;
        let mut scalar_grad_out = vec![0.0_f32; num_nodes * per_node_scalar];
        let mut vector_grad_out = vec![0.0_f32; num_nodes * per_node_vector];
        for node in 0..num_nodes {
            let out_src = node * per_node_out;
            let scalar_dst = node * per_node_scalar;
            let vector_dst = node * per_node_vector;
            scalar_grad_out[scalar_dst..scalar_dst + per_node_scalar]
                .copy_from_slice(&grad_out[out_src..out_src + per_node_scalar]);
            vector_grad_out[vector_dst..vector_dst + per_node_vector]
                .copy_from_slice(&grad_out[out_src + per_node_scalar..out_src + per_node_out]);
        }
        self.scalar
            .vjp_x_f32(x, node_attrs, &scalar_grad_out, grad_x)?;
        self.vector
            .vjp_x_f32(x, node_attrs, &vector_grad_out, grad_x)?;
        Ok(())
    }

    pub fn vjp_x(
        &self,
        x: &[f64],
        node_attrs: &[f64],
        grad_out: &[f64],
        grad_x: &mut [f64],
    ) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes(
            x,
            node_attrs,
            grad_out.len(),
            "mace_scalar_vector_symmetric_contraction_vjp_x grad_out",
        )?;
        let per_node_scalar = self.scalar.num_features;
        let per_node_vector = self.vector.num_features * self.vector.num_equivariance;
        let per_node_out = per_node_scalar + per_node_vector;
        let mut scalar_grad_out = vec![0.0; num_nodes * per_node_scalar];
        let mut vector_grad_out = vec![0.0; num_nodes * per_node_vector];
        for node in 0..num_nodes {
            let out_src = node * per_node_out;
            let scalar_dst = node * per_node_scalar;
            let vector_dst = node * per_node_vector;
            scalar_grad_out[scalar_dst..scalar_dst + per_node_scalar]
                .copy_from_slice(&grad_out[out_src..out_src + per_node_scalar]);
            vector_grad_out[vector_dst..vector_dst + per_node_vector]
                .copy_from_slice(&grad_out[out_src + per_node_scalar..out_src + per_node_out]);
        }
        self.scalar.vjp_x(x, node_attrs, &scalar_grad_out, grad_x)?;
        self.vector.vjp_x(x, node_attrs, &vector_grad_out, grad_x)?;
        Ok(())
    }

    pub fn jvp_x(
        &self,
        x: &[f64],
        x_dot: &[f64],
        node_attrs: &[f64],
        out_dot: &mut [f64],
    ) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes(
            x,
            node_attrs,
            out_dot.len(),
            "mace_scalar_vector_symmetric_contraction_jvp_x out_dot",
        )?;
        self.scalar.check_len(
            "mace_scalar_vector_symmetric_contraction_jvp_x x_dot",
            x.len(),
            x_dot.len(),
        )?;
        let scalar_len = num_nodes * self.scalar.num_features;
        let vector_len = num_nodes * self.vector.num_features * self.vector.num_equivariance;
        let mut scalar_out_dot = vec![0.0; scalar_len];
        let mut vector_out_dot = vec![0.0; vector_len];
        self.scalar
            .jvp_x(x, x_dot, node_attrs, &mut scalar_out_dot)?;
        self.vector
            .jvp_x(x, x_dot, node_attrs, &mut vector_out_dot)?;
        let per_node_scalar = self.scalar.num_features;
        let per_node_vector = self.vector.num_features * self.vector.num_equivariance;
        let per_node_out = per_node_scalar + per_node_vector;
        for node in 0..num_nodes {
            let scalar_src = node * per_node_scalar;
            let vector_src = node * per_node_vector;
            let out_dst = node * per_node_out;
            out_dot[out_dst..out_dst + per_node_scalar]
                .copy_from_slice(&scalar_out_dot[scalar_src..scalar_src + per_node_scalar]);
            out_dot[out_dst + per_node_scalar..out_dst + per_node_out]
                .copy_from_slice(&vector_out_dot[vector_src..vector_src + per_node_vector]);
        }
        Ok(())
    }

    pub fn vjp_x_linearize(
        &self,
        x: &[f64],
        x_dot: &[f64],
        node_attrs: &[f64],
        grad_out: &[f64],
        grad_out_dot: &[f64],
        grad_x_dot: &mut [f64],
    ) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes(
            x,
            node_attrs,
            grad_out.len(),
            "mace_scalar_vector_symmetric_contraction_vjp_x_linearize grad_out",
        )?;
        self.scalar.check_len(
            "mace_scalar_vector_symmetric_contraction_vjp_x_linearize x_dot",
            x.len(),
            x_dot.len(),
        )?;
        self.scalar.check_len(
            "mace_scalar_vector_symmetric_contraction_vjp_x_linearize grad_out_dot",
            grad_out.len(),
            grad_out_dot.len(),
        )?;
        self.scalar.check_len(
            "mace_scalar_vector_symmetric_contraction_vjp_x_linearize grad_x_dot",
            x.len(),
            grad_x_dot.len(),
        )?;
        let per_node_scalar = self.scalar.num_features;
        let per_node_vector = self.vector.num_features * self.vector.num_equivariance;
        let per_node_out = per_node_scalar + per_node_vector;
        let mut scalar_grad_out = vec![0.0; num_nodes * per_node_scalar];
        let mut scalar_grad_out_dot = vec![0.0; num_nodes * per_node_scalar];
        let mut vector_grad_out = vec![0.0; num_nodes * per_node_vector];
        let mut vector_grad_out_dot = vec![0.0; num_nodes * per_node_vector];
        for node in 0..num_nodes {
            let out_src = node * per_node_out;
            let scalar_dst = node * per_node_scalar;
            let vector_dst = node * per_node_vector;
            scalar_grad_out[scalar_dst..scalar_dst + per_node_scalar]
                .copy_from_slice(&grad_out[out_src..out_src + per_node_scalar]);
            scalar_grad_out_dot[scalar_dst..scalar_dst + per_node_scalar]
                .copy_from_slice(&grad_out_dot[out_src..out_src + per_node_scalar]);
            vector_grad_out[vector_dst..vector_dst + per_node_vector]
                .copy_from_slice(&grad_out[out_src + per_node_scalar..out_src + per_node_out]);
            vector_grad_out_dot[vector_dst..vector_dst + per_node_vector]
                .copy_from_slice(&grad_out_dot[out_src + per_node_scalar..out_src + per_node_out]);
        }
        self.scalar.vjp_x_linearize(
            x,
            x_dot,
            node_attrs,
            &scalar_grad_out,
            &scalar_grad_out_dot,
            grad_x_dot,
        )?;
        self.vector.vjp_x_linearize(
            x,
            x_dot,
            node_attrs,
            &vector_grad_out,
            &vector_grad_out_dot,
            grad_x_dot,
        )?;
        Ok(())
    }

    fn check_runtime_shapes(
        &self,
        x: &[f64],
        node_attrs: &[f64],
        out_len: usize,
        out_op: &'static str,
    ) -> Result<usize> {
        let per_node_x = self.scalar.num_features * self.scalar.num_ell;
        if !x.len().is_multiple_of(per_node_x) {
            return Err(CoreError::ShapeMismatch {
                op: "mace_scalar_vector_symmetric_contraction x",
                expected: per_node_x,
                actual: x.len(),
            });
        }
        let num_nodes = x.len() / per_node_x;
        let expected_node_attrs = num_nodes * self.scalar.num_elements;
        if node_attrs.len() != expected_node_attrs {
            return Err(CoreError::ShapeMismatch {
                op: "mace_scalar_vector_symmetric_contraction node_attrs",
                expected: expected_node_attrs,
                actual: node_attrs.len(),
            });
        }
        let per_node_out =
            self.scalar.num_features + self.vector.num_features * self.vector.num_equivariance;
        let expected_out = num_nodes * per_node_out;
        if out_len != expected_out {
            return Err(CoreError::ShapeMismatch {
                op: out_op,
                expected: expected_out,
                actual: out_len,
            });
        }
        Ok(num_nodes)
    }

    fn check_runtime_shapes_f32(
        &self,
        x: &[f32],
        node_attrs: &[f32],
        out_len: usize,
        out_op: &'static str,
    ) -> Result<usize> {
        let per_node_x = self.scalar.num_features * self.scalar.num_ell;
        if !x.len().is_multiple_of(per_node_x) {
            return Err(CoreError::ShapeMismatch {
                op: "mace_scalar_vector_symmetric_contraction_f32 x",
                expected: per_node_x,
                actual: x.len(),
            });
        }
        let num_nodes = x.len() / per_node_x;
        let expected_node_attrs = num_nodes * self.scalar.num_elements;
        if node_attrs.len() != expected_node_attrs {
            return Err(CoreError::ShapeMismatch {
                op: "mace_scalar_vector_symmetric_contraction_f32 node_attrs",
                expected: expected_node_attrs,
                actual: node_attrs.len(),
            });
        }
        let per_node_out =
            self.scalar.num_features + self.vector.num_features * self.vector.num_equivariance;
        let expected_out = num_nodes * per_node_out;
        if out_len != expected_out {
            return Err(CoreError::ShapeMismatch {
                op: out_op,
                expected: expected_out,
                actual: out_len,
            });
        }
        Ok(num_nodes)
    }
}

impl<'a> MaceVectorSymmetricContraction<'a> {
    pub fn validate(&self) -> Result<()> {
        self.check_len(
            "mace_vector_symmetric_contraction u1",
            self.num_equivariance * self.num_ell * self.u1_params,
            self.u1.len(),
        )?;
        self.check_len(
            "mace_vector_symmetric_contraction u2",
            self.num_equivariance * self.num_ell * self.num_ell * self.u2_params,
            self.u2.len(),
        )?;
        self.check_len(
            "mace_vector_symmetric_contraction u3",
            self.num_equivariance * self.num_ell * self.num_ell * self.num_ell * self.u3_params,
            self.u3.len(),
        )?;
        self.check_len(
            "mace_vector_symmetric_contraction weights_u1",
            self.num_elements * self.u1_params * self.num_features,
            self.weights_u1.len(),
        )?;
        self.check_len(
            "mace_vector_symmetric_contraction weights_u2",
            self.num_elements * self.u2_params * self.num_features,
            self.weights_u2.len(),
        )?;
        self.check_len(
            "mace_vector_symmetric_contraction weights_u3",
            self.num_elements * self.u3_params * self.num_features,
            self.weights_u3.len(),
        )?;
        validate_nonzero(
            "mace_vector_symmetric_contraction num_elements",
            self.num_elements,
        )?;
        validate_nonzero(
            "mace_vector_symmetric_contraction num_features",
            self.num_features,
        )?;
        validate_nonzero(
            "mace_vector_symmetric_contraction num_equivariance",
            self.num_equivariance,
        )?;
        validate_nonzero("mace_vector_symmetric_contraction num_ell", self.num_ell)?;
        validate_nonzero(
            "mace_vector_symmetric_contraction u1_params",
            self.u1_params,
        )?;
        validate_nonzero(
            "mace_vector_symmetric_contraction u2_params",
            self.u2_params,
        )?;
        validate_nonzero(
            "mace_vector_symmetric_contraction u3_params",
            self.u3_params,
        )?;
        Ok(())
    }

    pub fn entries(&self) -> Result<MaceVectorSymmetricContractionEntries> {
        self.validate()?;
        Ok(self.nonzero_entries())
    }

    pub fn prepare_single_element_product_plan(
        &self,
    ) -> Result<MaceVectorSingleElementProductPlan> {
        self.validate()?;
        if self.num_elements != 1 {
            return Err(CoreError::InvalidTensorProduct(
                "mace vector single-element product plan requires num_elements == 1".to_owned(),
            ));
        }
        let owned_entries;
        let entries = if let Some(entries) = self.entries {
            entries
        } else {
            owned_entries = self.nonzero_entries();
            &owned_entries
        };
        let u2_groups = vector_u2_entries_by_equiv_pair(&entries.u2);
        let u3_groups = vector_u3_entries_by_equiv_triple(&entries.u3);
        let mut weighted_u1 = vec![Vec::new(); self.num_features * self.num_equivariance];
        let mut weighted_u2 = vec![Vec::new(); self.num_features * self.num_equivariance];
        let mut weighted_u3 = vec![Vec::new(); self.num_features * self.num_equivariance];
        let mut feature_u1 = vec![Vec::new(); self.num_equivariance];
        let mut feature_u2 = vec![Vec::new(); self.num_equivariance];
        let mut feature_u3 = vec![Vec::new(); self.num_equivariance];
        let mut w1 = vec![0.0; self.u1_params];
        let mut w2 = vec![0.0; self.u2_params];
        let mut w3 = vec![0.0; self.u3_params];
        for feature in 0..self.num_features {
            self.feature_weights_for_element_into(0, feature, &mut w1, &mut w2, &mut w3);
            weight_vector_monomial_groups(
                &w1,
                &w2,
                &w3,
                &entries.u1,
                &u2_groups,
                &u3_groups,
                &mut feature_u1,
                &mut feature_u2,
                &mut feature_u3,
            );
            for equiv in 0..self.num_equivariance {
                let index = feature * self.num_equivariance + equiv;
                weighted_u1[index] = feature_u1[equiv].clone();
                weighted_u2[index] = feature_u2[equiv].clone();
                weighted_u3[index] = feature_u3[equiv].clone();
            }
        }
        let weighted_u1_f32: Vec<Vec<VectorU1WeightedEntryF32>> = weighted_u1
            .iter()
            .map(|entries| {
                entries
                    .iter()
                    .map(|entry| VectorU1WeightedEntryF32 {
                        ell: entry.ell,
                        value: entry.value as f32,
                    })
                    .collect()
            })
            .collect();
        let weighted_u2_f32: Vec<Vec<VectorU2WeightedEntryF32>> = weighted_u2
            .iter()
            .map(|entries| {
                entries
                    .iter()
                    .map(|entry| VectorU2WeightedEntryF32 {
                        left: entry.left,
                        right: entry.right,
                        value: entry.value as f32,
                    })
                    .collect()
            })
            .collect();
        let weighted_u3_f32: Vec<Vec<VectorU3WeightedEntryF32>> = weighted_u3
            .iter()
            .map(|entries| {
                entries
                    .iter()
                    .map(|entry| VectorU3WeightedEntryF32 {
                        left: entry.left,
                        right: entry.right,
                        inner: entry.inner,
                        value: entry.value as f32,
                    })
                    .collect()
            })
            .collect();
        let (weighted_u1_f32_flat, weighted_u1_f32_ranges) =
            flatten_weighted_entry_groups(&weighted_u1_f32);
        let (weighted_u2_f32_flat, weighted_u2_f32_ranges) =
            flatten_weighted_entry_groups(&weighted_u2_f32);
        let (weighted_u3_f32_flat, weighted_u3_f32_ranges) =
            flatten_weighted_entry_groups(&weighted_u3_f32);
        Ok(MaceVectorSingleElementProductPlan {
            num_features: self.num_features,
            num_ell: self.num_ell,
            num_equivariance: self.num_equivariance,
            weighted_u1,
            weighted_u2,
            weighted_u3,
            weighted_u1_f32_flat,
            weighted_u2_f32_flat,
            weighted_u3_f32_flat,
            weighted_u1_f32_ranges,
            weighted_u2_f32_ranges,
            weighted_u3_f32_ranges,
        })
    }

    pub fn forward(&self, x: &[f64], node_attrs: &[f64], out: &mut [f64]) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes(
            x,
            node_attrs,
            out.len(),
            "mace_vector_symmetric_contraction_forward out",
        )?;
        out.fill(0.0);
        let mut w1 = vec![0.0; self.u1_params];
        let mut w2 = vec![0.0; self.u2_params];
        let mut w3 = vec![0.0; self.u3_params];
        let owned_entries;
        let entries = if let Some(entries) = self.entries {
            entries
        } else {
            owned_entries = self.nonzero_entries();
            &owned_entries
        };
        let u1_entries = &entries.u1;
        let u2_entries = &entries.u2;
        let u3_entries = &entries.u3;
        if let Some(active_elements) =
            active_unit_one_hot_elements(node_attrs, num_nodes, self.num_elements)
        {
            #[cfg(target_arch = "x86_64")]
            if self.num_ell == 16 && mace_f64_avx2_enabled() {
                let node_x_len = self.num_features * 16;
                let out_row_len = self.num_features * self.num_equivariance;
                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() > 1 && num_nodes >= 32 {
                    out.par_chunks_mut(out_row_len)
                        .enumerate()
                        .with_min_len(2)
                        .for_each(|(node, out_row)| {
                            mace_vector_contraction_forward_node_avx2_f64_safe(
                                self,
                                &x[node * node_x_len..(node + 1) * node_x_len],
                                active_elements[node],
                                out_row,
                                u1_entries,
                                u2_entries,
                                u3_entries,
                            );
                        });
                    return Ok(());
                }
                for (node, out_row) in out.chunks_mut(out_row_len).enumerate() {
                    mace_vector_contraction_forward_node_avx2_f64_safe(
                        self,
                        &x[node * node_x_len..(node + 1) * node_x_len],
                        active_elements[node],
                        out_row,
                        u1_entries,
                        u2_entries,
                        u3_entries,
                    );
                }
                return Ok(());
            }
            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() > 1 && self.num_ell == 16 && num_nodes > 1 {
                let output_row_len = self.num_features * self.num_equivariance;
                out.par_chunks_mut(output_row_len)
                    .enumerate()
                    .for_each(|(node, out_row)| {
                        let element = active_elements[node];
                        let mut local_w1 = vec![0.0; self.u1_params];
                        let mut local_w2 = vec![0.0; self.u2_params];
                        let mut local_w3 = vec![0.0; self.u3_params];
                        for feature in 0..self.num_features {
                            self.feature_weights_for_element_into(
                                element,
                                feature,
                                &mut local_w1,
                                &mut local_w2,
                                &mut local_w3,
                            );
                            self.forward_node_feature_with_weights_into_row(
                                x, out_row, node, feature, &local_w1, &local_w2, &local_w3,
                                u1_entries, u2_entries, u3_entries,
                            );
                        }
                    });
                return Ok(());
            }
            let mut nodes_by_element = vec![Vec::new(); self.num_elements];
            for (node, &element) in active_elements.iter().enumerate() {
                nodes_by_element[element].push(node);
            }
            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() == 1 && self.num_ell == 16 {
                let u2_by_pair = vector_u2_entries_by_equiv_pair(u2_entries);
                let u3_by_triple = vector_u3_entries_by_equiv_triple(u3_entries);
                let mut weighted_u1 = vec![Vec::new(); self.num_equivariance];
                let mut weighted_u2 = vec![Vec::new(); self.num_equivariance];
                let mut weighted_u3 = vec![Vec::new(); self.num_equivariance];
                for feature in 0..self.num_features {
                    for (element, nodes) in nodes_by_element.iter().enumerate() {
                        if nodes.is_empty() {
                            continue;
                        }
                        self.feature_weights_for_element_into(
                            element, feature, &mut w1, &mut w2, &mut w3,
                        );
                        weight_vector_monomial_groups(
                            &w1,
                            &w2,
                            &w3,
                            u1_entries,
                            &u2_by_pair,
                            &u3_by_triple,
                            &mut weighted_u1,
                            &mut weighted_u2,
                            &mut weighted_u3,
                        );
                        for &node in nodes {
                            let x_base = self.x_index(node, feature, 0);
                            let x_row = &x[x_base..x_base + 16];
                            let out_base = self.out_index(node, feature, 0);
                            for equiv in 0..self.num_equivariance {
                                out[out_base + equiv] = evaluate_vector_weighted_monomials_ell16(
                                    x_row,
                                    &weighted_u1[equiv],
                                    &weighted_u2[equiv],
                                    &weighted_u3[equiv],
                                );
                            }
                        }
                    }
                }
                return Ok(());
            }
            let mut weighted_u1 = u1_entries
                .iter()
                .map(|entries| vec![0.0; entries.len()])
                .collect::<Vec<_>>();
            let mut weighted_u2 = u2_entries
                .iter()
                .map(|entries| vec![0.0; entries.len()])
                .collect::<Vec<_>>();
            let mut weighted_u3 = u3_entries
                .iter()
                .map(|entries| vec![0.0; entries.len()])
                .collect::<Vec<_>>();
            for feature in 0..self.num_features {
                for (element, nodes) in nodes_by_element.iter().enumerate() {
                    if nodes.is_empty() {
                        continue;
                    }
                    self.feature_weights_for_element_into(
                        element, feature, &mut w1, &mut w2, &mut w3,
                    );
                    self.weight_vector_entries(
                        &w1,
                        &w2,
                        &w3,
                        u1_entries,
                        u2_entries,
                        u3_entries,
                        &mut weighted_u1,
                        &mut weighted_u2,
                        &mut weighted_u3,
                    );
                    for &node in nodes {
                        self.forward_node_feature_with_weighted_entries(
                            x,
                            out,
                            node,
                            feature,
                            u1_entries,
                            &weighted_u1,
                            u2_entries,
                            &weighted_u2,
                            u3_entries,
                            &weighted_u3,
                        );
                    }
                }
            }
            return Ok(());
        }
        for node in 0..num_nodes {
            for feature in 0..self.num_features {
                self.forward_node_feature(
                    x, node_attrs, out, node, feature, &mut w1, &mut w2, &mut w3, u1_entries,
                    u2_entries, u3_entries,
                );
            }
        }
        Ok(())
    }

    pub fn forward_f32(&self, x: &[f32], node_attrs: &[f32], out: &mut [f32]) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes_f32(
            x,
            node_attrs,
            out.len(),
            "mace_vector_symmetric_contraction_forward_f32 out",
        )?;
        out.fill(0.0);
        let mut w1 = vec![0.0_f32; self.u1_params];
        let mut w2 = vec![0.0_f32; self.u2_params];
        let mut w3 = vec![0.0_f32; self.u3_params];
        let owned_entries;
        let entries = if let Some(entries) = self.entries {
            entries
        } else {
            owned_entries = self.nonzero_entries();
            &owned_entries
        };
        let u1_entries = &entries.u1;
        let u2_entries = &entries.u2;
        let u3_entries = &entries.u3;
        if let Some(active_elements) =
            active_unit_one_hot_elements_f32(node_attrs, num_nodes, self.num_elements)
        {
            let mut nodes_by_element = vec![Vec::new(); self.num_elements];
            for (node, &element) in active_elements.iter().enumerate() {
                nodes_by_element[element].push(node);
            }
            for feature in 0..self.num_features {
                for (element, nodes) in nodes_by_element.iter().enumerate() {
                    if nodes.is_empty() {
                        continue;
                    }
                    self.feature_weights_for_element_into_f32(
                        element, feature, &mut w1, &mut w2, &mut w3,
                    );
                    for &node in nodes {
                        self.forward_node_feature_with_weights_f32(
                            x, out, node, feature, &w1, &w2, &w3, u1_entries, u2_entries,
                            u3_entries,
                        );
                    }
                }
            }
            return Ok(());
        }
        for node in 0..num_nodes {
            for feature in 0..self.num_features {
                self.feature_weights_into_f32(node_attrs, node, feature, &mut w1, &mut w2, &mut w3);
                self.forward_node_feature_with_weights_f32(
                    x, out, node, feature, &w1, &w2, &w3, u1_entries, u2_entries, u3_entries,
                );
            }
        }
        Ok(())
    }

    pub fn vjp_x_f32(
        &self,
        x: &[f32],
        node_attrs: &[f32],
        grad_out: &[f32],
        grad_x: &mut [f32],
    ) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes_f32(
            x,
            node_attrs,
            grad_out.len(),
            "mace_vector_symmetric_contraction_vjp_x_f32 grad_out",
        )?;
        self.check_len(
            "mace_vector_symmetric_contraction_vjp_x_f32 grad_x",
            x.len(),
            grad_x.len(),
        )?;
        let mut w1 = vec![0.0_f32; self.u1_params];
        let mut w2 = vec![0.0_f32; self.u2_params];
        let mut w3 = vec![0.0_f32; self.u3_params];
        let owned_entries;
        let entries = if let Some(entries) = self.entries {
            entries
        } else {
            owned_entries = self.nonzero_entries();
            &owned_entries
        };
        let u1_entries = &entries.u1;
        let u2_entries = &entries.u2;
        let u3_entries = &entries.u3;
        if let Some(active_elements) =
            active_unit_one_hot_elements_f32(node_attrs, num_nodes, self.num_elements)
        {
            let mut nodes_by_element = vec![Vec::new(); self.num_elements];
            for (node, &element) in active_elements.iter().enumerate() {
                nodes_by_element[element].push(node);
            }
            for feature in 0..self.num_features {
                for (element, nodes) in nodes_by_element.iter().enumerate() {
                    if nodes.is_empty()
                        || !nodes.iter().any(|&node| {
                            let out_base = self.out_index(node, feature, 0);
                            grad_out[out_base..out_base + self.num_equivariance]
                                .iter()
                                .any(|&value| value != 0.0)
                        })
                    {
                        continue;
                    }
                    self.feature_weights_for_element_into_f32(
                        element, feature, &mut w1, &mut w2, &mut w3,
                    );
                    for &node in nodes {
                        self.vjp_node_feature_with_weights_f32(
                            x, grad_out, grad_x, node, feature, &w1, &w2, &w3, u1_entries,
                            u2_entries, u3_entries,
                        );
                    }
                }
            }
            return Ok(());
        }
        for node in 0..num_nodes {
            for feature in 0..self.num_features {
                let out_base = self.out_index(node, feature, 0);
                if grad_out[out_base..out_base + self.num_equivariance]
                    .iter()
                    .all(|&value| value == 0.0)
                {
                    continue;
                }
                self.feature_weights_into_f32(node_attrs, node, feature, &mut w1, &mut w2, &mut w3);
                self.vjp_node_feature_with_weights_f32(
                    x, grad_out, grad_x, node, feature, &w1, &w2, &w3, u1_entries, u2_entries,
                    u3_entries,
                );
            }
        }
        Ok(())
    }

    pub fn vjp_x(
        &self,
        x: &[f64],
        node_attrs: &[f64],
        grad_out: &[f64],
        grad_x: &mut [f64],
    ) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes(
            x,
            node_attrs,
            grad_out.len(),
            "mace_vector_symmetric_contraction_vjp_x grad_out",
        )?;
        self.check_len(
            "mace_vector_symmetric_contraction_vjp_x grad_x",
            x.len(),
            grad_x.len(),
        )?;
        let mut w1 = vec![0.0; self.u1_params];
        let mut w2 = vec![0.0; self.u2_params];
        let mut w3 = vec![0.0; self.u3_params];
        let mut grad_values = vec![0.0; self.num_ell];
        let owned_entries;
        let entries = if let Some(entries) = self.entries {
            entries
        } else {
            owned_entries = self.nonzero_entries();
            &owned_entries
        };
        let u1_entries = &entries.u1;
        let u2_entries = &entries.u2;
        let u3_entries = &entries.u3;
        if let Some(active_elements) =
            active_unit_one_hot_elements(node_attrs, num_nodes, self.num_elements)
        {
            #[cfg(target_arch = "x86_64")]
            if self.num_ell == 16 && mace_f64_avx2_enabled() {
                let node_x_len = self.num_features * 16;
                let out_row_len = self.num_features * self.num_equivariance;
                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() > 1 && num_nodes >= 32 {
                    grad_x
                        .par_chunks_mut(node_x_len)
                        .enumerate()
                        .with_min_len(2)
                        .for_each(|(node, grad_x_node)| {
                            mace_vector_contraction_vjp_node_avx2_f64_safe(
                                self,
                                &x[node * node_x_len..(node + 1) * node_x_len],
                                &grad_out[node * out_row_len..(node + 1) * out_row_len],
                                active_elements[node],
                                grad_x_node,
                                u1_entries,
                                u2_entries,
                                u3_entries,
                            );
                        });
                    return Ok(());
                }
                for (node, grad_x_node) in grad_x.chunks_mut(node_x_len).enumerate() {
                    mace_vector_contraction_vjp_node_avx2_f64_safe(
                        self,
                        &x[node * node_x_len..(node + 1) * node_x_len],
                        &grad_out[node * out_row_len..(node + 1) * out_row_len],
                        active_elements[node],
                        grad_x_node,
                        u1_entries,
                        u2_entries,
                        u3_entries,
                    );
                }
                return Ok(());
            }
            let mut nodes_by_element = vec![Vec::new(); self.num_elements];
            for (node, &element) in active_elements.iter().enumerate() {
                nodes_by_element[element].push(node);
            }
            let use_ell4_equiv3 = self.num_ell == 4 && self.num_equivariance == 3;
            let use_ell16 = self.num_ell == 16;
            let mut weighted_u1 = u1_entries
                .iter()
                .map(|entries| vec![0.0; entries.len()])
                .collect::<Vec<_>>();
            let mut weighted_u2 = u2_entries
                .iter()
                .map(|entries| vec![0.0; entries.len()])
                .collect::<Vec<_>>();
            let mut weighted_u3 = u3_entries
                .iter()
                .map(|entries| vec![0.0; entries.len()])
                .collect::<Vec<_>>();
            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() > 1 && use_ell4_equiv3 && num_nodes > 1 {
                let features = (0..self.num_features).collect::<Vec<_>>();
                let updates_by_chunk = features
                    .par_chunks(8)
                    .map(|feature_chunk| {
                        let mut local_w1 = vec![0.0; self.u1_params];
                        let mut local_w2 = vec![0.0; self.u2_params];
                        let mut local_w3 = vec![0.0; self.u3_params];
                        let mut local_weighted_u1 = u1_entries
                            .iter()
                            .map(|entries| vec![0.0; entries.len()])
                            .collect::<Vec<_>>();
                        let mut local_weighted_u2 = u2_entries
                            .iter()
                            .map(|entries| vec![0.0; entries.len()])
                            .collect::<Vec<_>>();
                        let mut local_weighted_u3 = u3_entries
                            .iter()
                            .map(|entries| vec![0.0; entries.len()])
                            .collect::<Vec<_>>();
                        let mut updates = Vec::new();
                        for &feature in feature_chunk {
                            for (element, nodes) in nodes_by_element.iter().enumerate() {
                                if nodes.is_empty() {
                                    continue;
                                }
                                self.feature_weights_for_element_into(
                                    element,
                                    feature,
                                    &mut local_w1,
                                    &mut local_w2,
                                    &mut local_w3,
                                );
                                self.weight_vector_entries(
                                    &local_w1,
                                    &local_w2,
                                    &local_w3,
                                    u1_entries,
                                    u2_entries,
                                    u3_entries,
                                    &mut local_weighted_u1,
                                    &mut local_weighted_u2,
                                    &mut local_weighted_u3,
                                );
                                for &node in nodes {
                                    if let Some(grad) = self
                                        .vjp_node_feature_grad_weighted_entries_ell4_equiv3(
                                            x,
                                            grad_out,
                                            node,
                                            feature,
                                            u1_entries,
                                            &local_weighted_u1,
                                            u2_entries,
                                            &local_weighted_u2,
                                            u3_entries,
                                            &local_weighted_u3,
                                        )
                                    {
                                        updates.push((node, feature, grad));
                                    }
                                }
                            }
                        }
                        updates
                    })
                    .collect::<Vec<_>>();
                for updates in updates_by_chunk {
                    for (node, feature, grad_values4) in updates {
                        self.add_ell4_grad_to_x(grad_x, node, feature, &grad_values4);
                    }
                }
                return Ok(());
            }
            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() > 1 && use_ell16 && num_nodes > 1 {
                let features = (0..self.num_features).collect::<Vec<_>>();
                let updates_by_chunk = features
                    .par_chunks(4)
                    .map(|feature_chunk| {
                        let mut local_w1 = vec![0.0; self.u1_params];
                        let mut local_w2 = vec![0.0; self.u2_params];
                        let mut local_w3 = vec![0.0; self.u3_params];
                        let mut local_weighted_u1 = u1_entries
                            .iter()
                            .map(|entries| vec![0.0; entries.len()])
                            .collect::<Vec<_>>();
                        let mut local_weighted_u2 = u2_entries
                            .iter()
                            .map(|entries| vec![0.0; entries.len()])
                            .collect::<Vec<_>>();
                        let mut local_weighted_u3 = u3_entries
                            .iter()
                            .map(|entries| vec![0.0; entries.len()])
                            .collect::<Vec<_>>();
                        let mut updates = Vec::with_capacity(feature_chunk.len() * num_nodes);
                        for &feature in feature_chunk {
                            for (element, nodes) in nodes_by_element.iter().enumerate() {
                                if nodes.is_empty() {
                                    continue;
                                }
                                self.feature_weights_for_element_into(
                                    element,
                                    feature,
                                    &mut local_w1,
                                    &mut local_w2,
                                    &mut local_w3,
                                );
                                self.weight_vector_entries(
                                    &local_w1,
                                    &local_w2,
                                    &local_w3,
                                    u1_entries,
                                    u2_entries,
                                    u3_entries,
                                    &mut local_weighted_u1,
                                    &mut local_weighted_u2,
                                    &mut local_weighted_u3,
                                );
                                for &node in nodes {
                                    if let Some(grad) = self
                                        .vjp_node_feature_grad_weighted_entries_ell16(
                                            x,
                                            grad_out,
                                            node,
                                            feature,
                                            u1_entries,
                                            &local_weighted_u1,
                                            u2_entries,
                                            &local_weighted_u2,
                                            u3_entries,
                                            &local_weighted_u3,
                                        )
                                    {
                                        updates.push((self.x_index(node, feature, 0), grad));
                                    }
                                }
                            }
                        }
                        updates
                    })
                    .collect::<Vec<_>>();
                for updates in updates_by_chunk {
                    for (x_base, grad_values16) in updates {
                        add_ell16_grad_to_x(grad_x, x_base, &grad_values16);
                    }
                }
                return Ok(());
            }
            #[cfg(feature = "parallel")]
            if use_ell16 && rayon::current_num_threads() == 1 {
                let u2_by_pair = vector_u2_entries_by_equiv_pair(u2_entries);
                let u3_by_triple = vector_u3_entries_by_equiv_triple(u3_entries);
                let mut weighted_u1 = vec![Vec::new(); self.num_equivariance];
                let mut weighted_u2 = vec![Vec::new(); self.num_equivariance];
                let mut weighted_u3 = vec![Vec::new(); self.num_equivariance];
                for feature in 0..self.num_features {
                    for (element, nodes) in nodes_by_element.iter().enumerate() {
                        if nodes.is_empty() {
                            continue;
                        }
                        self.feature_weights_for_element_into(
                            element, feature, &mut w1, &mut w2, &mut w3,
                        );
                        weight_vector_monomial_groups(
                            &w1,
                            &w2,
                            &w3,
                            u1_entries,
                            &u2_by_pair,
                            &u3_by_triple,
                            &mut weighted_u1,
                            &mut weighted_u2,
                            &mut weighted_u3,
                        );
                        for &node in nodes {
                            let out_base = self.out_index(node, feature, 0);
                            let grad_out_row =
                                &grad_out[out_base..out_base + self.num_equivariance];
                            if grad_out_row.iter().all(|&value| value == 0.0) {
                                continue;
                            }
                            let x_base = self.x_index(node, feature, 0);
                            let x_row = &x[x_base..x_base + 16];
                            let mut grad_values = [0.0_f64; 16];
                            for (equiv, &bar_output) in grad_out_row.iter().enumerate() {
                                accumulate_vector_weighted_monomials_ell16(
                                    bar_output,
                                    x_row,
                                    &weighted_u1[equiv],
                                    &weighted_u2[equiv],
                                    &weighted_u3[equiv],
                                    &mut grad_values,
                                );
                            }
                            add_ell16_grad_to_x(grad_x, x_base, &grad_values);
                        }
                    }
                }
                return Ok(());
            }
            #[cfg(feature = "parallel")]
            if use_ell16 {
                for feature in 0..self.num_features {
                    for (element, nodes) in nodes_by_element.iter().enumerate() {
                        if nodes.is_empty() {
                            continue;
                        }
                        self.feature_weights_for_element_into(
                            element, feature, &mut w1, &mut w2, &mut w3,
                        );
                        self.weight_vector_entries(
                            &w1,
                            &w2,
                            &w3,
                            u1_entries,
                            u2_entries,
                            u3_entries,
                            &mut weighted_u1,
                            &mut weighted_u2,
                            &mut weighted_u3,
                        );
                        for &node in nodes {
                            if let Some(grad_values16) = self
                                .vjp_node_feature_grad_weighted_entries_ell16(
                                    x,
                                    grad_out,
                                    node,
                                    feature,
                                    u1_entries,
                                    &weighted_u1,
                                    u2_entries,
                                    &weighted_u2,
                                    u3_entries,
                                    &weighted_u3,
                                )
                            {
                                add_ell16_grad_to_x(
                                    grad_x,
                                    self.x_index(node, feature, 0),
                                    &grad_values16,
                                );
                            }
                        }
                    }
                }
                return Ok(());
            }
            for feature in 0..self.num_features {
                for (element, nodes) in nodes_by_element.iter().enumerate() {
                    if nodes.is_empty() {
                        continue;
                    }
                    self.feature_weights_for_element_into(
                        element, feature, &mut w1, &mut w2, &mut w3,
                    );
                    self.weight_vector_entries(
                        &w1,
                        &w2,
                        &w3,
                        u1_entries,
                        u2_entries,
                        u3_entries,
                        &mut weighted_u1,
                        &mut weighted_u2,
                        &mut weighted_u3,
                    );
                    if use_ell4_equiv3 {
                        #[cfg(feature = "parallel")]
                        if rayon::current_num_threads() > 1 && nodes.len() >= 8 {
                            let grads = nodes
                                .par_iter()
                                .filter_map(|&node| {
                                    self.vjp_node_feature_grad_weighted_entries_ell4_equiv3(
                                        x,
                                        grad_out,
                                        node,
                                        feature,
                                        u1_entries,
                                        &weighted_u1,
                                        u2_entries,
                                        &weighted_u2,
                                        u3_entries,
                                        &weighted_u3,
                                    )
                                    .map(|grad| (node, grad))
                                })
                                .collect::<Vec<_>>();
                            for (node, grad_values4) in grads {
                                self.add_ell4_grad_to_x(grad_x, node, feature, &grad_values4);
                            }
                            continue;
                        }
                        let mut grad_values4 = [0.0; 4];
                        for &node in nodes {
                            self.vjp_node_feature_with_weighted_entries_ell4_equiv3(
                                x,
                                grad_out,
                                grad_x,
                                node,
                                feature,
                                &mut grad_values4,
                                u1_entries,
                                &weighted_u1,
                                u2_entries,
                                &weighted_u2,
                                u3_entries,
                                &weighted_u3,
                            );
                        }
                    } else if use_ell16 {
                        for &node in nodes {
                            if let Some(grad_values16) = self
                                .vjp_node_feature_grad_weighted_entries_ell16(
                                    x,
                                    grad_out,
                                    node,
                                    feature,
                                    u1_entries,
                                    &weighted_u1,
                                    u2_entries,
                                    &weighted_u2,
                                    u3_entries,
                                    &weighted_u3,
                                )
                            {
                                add_ell16_grad_to_x(
                                    grad_x,
                                    self.x_index(node, feature, 0),
                                    &grad_values16,
                                );
                            }
                        }
                    } else {
                        for &node in nodes {
                            self.vjp_node_feature_with_weighted_entries(
                                x,
                                grad_out,
                                grad_x,
                                node,
                                feature,
                                &mut grad_values,
                                u1_entries,
                                &weighted_u1,
                                u2_entries,
                                &weighted_u2,
                                u3_entries,
                                &weighted_u3,
                            );
                        }
                    }
                }
            }
            return Ok(());
        }
        for node in 0..num_nodes {
            for feature in 0..self.num_features {
                self.vjp_node_feature(
                    x,
                    node_attrs,
                    grad_out,
                    grad_x,
                    node,
                    feature,
                    &mut w1,
                    &mut w2,
                    &mut w3,
                    &mut grad_values,
                    u1_entries,
                    u2_entries,
                    u3_entries,
                );
            }
        }
        Ok(())
    }

    pub fn jvp_x(
        &self,
        x: &[f64],
        x_dot: &[f64],
        node_attrs: &[f64],
        out_dot: &mut [f64],
    ) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes(
            x,
            node_attrs,
            out_dot.len(),
            "mace_vector_symmetric_contraction_jvp_x out_dot",
        )?;
        self.check_len(
            "mace_vector_symmetric_contraction_jvp_x x_dot",
            x.len(),
            x_dot.len(),
        )?;
        out_dot.fill(0.0);
        let mut w1 = vec![0.0; self.u1_params];
        let mut w2 = vec![0.0; self.u2_params];
        let mut w3 = vec![0.0; self.u3_params];
        let owned_entries;
        let entries = if let Some(entries) = self.entries {
            entries
        } else {
            owned_entries = self.nonzero_entries();
            &owned_entries
        };
        let u1_entries = &entries.u1;
        let u2_entries = &entries.u2;
        let u3_entries = &entries.u3;
        if let Some(active_elements) =
            active_unit_one_hot_elements(node_attrs, num_nodes, self.num_elements)
        {
            let mut nodes_by_element = vec![Vec::new(); self.num_elements];
            for (node, &element) in active_elements.iter().enumerate() {
                nodes_by_element[element].push(node);
            }
            let mut weighted_u1 = u1_entries
                .iter()
                .map(|entries| vec![0.0; entries.len()])
                .collect::<Vec<_>>();
            let mut weighted_u2 = u2_entries
                .iter()
                .map(|entries| vec![0.0; entries.len()])
                .collect::<Vec<_>>();
            let mut weighted_u3 = u3_entries
                .iter()
                .map(|entries| vec![0.0; entries.len()])
                .collect::<Vec<_>>();
            for feature in 0..self.num_features {
                for (element, nodes) in nodes_by_element.iter().enumerate() {
                    if nodes.is_empty() {
                        continue;
                    }
                    if !nodes.iter().any(|&node| {
                        let x_base = self.x_index(node, feature, 0);
                        !all_values_zero(&x_dot[x_base..x_base + self.num_ell])
                    }) {
                        continue;
                    }
                    self.feature_weights_for_element_into(
                        element, feature, &mut w1, &mut w2, &mut w3,
                    );
                    self.weight_vector_entries(
                        &w1,
                        &w2,
                        &w3,
                        u1_entries,
                        u2_entries,
                        u3_entries,
                        &mut weighted_u1,
                        &mut weighted_u2,
                        &mut weighted_u3,
                    );
                    for &node in nodes {
                        self.jvp_node_feature_with_weighted_entries(
                            x,
                            x_dot,
                            out_dot,
                            node,
                            feature,
                            u1_entries,
                            &weighted_u1,
                            u2_entries,
                            &weighted_u2,
                            u3_entries,
                            &weighted_u3,
                        );
                    }
                }
            }
            return Ok(());
        }
        for node in 0..num_nodes {
            for feature in 0..self.num_features {
                let x_base = self.x_index(node, feature, 0);
                if all_values_zero(&x_dot[x_base..x_base + self.num_ell]) {
                    continue;
                }
                self.feature_weights_into(node_attrs, node, feature, &mut w1, &mut w2, &mut w3);
                self.jvp_node_feature_with_weights(
                    x, x_dot, out_dot, node, feature, &w1, &w2, &w3, u1_entries, u2_entries,
                    u3_entries,
                );
            }
        }
        Ok(())
    }

    pub fn vjp_x_linearize(
        &self,
        x: &[f64],
        x_dot: &[f64],
        node_attrs: &[f64],
        grad_out: &[f64],
        grad_out_dot: &[f64],
        grad_x_dot: &mut [f64],
    ) -> Result<()> {
        self.validate()?;
        let num_nodes = self.check_runtime_shapes(
            x,
            node_attrs,
            grad_out.len(),
            "mace_vector_symmetric_contraction_vjp_x_linearize grad_out",
        )?;
        self.check_len(
            "mace_vector_symmetric_contraction_vjp_x_linearize x_dot",
            x.len(),
            x_dot.len(),
        )?;
        self.check_len(
            "mace_vector_symmetric_contraction_vjp_x_linearize grad_out_dot",
            grad_out.len(),
            grad_out_dot.len(),
        )?;
        self.check_len(
            "mace_vector_symmetric_contraction_vjp_x_linearize grad_x_dot",
            x.len(),
            grad_x_dot.len(),
        )?;
        let mut w1 = vec![0.0; self.u1_params];
        let mut w2 = vec![0.0; self.u2_params];
        let mut w3 = vec![0.0; self.u3_params];
        let owned_entries;
        let entries = if let Some(entries) = self.entries {
            entries
        } else {
            owned_entries = self.nonzero_entries();
            &owned_entries
        };
        let u1_entries = &entries.u1;
        let u2_entries = &entries.u2;
        let u3_entries = &entries.u3;
        if let Some(active_elements) =
            active_unit_one_hot_elements(node_attrs, num_nodes, self.num_elements)
        {
            let mut nodes_by_element = vec![Vec::new(); self.num_elements];
            for (node, &element) in active_elements.iter().enumerate() {
                nodes_by_element[element].push(node);
            }
            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() > 1
                && rayon::current_thread_index().is_none()
                && self.num_ell == 16
                && num_nodes >= 32
                && self.num_features >= 64
            {
                let features = (0..self.num_features).collect::<Vec<_>>();
                let updates_by_chunk = features
                    .par_chunks(4)
                    .map(|feature_chunk| {
                        let mut local_w1 = vec![0.0; self.u1_params];
                        let mut local_w2 = vec![0.0; self.u2_params];
                        let mut local_w3 = vec![0.0; self.u3_params];
                        let mut local_weighted_u1 = u1_entries
                            .iter()
                            .map(|entries| vec![0.0; entries.len()])
                            .collect::<Vec<_>>();
                        let mut local_weighted_u2 = u2_entries
                            .iter()
                            .map(|entries| vec![0.0; entries.len()])
                            .collect::<Vec<_>>();
                        let mut local_weighted_u3 = u3_entries
                            .iter()
                            .map(|entries| vec![0.0; entries.len()])
                            .collect::<Vec<_>>();
                        let mut updates = Vec::with_capacity(feature_chunk.len() * num_nodes);
                        for &feature in feature_chunk {
                            for (element, nodes) in nodes_by_element.iter().enumerate() {
                                if nodes.is_empty() {
                                    continue;
                                }
                                if !nodes.iter().any(|&node| {
                                    self.vector_linearized_vjp_feature_can_contribute(
                                        x_dot,
                                        grad_out,
                                        grad_out_dot,
                                        node,
                                        feature,
                                    )
                                }) {
                                    continue;
                                }
                                self.feature_weights_for_element_into(
                                    element,
                                    feature,
                                    &mut local_w1,
                                    &mut local_w2,
                                    &mut local_w3,
                                );
                                self.weight_vector_entries(
                                    &local_w1,
                                    &local_w2,
                                    &local_w3,
                                    u1_entries,
                                    u2_entries,
                                    u3_entries,
                                    &mut local_weighted_u1,
                                    &mut local_weighted_u2,
                                    &mut local_weighted_u3,
                                );
                                for &node in nodes {
                                    if let Some(grad_values) = self
                                        .vjp_node_feature_linearize_grad_weighted_entries_ell16(
                                            x,
                                            x_dot,
                                            grad_out,
                                            grad_out_dot,
                                            node,
                                            feature,
                                            u1_entries,
                                            &local_weighted_u1,
                                            u2_entries,
                                            &local_weighted_u2,
                                            u3_entries,
                                            &local_weighted_u3,
                                        )
                                    {
                                        updates.push((self.x_index(node, feature, 0), grad_values));
                                    }
                                }
                            }
                        }
                        updates
                    })
                    .collect::<Vec<_>>();
                for updates in updates_by_chunk {
                    for (x_base, grad_values) in updates {
                        add_ell16_grad_to_x(grad_x_dot, x_base, &grad_values);
                    }
                }
                return Ok(());
            }
            let mut weighted_u1 = u1_entries
                .iter()
                .map(|entries| vec![0.0; entries.len()])
                .collect::<Vec<_>>();
            let mut weighted_u2 = u2_entries
                .iter()
                .map(|entries| vec![0.0; entries.len()])
                .collect::<Vec<_>>();
            let mut weighted_u3 = u3_entries
                .iter()
                .map(|entries| vec![0.0; entries.len()])
                .collect::<Vec<_>>();
            for feature in 0..self.num_features {
                for (element, nodes) in nodes_by_element.iter().enumerate() {
                    if nodes.is_empty() {
                        continue;
                    }
                    if !nodes.iter().any(|&node| {
                        self.vector_linearized_vjp_feature_can_contribute(
                            x_dot,
                            grad_out,
                            grad_out_dot,
                            node,
                            feature,
                        )
                    }) {
                        continue;
                    }
                    self.feature_weights_for_element_into(
                        element, feature, &mut w1, &mut w2, &mut w3,
                    );
                    self.weight_vector_entries(
                        &w1,
                        &w2,
                        &w3,
                        u1_entries,
                        u2_entries,
                        u3_entries,
                        &mut weighted_u1,
                        &mut weighted_u2,
                        &mut weighted_u3,
                    );
                    for &node in nodes {
                        self.vjp_node_feature_linearize_with_weighted_entries(
                            x,
                            x_dot,
                            grad_out,
                            grad_out_dot,
                            grad_x_dot,
                            node,
                            feature,
                            u1_entries,
                            &weighted_u1,
                            u2_entries,
                            &weighted_u2,
                            u3_entries,
                            &weighted_u3,
                        );
                    }
                }
            }
            return Ok(());
        }
        for node in 0..num_nodes {
            for feature in 0..self.num_features {
                if !self.vector_linearized_vjp_feature_can_contribute(
                    x_dot,
                    grad_out,
                    grad_out_dot,
                    node,
                    feature,
                ) {
                    continue;
                }
                self.feature_weights_into(node_attrs, node, feature, &mut w1, &mut w2, &mut w3);
                self.vjp_node_feature_linearize_with_weights(
                    x,
                    x_dot,
                    grad_out,
                    grad_out_dot,
                    grad_x_dot,
                    node,
                    feature,
                    &w1,
                    &w2,
                    &w3,
                    u1_entries,
                    u2_entries,
                    u3_entries,
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_node_feature(
        &self,
        x: &[f64],
        node_attrs: &[f64],
        out: &mut [f64],
        node: usize,
        feature: usize,
        w1: &mut [f64],
        w2: &mut [f64],
        w3: &mut [f64],
        u1_entries: &[Vec<VectorU1Entry>],
        u2_entries: &[Vec<VectorU2Entry>],
        u3_entries: &[Vec<VectorU3Entry>],
    ) {
        self.feature_weights_into(node_attrs, node, feature, w1, w2, w3);
        self.forward_node_feature_with_weights(
            x, out, node, feature, w1, w2, w3, u1_entries, u2_entries, u3_entries,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn forward_node_feature_with_weights(
        &self,
        x: &[f64],
        out: &mut [f64],
        node: usize,
        feature: usize,
        w1: &[f64],
        w2: &[f64],
        w3: &[f64],
        u1_entries: &[Vec<VectorU1Entry>],
        u2_entries: &[Vec<VectorU2Entry>],
        u3_entries: &[Vec<VectorU3Entry>],
    ) {
        let output_row_len = self.num_features * self.num_equivariance;
        let out_row = &mut out[node * output_row_len..(node + 1) * output_row_len];
        self.forward_node_feature_with_weights_into_row(
            x, out_row, node, feature, w1, w2, w3, u1_entries, u2_entries, u3_entries,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn forward_node_feature_with_weights_into_row(
        &self,
        x: &[f64],
        out_row: &mut [f64],
        node: usize,
        feature: usize,
        w1: &[f64],
        w2: &[f64],
        w3: &[f64],
        u1_entries: &[Vec<VectorU1Entry>],
        u2_entries: &[Vec<VectorU2Entry>],
        u3_entries: &[Vec<VectorU3Entry>],
    ) {
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];
        for equiv in 0..self.num_equivariance {
            let mut output = 0.0;
            let mut comp = 0.0;
            for entry in &u1_entries[equiv] {
                kahan_add(
                    &mut output,
                    &mut comp,
                    entry.value * w1[entry.param] * x_row[entry.ell],
                );
            }
            for entry in &u2_entries[equiv] {
                kahan_add(
                    &mut output,
                    &mut comp,
                    entry.value * w2[entry.param] * x_row[entry.left] * x_row[entry.right],
                );
            }
            for entry in &u3_entries[equiv] {
                kahan_add(
                    &mut output,
                    &mut comp,
                    entry.value
                        * w3[entry.param]
                        * x_row[entry.left]
                        * x_row[entry.right]
                        * x_row[entry.inner],
                );
            }
            out_row[feature * self.num_equivariance + equiv] = output;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn forward_node_feature_with_weighted_entries(
        &self,
        x: &[f64],
        out: &mut [f64],
        node: usize,
        feature: usize,
        u1_entries: &[Vec<VectorU1Entry>],
        weighted_u1: &[Vec<f64>],
        u2_entries: &[Vec<VectorU2Entry>],
        weighted_u2: &[Vec<f64>],
        u3_entries: &[Vec<VectorU3Entry>],
        weighted_u3: &[Vec<f64>],
    ) {
        let output_row_len = self.num_features * self.num_equivariance;
        let out_row = &mut out[node * output_row_len..(node + 1) * output_row_len];
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];
        for equiv in 0..self.num_equivariance {
            let mut output = 0.0;
            let mut comp = 0.0;
            for (entry, &weighted_value) in u1_entries[equiv].iter().zip(&weighted_u1[equiv]) {
                kahan_add(&mut output, &mut comp, weighted_value * x_row[entry.ell]);
            }
            for (entry, &weighted_value) in u2_entries[equiv].iter().zip(&weighted_u2[equiv]) {
                kahan_add(
                    &mut output,
                    &mut comp,
                    weighted_value * x_row[entry.left] * x_row[entry.right],
                );
            }
            for (entry, &weighted_value) in u3_entries[equiv].iter().zip(&weighted_u3[equiv]) {
                kahan_add(
                    &mut output,
                    &mut comp,
                    weighted_value * x_row[entry.left] * x_row[entry.right] * x_row[entry.inner],
                );
            }
            out_row[feature * self.num_equivariance + equiv] = output;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn jvp_node_feature_with_weights(
        &self,
        x: &[f64],
        x_dot: &[f64],
        out_dot: &mut [f64],
        node: usize,
        feature: usize,
        w1: &[f64],
        w2: &[f64],
        w3: &[f64],
        u1_entries: &[Vec<VectorU1Entry>],
        u2_entries: &[Vec<VectorU2Entry>],
        u3_entries: &[Vec<VectorU3Entry>],
    ) {
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];
        let x_dot_row = &x_dot[x_base..x_base + self.num_ell];
        let output_row_len = self.num_features * self.num_equivariance;
        let out_row = &mut out_dot[node * output_row_len..(node + 1) * output_row_len];
        for equiv in 0..self.num_equivariance {
            let mut output = 0.0;
            let mut comp = 0.0;
            for entry in &u1_entries[equiv] {
                kahan_add(
                    &mut output,
                    &mut comp,
                    entry.value * w1[entry.param] * x_dot_row[entry.ell],
                );
            }
            for entry in &u2_entries[equiv] {
                kahan_add(
                    &mut output,
                    &mut comp,
                    entry.value
                        * w2[entry.param]
                        * (x_dot_row[entry.left] * x_row[entry.right]
                            + x_row[entry.left] * x_dot_row[entry.right]),
                );
            }
            for entry in &u3_entries[equiv] {
                kahan_add(
                    &mut output,
                    &mut comp,
                    entry.value
                        * w3[entry.param]
                        * (x_dot_row[entry.left] * x_row[entry.right] * x_row[entry.inner]
                            + x_row[entry.left] * x_dot_row[entry.right] * x_row[entry.inner]
                            + x_row[entry.left] * x_row[entry.right] * x_dot_row[entry.inner]),
                );
            }
            out_row[feature * self.num_equivariance + equiv] = output;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn jvp_node_feature_with_weighted_entries(
        &self,
        x: &[f64],
        x_dot: &[f64],
        out_dot: &mut [f64],
        node: usize,
        feature: usize,
        u1_entries: &[Vec<VectorU1Entry>],
        weighted_u1: &[Vec<f64>],
        u2_entries: &[Vec<VectorU2Entry>],
        weighted_u2: &[Vec<f64>],
        u3_entries: &[Vec<VectorU3Entry>],
        weighted_u3: &[Vec<f64>],
    ) {
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];
        let x_dot_row = &x_dot[x_base..x_base + self.num_ell];
        let output_row_len = self.num_features * self.num_equivariance;
        let out_row = &mut out_dot[node * output_row_len..(node + 1) * output_row_len];
        for equiv in 0..self.num_equivariance {
            let mut output = 0.0;
            let mut comp = 0.0;
            for (entry, &weighted_value) in u1_entries[equiv].iter().zip(&weighted_u1[equiv]) {
                kahan_add(
                    &mut output,
                    &mut comp,
                    weighted_value * x_dot_row[entry.ell],
                );
            }
            for (entry, &weighted_value) in u2_entries[equiv].iter().zip(&weighted_u2[equiv]) {
                kahan_add(
                    &mut output,
                    &mut comp,
                    weighted_value
                        * (x_dot_row[entry.left] * x_row[entry.right]
                            + x_row[entry.left] * x_dot_row[entry.right]),
                );
            }
            for (entry, &weighted_value) in u3_entries[equiv].iter().zip(&weighted_u3[equiv]) {
                kahan_add(
                    &mut output,
                    &mut comp,
                    weighted_value
                        * (x_dot_row[entry.left] * x_row[entry.right] * x_row[entry.inner]
                            + x_row[entry.left] * x_dot_row[entry.right] * x_row[entry.inner]
                            + x_row[entry.left] * x_row[entry.right] * x_dot_row[entry.inner]),
                );
            }
            out_row[feature * self.num_equivariance + equiv] = output;
        }
    }

    #[inline(always)]
    fn vector_linearized_vjp_feature_can_contribute(
        &self,
        x_dot: &[f64],
        grad_out: &[f64],
        grad_out_dot: &[f64],
        node: usize,
        feature: usize,
    ) -> bool {
        let out_base = self.out_index(node, feature, 0);
        let has_grad = grad_out[out_base..out_base + self.num_equivariance]
            .iter()
            .any(|&value| value != 0.0);
        let has_grad_dot = grad_out_dot[out_base..out_base + self.num_equivariance]
            .iter()
            .any(|&value| value != 0.0);
        if has_grad_dot {
            return true;
        }
        if !has_grad {
            return false;
        }
        let x_base = self.x_index(node, feature, 0);
        !all_values_zero(&x_dot[x_base..x_base + self.num_ell])
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn vjp_node_feature_linearize_grad_weighted_entries_ell16(
        &self,
        x: &[f64],
        x_dot: &[f64],
        grad_out: &[f64],
        grad_out_dot: &[f64],
        node: usize,
        feature: usize,
        u1_entries: &[Vec<VectorU1Entry>],
        weighted_u1: &[Vec<f64>],
        u2_entries: &[Vec<VectorU2Entry>],
        weighted_u2: &[Vec<f64>],
        u3_entries: &[Vec<VectorU3Entry>],
        weighted_u3: &[Vec<f64>],
    ) -> Option<[f64; 16]> {
        let out_base = self.out_index(node, feature, 0);
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + 16];
        let x_dot_row = &x_dot[x_base..x_base + 16];
        let has_x_dot = !all_values_zero(x_dot_row);
        let mut grad_values = [0.0; 16];
        let mut contributed = false;
        for equiv in 0..self.num_equivariance {
            let bar_output = grad_out[out_base + equiv];
            let bar_output_dot = grad_out_dot[out_base + equiv];
            if bar_output_dot == 0.0 && (bar_output == 0.0 || !has_x_dot) {
                continue;
            }
            contributed = true;
            for (entry, &weighted_value) in u1_entries[equiv].iter().zip(&weighted_u1[equiv]) {
                grad_values[entry.ell] += weighted_value * bar_output_dot;
            }
            for (entry, &weighted_value) in u2_entries[equiv].iter().zip(&weighted_u2[equiv]) {
                grad_values[entry.left] += weighted_value
                    * (bar_output_dot * x_row[entry.right] + bar_output * x_dot_row[entry.right]);
                grad_values[entry.right] += weighted_value
                    * (bar_output_dot * x_row[entry.left] + bar_output * x_dot_row[entry.left]);
            }
            for (entry, &weighted_value) in u3_entries[equiv].iter().zip(&weighted_u3[equiv]) {
                let x_left = x_row[entry.left];
                let x_right = x_row[entry.right];
                let x_inner = x_row[entry.inner];
                let x_dot_left = x_dot_row[entry.left];
                let x_dot_right = x_dot_row[entry.right];
                let x_dot_inner = x_dot_row[entry.inner];
                grad_values[entry.left] += weighted_value
                    * (bar_output_dot * x_right * x_inner
                        + bar_output * (x_dot_right * x_inner + x_right * x_dot_inner));
                grad_values[entry.right] += weighted_value
                    * (bar_output_dot * x_left * x_inner
                        + bar_output * (x_dot_left * x_inner + x_left * x_dot_inner));
                grad_values[entry.inner] += weighted_value
                    * (bar_output_dot * x_left * x_right
                        + bar_output * (x_dot_left * x_right + x_left * x_dot_right));
            }
        }
        contributed.then_some(grad_values)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vjp_node_feature_linearize_with_weights(
        &self,
        x: &[f64],
        x_dot: &[f64],
        grad_out: &[f64],
        grad_out_dot: &[f64],
        grad_x_dot: &mut [f64],
        node: usize,
        feature: usize,
        w1: &[f64],
        w2: &[f64],
        w3: &[f64],
        u1_entries: &[Vec<VectorU1Entry>],
        u2_entries: &[Vec<VectorU2Entry>],
        u3_entries: &[Vec<VectorU3Entry>],
    ) {
        let out_base = self.out_index(node, feature, 0);
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];
        let x_dot_row = &x_dot[x_base..x_base + self.num_ell];
        let grad_x_dot_row = &mut grad_x_dot[x_base..x_base + self.num_ell];
        for equiv in 0..self.num_equivariance {
            let bar_output = grad_out[out_base + equiv];
            let bar_output_dot = grad_out_dot[out_base + equiv];
            if bar_output == 0.0 && bar_output_dot == 0.0 {
                continue;
            }
            for entry in &u1_entries[equiv] {
                accumulate_monomial_vjp_linearize(
                    &[entry.ell],
                    entry.value * w1[entry.param],
                    bar_output,
                    bar_output_dot,
                    x_row,
                    x_dot_row,
                    grad_x_dot_row,
                );
            }
            for entry in &u2_entries[equiv] {
                accumulate_monomial_vjp_linearize(
                    &[entry.left, entry.right],
                    entry.value * w2[entry.param],
                    bar_output,
                    bar_output_dot,
                    x_row,
                    x_dot_row,
                    grad_x_dot_row,
                );
            }
            for entry in &u3_entries[equiv] {
                accumulate_monomial_vjp_linearize(
                    &[entry.left, entry.right, entry.inner],
                    entry.value * w3[entry.param],
                    bar_output,
                    bar_output_dot,
                    x_row,
                    x_dot_row,
                    grad_x_dot_row,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vjp_node_feature_linearize_with_weighted_entries(
        &self,
        x: &[f64],
        x_dot: &[f64],
        grad_out: &[f64],
        grad_out_dot: &[f64],
        grad_x_dot: &mut [f64],
        node: usize,
        feature: usize,
        u1_entries: &[Vec<VectorU1Entry>],
        weighted_u1: &[Vec<f64>],
        u2_entries: &[Vec<VectorU2Entry>],
        weighted_u2: &[Vec<f64>],
        u3_entries: &[Vec<VectorU3Entry>],
        weighted_u3: &[Vec<f64>],
    ) {
        let out_base = self.out_index(node, feature, 0);
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];
        let x_dot_row = &x_dot[x_base..x_base + self.num_ell];
        let grad_x_dot_row = &mut grad_x_dot[x_base..x_base + self.num_ell];
        for equiv in 0..self.num_equivariance {
            let bar_output = grad_out[out_base + equiv];
            let bar_output_dot = grad_out_dot[out_base + equiv];
            if bar_output == 0.0 && bar_output_dot == 0.0 {
                continue;
            }
            for (entry, &weighted_value) in u1_entries[equiv].iter().zip(&weighted_u1[equiv]) {
                accumulate_monomial_vjp_linearize(
                    &[entry.ell],
                    weighted_value,
                    bar_output,
                    bar_output_dot,
                    x_row,
                    x_dot_row,
                    grad_x_dot_row,
                );
            }
            for (entry, &weighted_value) in u2_entries[equiv].iter().zip(&weighted_u2[equiv]) {
                accumulate_monomial_vjp_linearize(
                    &[entry.left, entry.right],
                    weighted_value,
                    bar_output,
                    bar_output_dot,
                    x_row,
                    x_dot_row,
                    grad_x_dot_row,
                );
            }
            for (entry, &weighted_value) in u3_entries[equiv].iter().zip(&weighted_u3[equiv]) {
                accumulate_monomial_vjp_linearize(
                    &[entry.left, entry.right, entry.inner],
                    weighted_value,
                    bar_output,
                    bar_output_dot,
                    x_row,
                    x_dot_row,
                    grad_x_dot_row,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn vjp_node_feature(
        &self,
        x: &[f64],
        node_attrs: &[f64],
        grad_out: &[f64],
        grad_x: &mut [f64],
        node: usize,
        feature: usize,
        w1: &mut [f64],
        w2: &mut [f64],
        w3: &mut [f64],
        grad_values: &mut [f64],
        u1_entries: &[Vec<VectorU1Entry>],
        u2_entries: &[Vec<VectorU2Entry>],
        u3_entries: &[Vec<VectorU3Entry>],
    ) {
        self.feature_weights_into(node_attrs, node, feature, w1, w2, w3);
        self.vjp_node_feature_with_weights(
            x,
            grad_out,
            grad_x,
            node,
            feature,
            w1,
            w2,
            w3,
            grad_values,
            u1_entries,
            u2_entries,
            u3_entries,
        );
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vjp_node_feature_with_weights(
        &self,
        x: &[f64],
        grad_out: &[f64],
        grad_x: &mut [f64],
        node: usize,
        feature: usize,
        w1: &[f64],
        w2: &[f64],
        w3: &[f64],
        grad_values: &mut [f64],
        u1_entries: &[Vec<VectorU1Entry>],
        u2_entries: &[Vec<VectorU2Entry>],
        u3_entries: &[Vec<VectorU3Entry>],
    ) {
        let out_base = self.out_index(node, feature, 0);
        let grad_out_row = &grad_out[out_base..out_base + self.num_equivariance];
        if grad_out_row.iter().all(|&value| value == 0.0) {
            return;
        }
        debug_assert_eq!(grad_values.len(), self.num_ell);
        grad_values.fill(0.0);
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];
        for (equiv, &bar_output) in grad_out_row.iter().enumerate() {
            if bar_output == 0.0 {
                continue;
            }
            for entry in &u1_entries[equiv] {
                grad_values[entry.ell] += bar_output * entry.value * w1[entry.param];
            }
            for entry in &u2_entries[equiv] {
                let derivative = bar_output * entry.value * w2[entry.param];
                grad_values[entry.left] += derivative * x_row[entry.right];
                grad_values[entry.right] += derivative * x_row[entry.left];
            }
            for entry in &u3_entries[equiv] {
                let derivative = bar_output * entry.value * w3[entry.param];
                let x_left = x_row[entry.left];
                let x_right = x_row[entry.right];
                let x_inner = x_row[entry.inner];
                if entry.left == entry.right {
                    if entry.right == entry.inner {
                        grad_values[entry.left] += 3.0 * derivative * x_left * x_left;
                    } else {
                        grad_values[entry.left] += 2.0 * derivative * x_left * x_inner;
                        grad_values[entry.inner] += derivative * x_left * x_left;
                    }
                } else if entry.right == entry.inner {
                    grad_values[entry.left] += derivative * x_right * x_right;
                    grad_values[entry.right] += 2.0 * derivative * x_left * x_right;
                } else {
                    grad_values[entry.left] += derivative * x_right * x_inner;
                    grad_values[entry.right] += derivative * x_left * x_inner;
                    grad_values[entry.inner] += derivative * x_left * x_right;
                }
            }
        }
        for (target, &value) in grad_x[x_base..x_base + self.num_ell]
            .iter_mut()
            .zip(grad_values.iter())
        {
            *target += value;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vjp_node_feature_grad_weighted_entries_ell4_equiv3(
        &self,
        x: &[f64],
        grad_out: &[f64],
        node: usize,
        feature: usize,
        u1_entries: &[Vec<VectorU1Entry>],
        weighted_u1: &[Vec<f64>],
        u2_entries: &[Vec<VectorU2Entry>],
        weighted_u2: &[Vec<f64>],
        u3_entries: &[Vec<VectorU3Entry>],
        weighted_u3: &[Vec<f64>],
    ) -> Option<[f64; 4]> {
        let out_base = self.out_index(node, feature, 0);
        let bar0 = grad_out[out_base];
        let bar1 = grad_out[out_base + 1];
        let bar2 = grad_out[out_base + 2];
        if bar0 == 0.0 && bar1 == 0.0 && bar2 == 0.0 {
            return None;
        }
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + 4];
        let mut grad_values = [0.0; 4];
        accumulate_vector_weighted_entries_ell4(
            bar0,
            x_row,
            &u1_entries[0],
            &weighted_u1[0],
            &u2_entries[0],
            &weighted_u2[0],
            &u3_entries[0],
            &weighted_u3[0],
            &mut grad_values,
        );
        accumulate_vector_weighted_entries_ell4(
            bar1,
            x_row,
            &u1_entries[1],
            &weighted_u1[1],
            &u2_entries[1],
            &weighted_u2[1],
            &u3_entries[1],
            &weighted_u3[1],
            &mut grad_values,
        );
        accumulate_vector_weighted_entries_ell4(
            bar2,
            x_row,
            &u1_entries[2],
            &weighted_u1[2],
            &u2_entries[2],
            &weighted_u2[2],
            &u3_entries[2],
            &weighted_u3[2],
            &mut grad_values,
        );
        Some(grad_values)
    }

    #[inline(always)]
    fn add_ell4_grad_to_x(
        &self,
        grad_x: &mut [f64],
        node: usize,
        feature: usize,
        grad_values: &[f64; 4],
    ) {
        let x_base = self.x_index(node, feature, 0);
        grad_x[x_base] += grad_values[0];
        grad_x[x_base + 1] += grad_values[1];
        grad_x[x_base + 2] += grad_values[2];
        grad_x[x_base + 3] += grad_values[3];
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vjp_node_feature_grad_weighted_entries_ell16(
        &self,
        x: &[f64],
        grad_out: &[f64],
        node: usize,
        feature: usize,
        u1_entries: &[Vec<VectorU1Entry>],
        weighted_u1: &[Vec<f64>],
        u2_entries: &[Vec<VectorU2Entry>],
        weighted_u2: &[Vec<f64>],
        u3_entries: &[Vec<VectorU3Entry>],
        weighted_u3: &[Vec<f64>],
    ) -> Option<[f64; 16]> {
        let out_base = self.out_index(node, feature, 0);
        let grad_out_row = &grad_out[out_base..out_base + self.num_equivariance];
        if grad_out_row.iter().all(|&value| value == 0.0) {
            return None;
        }
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + 16];
        let mut grad_values = [0.0; 16];
        for (equiv, &bar_output) in grad_out_row.iter().enumerate() {
            accumulate_vector_weighted_entries_fixed(
                bar_output,
                x_row,
                &u1_entries[equiv],
                &weighted_u1[equiv],
                &u2_entries[equiv],
                &weighted_u2[equiv],
                &u3_entries[equiv],
                &weighted_u3[equiv],
                &mut grad_values,
            );
        }
        Some(grad_values)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vjp_node_feature_with_weighted_entries_ell4_equiv3(
        &self,
        x: &[f64],
        grad_out: &[f64],
        grad_x: &mut [f64],
        node: usize,
        feature: usize,
        grad_values: &mut [f64; 4],
        u1_entries: &[Vec<VectorU1Entry>],
        weighted_u1: &[Vec<f64>],
        u2_entries: &[Vec<VectorU2Entry>],
        weighted_u2: &[Vec<f64>],
        u3_entries: &[Vec<VectorU3Entry>],
        weighted_u3: &[Vec<f64>],
    ) {
        if let Some(values) = self.vjp_node_feature_grad_weighted_entries_ell4_equiv3(
            x,
            grad_out,
            node,
            feature,
            u1_entries,
            weighted_u1,
            u2_entries,
            weighted_u2,
            u3_entries,
            weighted_u3,
        ) {
            *grad_values = values;
            self.add_ell4_grad_to_x(grad_x, node, feature, grad_values);
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn weight_vector_entries(
        &self,
        w1: &[f64],
        w2: &[f64],
        w3: &[f64],
        u1_entries: &[Vec<VectorU1Entry>],
        u2_entries: &[Vec<VectorU2Entry>],
        u3_entries: &[Vec<VectorU3Entry>],
        weighted_u1: &mut [Vec<f64>],
        weighted_u2: &mut [Vec<f64>],
        weighted_u3: &mut [Vec<f64>],
    ) {
        for (entries, values) in u1_entries.iter().zip(weighted_u1.iter_mut()) {
            for (entry, value) in entries.iter().zip(values.iter_mut()) {
                *value = entry.value * w1[entry.param];
            }
        }
        for (entries, values) in u2_entries.iter().zip(weighted_u2.iter_mut()) {
            for (entry, value) in entries.iter().zip(values.iter_mut()) {
                *value = entry.value * w2[entry.param];
            }
        }
        for (entries, values) in u3_entries.iter().zip(weighted_u3.iter_mut()) {
            for (entry, value) in entries.iter().zip(values.iter_mut()) {
                *value = entry.value * w3[entry.param];
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vjp_node_feature_with_weighted_entries(
        &self,
        x: &[f64],
        grad_out: &[f64],
        grad_x: &mut [f64],
        node: usize,
        feature: usize,
        grad_values: &mut [f64],
        u1_entries: &[Vec<VectorU1Entry>],
        weighted_u1: &[Vec<f64>],
        u2_entries: &[Vec<VectorU2Entry>],
        weighted_u2: &[Vec<f64>],
        u3_entries: &[Vec<VectorU3Entry>],
        weighted_u3: &[Vec<f64>],
    ) {
        let out_base = self.out_index(node, feature, 0);
        let grad_out_row = &grad_out[out_base..out_base + self.num_equivariance];
        if grad_out_row.iter().all(|&value| value == 0.0) {
            return;
        }
        debug_assert_eq!(grad_values.len(), self.num_ell);
        grad_values.fill(0.0);
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];
        for (equiv, &bar_output) in grad_out_row.iter().enumerate() {
            if bar_output == 0.0 {
                continue;
            }
            for (entry, &weighted_value) in u1_entries[equiv].iter().zip(&weighted_u1[equiv]) {
                grad_values[entry.ell] += bar_output * weighted_value;
            }
            for (entry, &weighted_value) in u2_entries[equiv].iter().zip(&weighted_u2[equiv]) {
                let derivative = bar_output * weighted_value;
                grad_values[entry.left] += derivative * x_row[entry.right];
                grad_values[entry.right] += derivative * x_row[entry.left];
            }
            for (entry, &weighted_value) in u3_entries[equiv].iter().zip(&weighted_u3[equiv]) {
                let derivative = bar_output * weighted_value;
                let x_left = x_row[entry.left];
                let x_right = x_row[entry.right];
                let x_inner = x_row[entry.inner];
                if entry.left == entry.right {
                    if entry.right == entry.inner {
                        grad_values[entry.left] += 3.0 * derivative * x_left * x_left;
                    } else {
                        grad_values[entry.left] += 2.0 * derivative * x_left * x_inner;
                        grad_values[entry.inner] += derivative * x_left * x_left;
                    }
                } else if entry.right == entry.inner {
                    grad_values[entry.left] += derivative * x_right * x_right;
                    grad_values[entry.right] += 2.0 * derivative * x_left * x_right;
                } else {
                    grad_values[entry.left] += derivative * x_right * x_inner;
                    grad_values[entry.right] += derivative * x_left * x_inner;
                    grad_values[entry.inner] += derivative * x_left * x_right;
                }
            }
        }
        for (target, &value) in grad_x[x_base..x_base + self.num_ell]
            .iter_mut()
            .zip(grad_values.iter())
        {
            *target += value;
        }
    }

    fn nonzero_entries(&self) -> MaceVectorSymmetricContractionEntries {
        MaceVectorSymmetricContractionEntries {
            u1: self.u1_nonzero_entries_by_equiv(),
            u2: self.u2_nonzero_entries_by_equiv(),
            u3: self.u3_nonzero_entries_by_equiv(),
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn u1_nonzero_entries_by_equiv(&self) -> Vec<Vec<VectorU1Entry>> {
        let mut entries = vec![Vec::new(); self.num_equivariance];
        for equiv in 0..self.num_equivariance {
            for ell in 0..self.num_ell {
                for param in 0..self.u1_params {
                    let value = self.u1[((equiv * self.num_ell + ell) * self.u1_params) + param];
                    if value != 0.0 {
                        entries[equiv].push(VectorU1Entry { ell, param, value });
                    }
                }
            }
        }
        entries
    }

    #[allow(clippy::needless_range_loop)]
    fn u2_nonzero_entries_by_equiv(&self) -> Vec<Vec<VectorU2Entry>> {
        let mut entries = vec![Vec::new(); self.num_equivariance];
        for equiv in 0..self.num_equivariance {
            for left in 0..self.num_ell {
                for right in 0..self.num_ell {
                    for param in 0..self.u2_params {
                        let value = self.u2[(((equiv * self.num_ell + left) * self.num_ell
                            + right)
                            * self.u2_params)
                            + param];
                        if value != 0.0 {
                            entries[equiv].push(VectorU2Entry {
                                left,
                                right,
                                param,
                                value,
                            });
                        }
                    }
                }
            }
        }
        entries
    }

    #[allow(clippy::needless_range_loop)]
    fn u3_nonzero_entries_by_equiv(&self) -> Vec<Vec<VectorU3Entry>> {
        let mut entries = vec![Vec::new(); self.num_equivariance];
        for equiv in 0..self.num_equivariance {
            for left in 0..self.num_ell {
                for right in 0..self.num_ell {
                    for inner in 0..self.num_ell {
                        for param in 0..self.u3_params {
                            let value = self.u3[((((equiv * self.num_ell + left) * self.num_ell
                                + right)
                                * self.num_ell
                                + inner)
                                * self.u3_params)
                                + param];
                            if value != 0.0 {
                                let (left, right, inner) = sorted_triple(left, right, inner);
                                entries[equiv].push(VectorU3Entry {
                                    left,
                                    right,
                                    inner,
                                    param,
                                    value,
                                });
                            }
                        }
                    }
                }
            }
            entries[equiv] = combine_vector_u3_entries(std::mem::take(&mut entries[equiv]));
        }
        entries
    }

    #[inline(always)]
    fn feature_weights_for_element_into(
        &self,
        element: usize,
        feature: usize,
        w1: &mut [f64],
        w2: &mut [f64],
        w3: &mut [f64],
    ) {
        debug_assert_eq!(w1.len(), self.u1_params);
        debug_assert_eq!(w2.len(), self.u2_params);
        debug_assert_eq!(w3.len(), self.u3_params);
        for (param, value) in w1.iter_mut().enumerate() {
            *value = self.weights_u1[self.weight_index(element, param, feature, self.u1_params)];
        }
        for (param, value) in w2.iter_mut().enumerate() {
            *value = self.weights_u2[self.weight_index(element, param, feature, self.u2_params)];
        }
        for (param, value) in w3.iter_mut().enumerate() {
            *value = self.weights_u3[self.weight_index(element, param, feature, self.u3_params)];
        }
    }

    fn feature_weights_into(
        &self,
        node_attrs: &[f64],
        node: usize,
        feature: usize,
        w1: &mut [f64],
        w2: &mut [f64],
        w3: &mut [f64],
    ) {
        debug_assert_eq!(w1.len(), self.u1_params);
        debug_assert_eq!(w2.len(), self.u2_params);
        debug_assert_eq!(w3.len(), self.u3_params);
        w1.fill(0.0);
        w2.fill(0.0);
        w3.fill(0.0);
        let attr_row = &node_attrs[node * self.num_elements..(node + 1) * self.num_elements];
        if let Some((element, coeff)) = active_one_hot_attr(attr_row) {
            for (param, value) in w1.iter_mut().enumerate() {
                *value = coeff
                    * self.weights_u1[self.weight_index(element, param, feature, self.u1_params)];
            }
            for (param, value) in w2.iter_mut().enumerate() {
                *value = coeff
                    * self.weights_u2[self.weight_index(element, param, feature, self.u2_params)];
            }
            for (param, value) in w3.iter_mut().enumerate() {
                *value = coeff
                    * self.weights_u3[self.weight_index(element, param, feature, self.u3_params)];
            }
            return;
        }
        for (element, &coeff) in attr_row.iter().enumerate() {
            if coeff == 0.0 {
                continue;
            }
            for (param, value) in w1.iter_mut().enumerate() {
                *value += coeff
                    * self.weights_u1[self.weight_index(element, param, feature, self.u1_params)];
            }
            for (param, value) in w2.iter_mut().enumerate() {
                *value += coeff
                    * self.weights_u2[self.weight_index(element, param, feature, self.u2_params)];
            }
            for (param, value) in w3.iter_mut().enumerate() {
                *value += coeff
                    * self.weights_u3[self.weight_index(element, param, feature, self.u3_params)];
            }
        }
    }

    fn feature_weights_for_element_into_f32(
        &self,
        element: usize,
        feature: usize,
        w1: &mut [f32],
        w2: &mut [f32],
        w3: &mut [f32],
    ) {
        debug_assert_eq!(w1.len(), self.u1_params);
        debug_assert_eq!(w2.len(), self.u2_params);
        debug_assert_eq!(w3.len(), self.u3_params);
        for (param, value) in w1.iter_mut().enumerate() {
            *value =
                self.weights_u1[self.weight_index(element, param, feature, self.u1_params)] as f32;
        }
        for (param, value) in w2.iter_mut().enumerate() {
            *value =
                self.weights_u2[self.weight_index(element, param, feature, self.u2_params)] as f32;
        }
        for (param, value) in w3.iter_mut().enumerate() {
            *value =
                self.weights_u3[self.weight_index(element, param, feature, self.u3_params)] as f32;
        }
    }

    fn feature_weights_into_f32(
        &self,
        node_attrs: &[f32],
        node: usize,
        feature: usize,
        w1: &mut [f32],
        w2: &mut [f32],
        w3: &mut [f32],
    ) {
        debug_assert_eq!(w1.len(), self.u1_params);
        debug_assert_eq!(w2.len(), self.u2_params);
        debug_assert_eq!(w3.len(), self.u3_params);
        w1.fill(0.0);
        w2.fill(0.0);
        w3.fill(0.0);
        let attr_row = &node_attrs[node * self.num_elements..(node + 1) * self.num_elements];
        if let Some((element, coeff)) = active_one_hot_attr_f32(attr_row) {
            for (param, value) in w1.iter_mut().enumerate() {
                *value = coeff
                    * self.weights_u1[self.weight_index(element, param, feature, self.u1_params)]
                        as f32;
            }
            for (param, value) in w2.iter_mut().enumerate() {
                *value = coeff
                    * self.weights_u2[self.weight_index(element, param, feature, self.u2_params)]
                        as f32;
            }
            for (param, value) in w3.iter_mut().enumerate() {
                *value = coeff
                    * self.weights_u3[self.weight_index(element, param, feature, self.u3_params)]
                        as f32;
            }
            return;
        }
        for (element, &coeff) in attr_row.iter().enumerate() {
            if coeff == 0.0 {
                continue;
            }
            for (param, value) in w1.iter_mut().enumerate() {
                *value += coeff
                    * self.weights_u1[self.weight_index(element, param, feature, self.u1_params)]
                        as f32;
            }
            for (param, value) in w2.iter_mut().enumerate() {
                *value += coeff
                    * self.weights_u2[self.weight_index(element, param, feature, self.u2_params)]
                        as f32;
            }
            for (param, value) in w3.iter_mut().enumerate() {
                *value += coeff
                    * self.weights_u3[self.weight_index(element, param, feature, self.u3_params)]
                        as f32;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn forward_node_feature_with_weights_f32(
        &self,
        x: &[f32],
        out: &mut [f32],
        node: usize,
        feature: usize,
        w1: &[f32],
        w2: &[f32],
        w3: &[f32],
        u1_entries: &[Vec<VectorU1Entry>],
        u2_entries: &[Vec<VectorU2Entry>],
        u3_entries: &[Vec<VectorU3Entry>],
    ) {
        let output_row_len = self.num_features * self.num_equivariance;
        let out_row = &mut out[node * output_row_len..(node + 1) * output_row_len];
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];
        for equiv in 0..self.num_equivariance {
            let mut output = 0.0_f32;
            for entry in &u1_entries[equiv] {
                output += entry.value as f32 * w1[entry.param] * x_row[entry.ell];
            }
            for entry in &u2_entries[equiv] {
                output +=
                    entry.value as f32 * w2[entry.param] * x_row[entry.left] * x_row[entry.right];
            }
            for entry in &u3_entries[equiv] {
                output += entry.value as f32
                    * w3[entry.param]
                    * x_row[entry.left]
                    * x_row[entry.right]
                    * x_row[entry.inner];
            }
            out_row[feature * self.num_equivariance + equiv] = output;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn vjp_node_feature_with_weights_f32(
        &self,
        x: &[f32],
        grad_out: &[f32],
        grad_x: &mut [f32],
        node: usize,
        feature: usize,
        w1: &[f32],
        w2: &[f32],
        w3: &[f32],
        u1_entries: &[Vec<VectorU1Entry>],
        u2_entries: &[Vec<VectorU2Entry>],
        u3_entries: &[Vec<VectorU3Entry>],
    ) {
        let out_base = self.out_index(node, feature, 0);
        let grad_out_row = &grad_out[out_base..out_base + self.num_equivariance];
        if grad_out_row.iter().all(|&value| value == 0.0) {
            return;
        }
        let x_base = self.x_index(node, feature, 0);
        let x_row = &x[x_base..x_base + self.num_ell];
        let grad_row = &mut grad_x[x_base..x_base + self.num_ell];
        for (equiv, &bar_output) in grad_out_row.iter().enumerate() {
            if bar_output == 0.0 {
                continue;
            }
            for entry in &u1_entries[equiv] {
                grad_row[entry.ell] += bar_output * entry.value as f32 * w1[entry.param];
            }
            for entry in &u2_entries[equiv] {
                let derivative = bar_output * entry.value as f32 * w2[entry.param];
                grad_row[entry.left] += derivative * x_row[entry.right];
                grad_row[entry.right] += derivative * x_row[entry.left];
            }
            for entry in &u3_entries[equiv] {
                let derivative = bar_output * entry.value as f32 * w3[entry.param];
                let x_left = x_row[entry.left];
                let x_right = x_row[entry.right];
                let x_inner = x_row[entry.inner];
                if entry.left == entry.right {
                    if entry.right == entry.inner {
                        grad_row[entry.left] += 3.0 * derivative * x_left * x_left;
                    } else {
                        grad_row[entry.left] += 2.0 * derivative * x_left * x_inner;
                        grad_row[entry.inner] += derivative * x_left * x_left;
                    }
                } else if entry.right == entry.inner {
                    grad_row[entry.left] += derivative * x_right * x_right;
                    grad_row[entry.right] += 2.0 * derivative * x_left * x_right;
                } else {
                    grad_row[entry.left] += derivative * x_right * x_inner;
                    grad_row[entry.right] += derivative * x_left * x_inner;
                    grad_row[entry.inner] += derivative * x_left * x_right;
                }
            }
        }
    }

    fn check_runtime_shapes(
        &self,
        x: &[f64],
        node_attrs: &[f64],
        out_len: usize,
        out_op: &'static str,
    ) -> Result<usize> {
        let per_node_x = self.num_features * self.num_ell;
        if !x.len().is_multiple_of(per_node_x) {
            return Err(CoreError::ShapeMismatch {
                op: "mace_vector_symmetric_contraction x",
                expected: per_node_x,
                actual: x.len(),
            });
        }
        let num_nodes = x.len() / per_node_x;
        self.check_len(
            "mace_vector_symmetric_contraction node_attrs",
            num_nodes * self.num_elements,
            node_attrs.len(),
        )?;
        self.check_len(
            out_op,
            num_nodes * self.num_features * self.num_equivariance,
            out_len,
        )?;
        Ok(num_nodes)
    }

    fn check_runtime_shapes_f32(
        &self,
        x: &[f32],
        node_attrs: &[f32],
        out_len: usize,
        out_op: &'static str,
    ) -> Result<usize> {
        let per_node_x = self.num_features * self.num_ell;
        if !x.len().is_multiple_of(per_node_x) {
            return Err(CoreError::ShapeMismatch {
                op: "mace_vector_symmetric_contraction_f32 x",
                expected: per_node_x,
                actual: x.len(),
            });
        }
        let num_nodes = x.len() / per_node_x;
        self.check_len(
            "mace_vector_symmetric_contraction_f32 node_attrs",
            num_nodes * self.num_elements,
            node_attrs.len(),
        )?;
        self.check_len(
            out_op,
            num_nodes * self.num_features * self.num_equivariance,
            out_len,
        )?;
        Ok(num_nodes)
    }

    fn check_len(&self, op: &'static str, expected: usize, actual: usize) -> Result<()> {
        if expected == actual {
            Ok(())
        } else {
            Err(CoreError::ShapeMismatch {
                op,
                expected,
                actual,
            })
        }
    }

    #[inline(always)]
    fn x_index(&self, node: usize, feature: usize, ell: usize) -> usize {
        (node * self.num_features + feature) * self.num_ell + ell
    }

    #[inline(always)]
    fn out_index(&self, node: usize, feature: usize, equiv: usize) -> usize {
        (node * self.num_features + feature) * self.num_equivariance + equiv
    }

    #[inline(always)]
    fn weight_index(&self, element: usize, param: usize, feature: usize, params: usize) -> usize {
        (element * params + param) * self.num_features + feature
    }
}

fn active_one_hot_attr(attrs: &[f64]) -> Option<(usize, f64)> {
    let mut active = None;
    for (index, &value) in attrs.iter().enumerate() {
        if value == 0.0 {
            continue;
        }
        if active.is_some() {
            return None;
        }
        active = Some((index, value));
    }
    active
}

fn active_one_hot_attr_f32(attrs: &[f32]) -> Option<(usize, f32)> {
    let mut active = None;
    for (index, &value) in attrs.iter().enumerate() {
        if value == 0.0 {
            continue;
        }
        if active.is_some() {
            return None;
        }
        active = Some((index, value));
    }
    active
}

fn active_unit_one_hot_elements_f32(
    node_attrs: &[f32],
    num_nodes: usize,
    num_elements: usize,
) -> Option<Vec<usize>> {
    let mut elements = Vec::with_capacity(num_nodes);
    for node in 0..num_nodes {
        let attr_row = &node_attrs[node * num_elements..(node + 1) * num_elements];
        let (element, coeff) = active_one_hot_attr_f32(attr_row)?;
        if coeff != 1.0 {
            return None;
        }
        elements.push(element);
    }
    Some(elements)
}

#[inline(always)]
fn all_values_zero(values: &[f64]) -> bool {
    values.iter().all(|&value| value == 0.0)
}

#[inline(always)]
fn scalar_linearized_vjp_row_can_contribute(
    bar_output: f64,
    bar_output_dot: f64,
    x_dot_row: &[f64],
) -> bool {
    bar_output_dot != 0.0 || (bar_output != 0.0 && !all_values_zero(x_dot_row))
}

#[allow(clippy::too_many_arguments)]
fn weight_scalar_entries(
    w1: &[f64],
    w2: &[f64],
    w3: &[f64],
    u1_entries: &[ScalarU1Entry],
    u2_entries: &[ScalarU2Entry],
    u3_entries: &[ScalarU3Entry],
    weighted_u1: &mut [f64],
    weighted_u2: &mut [f64],
    weighted_u3: &mut [f64],
) {
    for (entry, value) in u1_entries.iter().zip(weighted_u1.iter_mut()) {
        *value = entry.value * w1[entry.param];
    }
    for (entry, value) in u2_entries.iter().zip(weighted_u2.iter_mut()) {
        *value = entry.value * w2[entry.param];
    }
    for (entry, value) in u3_entries.iter().zip(weighted_u3.iter_mut()) {
        *value = entry.value * w3[entry.param];
    }
}

fn weight_scalar_u3_groups(w3: &[f64], u3_groups: &[ScalarU3TripleGroup], weighted_u3: &mut [f64]) {
    for (group, value) in u3_groups.iter().zip(weighted_u3.iter_mut()) {
        let mut weighted_value = 0.0;
        for &(param, coefficient) in &group.terms {
            weighted_value += coefficient * w3[param];
        }
        *value = weighted_value;
    }
}

fn weight_scalar_u2_groups(w2: &[f64], u2_groups: &[ScalarU2PairGroup], weighted_u2: &mut [f64]) {
    for (group, value) in u2_groups.iter().zip(weighted_u2.iter_mut()) {
        let mut weighted_value = 0.0;
        for &(param, coefficient) in &group.terms {
            weighted_value += coefficient * w2[param];
        }
        *value = weighted_value;
    }
}

fn weight_scalar_u3_groups_f32(
    w3: &[f64],
    u3_groups: &[ScalarU3TripleGroup],
    weighted_u3: &mut [f32],
) {
    for (group, value) in u3_groups.iter().zip(weighted_u3.iter_mut()) {
        let mut weighted_value = 0.0;
        for &(param, coefficient) in &group.terms {
            weighted_value += coefficient * w3[param];
        }
        *value = weighted_value as f32;
    }
}

fn weight_scalar_u2_groups_f32(
    w2: &[f64],
    u2_groups: &[ScalarU2PairGroup],
    weighted_u2: &mut [f32],
) {
    for (group, value) in u2_groups.iter().zip(weighted_u2.iter_mut()) {
        let mut weighted_value = 0.0;
        for &(param, coefficient) in &group.terms {
            weighted_value += coefficient * w2[param];
        }
        *value = weighted_value as f32;
    }
}

fn active_unit_one_hot_elements(
    node_attrs: &[f64],
    num_nodes: usize,
    num_elements: usize,
) -> Option<Vec<usize>> {
    let mut elements = Vec::with_capacity(num_nodes);
    for node in 0..num_nodes {
        let attr_row = &node_attrs[node * num_elements..(node + 1) * num_elements];
        let (element, coeff) = active_one_hot_attr(attr_row)?;
        if coeff != 1.0 {
            return None;
        }
        elements.push(element);
    }
    Some(elements)
}

#[cfg(feature = "parallel")]
fn active_element_dense_map(
    active_elements: &[usize],
    num_elements: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut element_to_dense = vec![usize::MAX; num_elements];
    let mut active_element_list = Vec::new();
    for &element in active_elements {
        if element_to_dense[element] == usize::MAX {
            element_to_dense[element] = active_element_list.len();
            active_element_list.push(element);
        }
    }
    (active_element_list, element_to_dense)
}

fn sorted_triple(left: usize, right: usize, inner: usize) -> (usize, usize, usize) {
    let mut values = [left, right, inner];
    values.sort_unstable();
    (values[0], values[1], values[2])
}

fn combine_scalar_u3_entries(mut entries: Vec<ScalarU3Entry>) -> Vec<ScalarU3Entry> {
    entries.sort_unstable_by_key(|entry| (entry.left, entry.right, entry.inner, entry.param));
    let mut combined: Vec<ScalarU3Entry> = Vec::with_capacity(entries.len());
    for entry in entries {
        if let Some(last) = combined.last_mut() {
            if (last.left, last.right, last.inner, last.param)
                == (entry.left, entry.right, entry.inner, entry.param)
            {
                last.value += entry.value;
                continue;
            }
        }
        combined.push(entry);
    }
    combined.retain(|entry| entry.value != 0.0);
    combined
}

fn combine_vector_u3_entries(mut entries: Vec<VectorU3Entry>) -> Vec<VectorU3Entry> {
    entries.sort_unstable_by_key(|entry| (entry.left, entry.right, entry.inner, entry.param));
    let mut combined: Vec<VectorU3Entry> = Vec::with_capacity(entries.len());
    for entry in entries {
        if let Some(last) = combined.last_mut() {
            if (last.left, last.right, last.inner, last.param)
                == (entry.left, entry.right, entry.inner, entry.param)
            {
                last.value += entry.value;
                continue;
            }
        }
        combined.push(entry);
    }
    combined.retain(|entry| entry.value != 0.0);
    combined
}

#[cfg(feature = "parallel")]
fn scalar_u1_entries_by_param(entries: &[ScalarU1Entry], params: usize) -> Vec<Vec<ScalarU1Entry>> {
    let mut by_param = vec![Vec::new(); params];
    for &entry in entries {
        by_param[entry.param].push(entry);
    }
    by_param
}

#[cfg(feature = "parallel")]
fn scalar_u2_entries_by_param(entries: &[ScalarU2Entry], params: usize) -> Vec<Vec<ScalarU2Entry>> {
    let mut by_param = vec![Vec::new(); params];
    for &entry in entries {
        by_param[entry.param].push(entry);
    }
    by_param
}

#[cfg(feature = "parallel")]
fn scalar_u3_entries_by_param(entries: &[ScalarU3Entry], params: usize) -> Vec<Vec<ScalarU3Entry>> {
    let mut by_param = vec![Vec::new(); params];
    for &entry in entries {
        by_param[entry.param].push(entry);
    }
    by_param
}

fn scalar_u2_entries_by_pair(entries: &[ScalarU2Entry]) -> Vec<ScalarU2PairGroup> {
    let mut groups: Vec<ScalarU2PairGroup> = Vec::new();
    for entry in entries {
        if let Some(last) = groups.last_mut() {
            if (last.left, last.right) == (entry.left, entry.right) {
                last.terms.push((entry.param, entry.value));
                continue;
            }
        }
        groups.push(ScalarU2PairGroup {
            left: entry.left,
            right: entry.right,
            terms: vec![(entry.param, entry.value)],
        });
    }
    groups
}

fn scalar_u3_entries_by_triple(entries: &[ScalarU3Entry]) -> Vec<ScalarU3TripleGroup> {
    let mut groups: Vec<ScalarU3TripleGroup> = Vec::new();
    for entry in entries {
        if let Some(last) = groups.last_mut() {
            if (last.left, last.right, last.inner) == (entry.left, entry.right, entry.inner) {
                last.terms.push((entry.param, entry.value));
                continue;
            }
        }
        groups.push(ScalarU3TripleGroup {
            left: entry.left,
            right: entry.right,
            inner: entry.inner,
            terms: vec![(entry.param, entry.value)],
        });
    }
    groups
}

fn vector_u2_entries_by_equiv_pair(
    entries_by_equiv: &[Vec<VectorU2Entry>],
) -> Vec<Vec<VectorU2PairGroup>> {
    entries_by_equiv
        .iter()
        .map(|entries| {
            let mut groups: Vec<VectorU2PairGroup> = Vec::new();
            for entry in entries {
                if let Some(last) = groups.last_mut() {
                    if (last.left, last.right) == (entry.left, entry.right) {
                        last.terms.push((entry.param, entry.value));
                        continue;
                    }
                }
                groups.push(VectorU2PairGroup {
                    left: entry.left,
                    right: entry.right,
                    terms: vec![(entry.param, entry.value)],
                });
            }
            groups
        })
        .collect()
}

fn vector_u3_entries_by_equiv_triple(
    entries_by_equiv: &[Vec<VectorU3Entry>],
) -> Vec<Vec<VectorU3TripleGroup>> {
    entries_by_equiv
        .iter()
        .map(|entries| {
            let mut groups: Vec<VectorU3TripleGroup> = Vec::new();
            for entry in entries {
                if let Some(last) = groups.last_mut() {
                    if (last.left, last.right, last.inner) == (entry.left, entry.right, entry.inner)
                    {
                        last.terms.push((entry.param, entry.value));
                        continue;
                    }
                }
                groups.push(VectorU3TripleGroup {
                    left: entry.left,
                    right: entry.right,
                    inner: entry.inner,
                    terms: vec![(entry.param, entry.value)],
                });
            }
            groups
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn weight_vector_monomial_groups(
    w1: &[f64],
    w2: &[f64],
    w3: &[f64],
    u1_entries: &[Vec<VectorU1Entry>],
    u2_groups: &[Vec<VectorU2PairGroup>],
    u3_groups: &[Vec<VectorU3TripleGroup>],
    weighted_u1: &mut [Vec<VectorU1WeightedEntry>],
    weighted_u2: &mut [Vec<VectorU2WeightedEntry>],
    weighted_u3: &mut [Vec<VectorU3WeightedEntry>],
) {
    for (entries, values) in u1_entries.iter().zip(weighted_u1.iter_mut()) {
        values.clear();
        values.reserve(entries.len());
        for entry in entries {
            let value = entry.value * w1[entry.param];
            if value != 0.0 {
                values.push(VectorU1WeightedEntry {
                    ell: entry.ell,
                    value,
                });
            }
        }
    }
    for (groups, values) in u2_groups.iter().zip(weighted_u2.iter_mut()) {
        values.clear();
        values.reserve(groups.len());
        for group in groups {
            let mut value = 0.0;
            for &(param, coefficient) in &group.terms {
                value += coefficient * w2[param];
            }
            if value != 0.0 {
                values.push(VectorU2WeightedEntry {
                    left: group.left,
                    right: group.right,
                    value,
                });
            }
        }
    }
    for (groups, values) in u3_groups.iter().zip(weighted_u3.iter_mut()) {
        values.clear();
        values.reserve(groups.len());
        for group in groups {
            let mut value = 0.0;
            for &(param, coefficient) in &group.terms {
                value += coefficient * w3[param];
            }
            if value != 0.0 {
                values.push(VectorU3WeightedEntry {
                    left: group.left,
                    right: group.right,
                    inner: group.inner,
                    value,
                });
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn accumulate_vector_weighted_entries_ell4(
    bar_output: f64,
    x_row: &[f64],
    u1_entries: &[VectorU1Entry],
    weighted_u1: &[f64],
    u2_entries: &[VectorU2Entry],
    weighted_u2: &[f64],
    u3_entries: &[VectorU3Entry],
    weighted_u3: &[f64],
    grad_values: &mut [f64; 4],
) {
    if bar_output == 0.0 {
        return;
    }
    debug_assert_eq!(x_row.len(), 4);
    for (entry, &weighted_value) in u1_entries.iter().zip(weighted_u1) {
        grad_values[entry.ell] += bar_output * weighted_value;
    }
    for (entry, &weighted_value) in u2_entries.iter().zip(weighted_u2) {
        let derivative = bar_output * weighted_value;
        grad_values[entry.left] += derivative * x_row[entry.right];
        grad_values[entry.right] += derivative * x_row[entry.left];
    }
    for (entry, &weighted_value) in u3_entries.iter().zip(weighted_u3) {
        let derivative = bar_output * weighted_value;
        let x_left = x_row[entry.left];
        let x_right = x_row[entry.right];
        let x_inner = x_row[entry.inner];
        if entry.left == entry.right {
            if entry.right == entry.inner {
                grad_values[entry.left] += 3.0 * derivative * x_left * x_left;
            } else {
                grad_values[entry.left] += 2.0 * derivative * x_left * x_inner;
                grad_values[entry.inner] += derivative * x_left * x_left;
            }
        } else if entry.right == entry.inner {
            grad_values[entry.left] += derivative * x_right * x_right;
            grad_values[entry.right] += 2.0 * derivative * x_left * x_right;
        } else {
            grad_values[entry.left] += derivative * x_right * x_inner;
            grad_values[entry.right] += derivative * x_left * x_inner;
            grad_values[entry.inner] += derivative * x_left * x_right;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "parallel")]
#[inline(always)]
fn accumulate_scalar_grouped_entries_ell16(
    bar_output: f64,
    x_row: &[f64],
    u1_by_param: &[Vec<ScalarU1Entry>],
    w1: &[f64],
    u2_by_param: &[Vec<ScalarU2Entry>],
    w2: &[f64],
    u3_by_param: &[Vec<ScalarU3Entry>],
    w3: &[f64],
    grad_values: &mut [f64; 16],
) {
    debug_assert_ne!(bar_output, 0.0);
    debug_assert_eq!(x_row.len(), 16);
    for (param, entries) in u1_by_param.iter().enumerate() {
        let scale = bar_output * w1[param];
        if scale == 0.0 {
            continue;
        }
        for entry in entries {
            grad_values[entry.ell] += scale * entry.value;
        }
    }
    for (param, entries) in u2_by_param.iter().enumerate() {
        let scale = bar_output * w2[param];
        if scale == 0.0 {
            continue;
        }
        for entry in entries {
            let derivative = scale * entry.value;
            grad_values[entry.left] += derivative * x_row[entry.right];
            grad_values[entry.right] += derivative * x_row[entry.left];
        }
    }
    for (param, entries) in u3_by_param.iter().enumerate() {
        let scale = bar_output * w3[param];
        if scale == 0.0 {
            continue;
        }
        for entry in entries {
            let derivative = scale * entry.value;
            let x_left = x_row[entry.left];
            let x_right = x_row[entry.right];
            let x_inner = x_row[entry.inner];
            if entry.left == entry.right {
                if entry.right == entry.inner {
                    grad_values[entry.left] += 3.0 * derivative * x_left * x_left;
                } else {
                    grad_values[entry.left] += 2.0 * derivative * x_left * x_inner;
                    grad_values[entry.inner] += derivative * x_left * x_left;
                }
            } else if entry.right == entry.inner {
                grad_values[entry.left] += derivative * x_right * x_right;
                grad_values[entry.right] += 2.0 * derivative * x_left * x_right;
            } else {
                grad_values[entry.left] += derivative * x_right * x_inner;
                grad_values[entry.right] += derivative * x_left * x_inner;
                grad_values[entry.inner] += derivative * x_left * x_right;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn evaluate_scalar_weighted_monomials_ell16(
    x_row: &[f64],
    u1_entries: &[ScalarU1Entry],
    weighted_u1: &[f64],
    u2_entries: &[ScalarU2Entry],
    weighted_u2: &[f64],
    u3_entries: &[ScalarU3Entry],
    weighted_u3: &[f64],
) -> f64 {
    debug_assert_eq!(x_row.len(), 16);
    let mut output = 0.0;
    for (entry, &weighted_value) in u1_entries.iter().zip(weighted_u1) {
        output += weighted_value * x_row[entry.ell];
    }
    for (entry, &weighted_value) in u2_entries.iter().zip(weighted_u2) {
        output += weighted_value * x_row[entry.left] * x_row[entry.right];
    }
    for (entry, &weighted_value) in u3_entries.iter().zip(weighted_u3) {
        output += weighted_value * x_row[entry.left] * x_row[entry.right] * x_row[entry.inner];
    }
    output
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn evaluate_scalar_weighted_monomials_ell16_f32_prepared(
    x_row: &[f32],
    u1_entries: &[ScalarU1Entry],
    weighted_u1: &[f32],
    u2_entries: &[ScalarU2WeightedEntry],
    weighted_u2: &[f32],
    u3_entries: &[ScalarU3WeightedEntry],
    weighted_u3: &[f32],
) -> f32 {
    debug_assert_eq!(x_row.len(), 16);
    let mut output = 0.0_f32;
    for (entry, &weighted_value) in u1_entries.iter().zip(weighted_u1) {
        output += weighted_value * x_row[entry.ell];
    }
    for (entry, &weighted_value) in u2_entries.iter().zip(weighted_u2) {
        output += weighted_value * x_row[entry.left] * x_row[entry.right];
    }
    for (entry, &weighted_value) in u3_entries.iter().zip(weighted_u3) {
        output += weighted_value * x_row[entry.left] * x_row[entry.right] * x_row[entry.inner];
    }
    output
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn accumulate_scalar_weighted_monomials_ell16_prepared_into(
    bar_output: f64,
    x_row: &[f64],
    u1_entries: &[ScalarU1Entry],
    weighted_u1: &[f64],
    u2_entries: &[ScalarU2WeightedEntry],
    weighted_u2: &[f64],
    u3_entries: &[ScalarU3WeightedEntry],
    weighted_u3: &[f64],
    grad_row: &mut [f64],
) {
    debug_assert_eq!(x_row.len(), 16);
    debug_assert_eq!(grad_row.len(), 16);
    for (entry, &weighted_value) in u1_entries.iter().zip(weighted_u1) {
        grad_row[entry.ell] += bar_output * weighted_value;
    }
    for (entry, &weighted_value) in u2_entries.iter().zip(weighted_u2) {
        let derivative = bar_output * weighted_value;
        grad_row[entry.left] += derivative * x_row[entry.right];
        grad_row[entry.right] += derivative * x_row[entry.left];
    }
    for (entry, &weighted_value) in u3_entries.iter().zip(weighted_u3) {
        let derivative = bar_output * weighted_value;
        let x_left = x_row[entry.left];
        let x_right = x_row[entry.right];
        let x_inner = x_row[entry.inner];
        if entry.left == entry.right {
            if entry.right == entry.inner {
                grad_row[entry.left] += 3.0 * derivative * x_left * x_left;
            } else {
                grad_row[entry.left] += 2.0 * derivative * x_left * x_inner;
                grad_row[entry.inner] += derivative * x_left * x_left;
            }
        } else if entry.right == entry.inner {
            grad_row[entry.left] += derivative * x_right * x_right;
            grad_row[entry.right] += 2.0 * derivative * x_left * x_right;
        } else {
            grad_row[entry.left] += derivative * x_right * x_inner;
            grad_row[entry.right] += derivative * x_left * x_inner;
            grad_row[entry.inner] += derivative * x_left * x_right;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn accumulate_scalar_weighted_monomials_ell16_f32_prepared_into(
    bar_output: f32,
    x_row: &[f32],
    u1_entries: &[ScalarU1Entry],
    weighted_u1: &[f32],
    u2_entries: &[ScalarU2WeightedEntry],
    weighted_u2: &[f32],
    u3_entries: &[ScalarU3WeightedEntry],
    weighted_u3: &[f32],
    grad_row: &mut [f32],
) {
    debug_assert_eq!(x_row.len(), 16);
    debug_assert_eq!(grad_row.len(), 16);
    for (entry, &weighted_value) in u1_entries.iter().zip(weighted_u1) {
        grad_row[entry.ell] += bar_output * weighted_value;
    }
    for (entry, &weighted_value) in u2_entries.iter().zip(weighted_u2) {
        let derivative = bar_output * weighted_value;
        grad_row[entry.left] += derivative * x_row[entry.right];
        grad_row[entry.right] += derivative * x_row[entry.left];
    }
    for (entry, &weighted_value) in u3_entries.iter().zip(weighted_u3) {
        let derivative = bar_output * weighted_value;
        let x_left = x_row[entry.left];
        let x_right = x_row[entry.right];
        let x_inner = x_row[entry.inner];
        if entry.left == entry.right {
            if entry.right == entry.inner {
                grad_row[entry.left] += 3.0 * derivative * x_left * x_left;
            } else {
                grad_row[entry.left] += 2.0 * derivative * x_left * x_inner;
                grad_row[entry.inner] += derivative * x_left * x_left;
            }
        } else if entry.right == entry.inner {
            grad_row[entry.left] += derivative * x_right * x_right;
            grad_row[entry.right] += 2.0 * derivative * x_left * x_right;
        } else {
            grad_row[entry.left] += derivative * x_right * x_inner;
            grad_row[entry.right] += derivative * x_left * x_inner;
            grad_row[entry.inner] += derivative * x_left * x_right;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn evaluate_vector_weighted_monomials_ell16(
    x_row: &[f64],
    u1_entries: &[VectorU1WeightedEntry],
    u2_entries: &[VectorU2WeightedEntry],
    u3_entries: &[VectorU3WeightedEntry],
) -> f64 {
    debug_assert_eq!(x_row.len(), 16);
    let mut output = 0.0;
    for entry in u1_entries {
        output += entry.value * x_row[entry.ell];
    }
    for entry in u2_entries {
        output += entry.value * x_row[entry.left] * x_row[entry.right];
    }
    for entry in u3_entries {
        output += entry.value * x_row[entry.left] * x_row[entry.right] * x_row[entry.inner];
    }
    output
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn evaluate_vector_weighted_monomials_ell16_f32(
    x_row: &[f32],
    u1_entries: &[VectorU1WeightedEntryF32],
    u2_entries: &[VectorU2WeightedEntryF32],
    u3_entries: &[VectorU3WeightedEntryF32],
) -> f32 {
    debug_assert_eq!(x_row.len(), 16);
    let mut output = 0.0_f32;
    for entry in u1_entries {
        output += entry.value * x_row[entry.ell];
    }
    for entry in u2_entries {
        output += entry.value * x_row[entry.left] * x_row[entry.right];
    }
    for entry in u3_entries {
        output += entry.value * x_row[entry.left] * x_row[entry.right] * x_row[entry.inner];
    }
    output
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn accumulate_vector_weighted_monomials_ell16(
    bar_output: f64,
    x_row: &[f64],
    u1_entries: &[VectorU1WeightedEntry],
    u2_entries: &[VectorU2WeightedEntry],
    u3_entries: &[VectorU3WeightedEntry],
    grad_values: &mut [f64; 16],
) {
    if bar_output == 0.0 {
        return;
    }
    debug_assert_eq!(x_row.len(), 16);
    for entry in u1_entries {
        grad_values[entry.ell] += bar_output * entry.value;
    }
    for entry in u2_entries {
        let derivative = bar_output * entry.value;
        grad_values[entry.left] += derivative * x_row[entry.right];
        grad_values[entry.right] += derivative * x_row[entry.left];
    }
    for entry in u3_entries {
        let derivative = bar_output * entry.value;
        let x_left = x_row[entry.left];
        let x_right = x_row[entry.right];
        let x_inner = x_row[entry.inner];
        if entry.left == entry.right {
            if entry.right == entry.inner {
                grad_values[entry.left] += 3.0 * derivative * x_left * x_left;
            } else {
                grad_values[entry.left] += 2.0 * derivative * x_left * x_inner;
                grad_values[entry.inner] += derivative * x_left * x_left;
            }
        } else if entry.right == entry.inner {
            grad_values[entry.left] += derivative * x_right * x_right;
            grad_values[entry.right] += 2.0 * derivative * x_left * x_right;
        } else {
            grad_values[entry.left] += derivative * x_right * x_inner;
            grad_values[entry.right] += derivative * x_left * x_inner;
            grad_values[entry.inner] += derivative * x_left * x_right;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn accumulate_vector_weighted_monomials_ell16_f32_into(
    bar_output: f32,
    x_row: &[f32],
    u1_entries: &[VectorU1WeightedEntryF32],
    u2_entries: &[VectorU2WeightedEntryF32],
    u3_entries: &[VectorU3WeightedEntryF32],
    grad_row: &mut [f32],
) {
    if bar_output == 0.0 {
        return;
    }
    debug_assert_eq!(x_row.len(), 16);
    debug_assert_eq!(grad_row.len(), 16);
    for entry in u1_entries {
        grad_row[entry.ell] += bar_output * entry.value;
    }
    for entry in u2_entries {
        let derivative = bar_output * entry.value;
        grad_row[entry.left] += derivative * x_row[entry.right];
        grad_row[entry.right] += derivative * x_row[entry.left];
    }
    for entry in u3_entries {
        let derivative = bar_output * entry.value;
        let x_left = x_row[entry.left];
        let x_right = x_row[entry.right];
        let x_inner = x_row[entry.inner];
        if entry.left == entry.right {
            if entry.right == entry.inner {
                grad_row[entry.left] += 3.0 * derivative * x_left * x_left;
            } else {
                grad_row[entry.left] += 2.0 * derivative * x_left * x_inner;
                grad_row[entry.inner] += derivative * x_left * x_left;
            }
        } else if entry.right == entry.inner {
            grad_row[entry.left] += derivative * x_right * x_right;
            grad_row[entry.right] += 2.0 * derivative * x_left * x_right;
        } else {
            grad_row[entry.left] += derivative * x_right * x_inner;
            grad_row[entry.right] += derivative * x_left * x_inner;
            grad_row[entry.inner] += derivative * x_left * x_right;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn accumulate_vector_weighted_entries_fixed<const N: usize>(
    bar_output: f64,
    x_row: &[f64],
    u1_entries: &[VectorU1Entry],
    weighted_u1: &[f64],
    u2_entries: &[VectorU2Entry],
    weighted_u2: &[f64],
    u3_entries: &[VectorU3Entry],
    weighted_u3: &[f64],
    grad_values: &mut [f64; N],
) {
    if bar_output == 0.0 {
        return;
    }
    debug_assert_eq!(x_row.len(), N);
    for (entry, &weighted_value) in u1_entries.iter().zip(weighted_u1) {
        grad_values[entry.ell] += bar_output * weighted_value;
    }
    for (entry, &weighted_value) in u2_entries.iter().zip(weighted_u2) {
        let derivative = bar_output * weighted_value;
        grad_values[entry.left] += derivative * x_row[entry.right];
        grad_values[entry.right] += derivative * x_row[entry.left];
    }
    for (entry, &weighted_value) in u3_entries.iter().zip(weighted_u3) {
        let derivative = bar_output * weighted_value;
        let x_left = x_row[entry.left];
        let x_right = x_row[entry.right];
        let x_inner = x_row[entry.inner];
        if entry.left == entry.right {
            if entry.right == entry.inner {
                grad_values[entry.left] += 3.0 * derivative * x_left * x_left;
            } else {
                grad_values[entry.left] += 2.0 * derivative * x_left * x_inner;
                grad_values[entry.inner] += derivative * x_left * x_left;
            }
        } else if entry.right == entry.inner {
            grad_values[entry.left] += derivative * x_right * x_right;
            grad_values[entry.right] += 2.0 * derivative * x_left * x_right;
        } else {
            grad_values[entry.left] += derivative * x_right * x_inner;
            grad_values[entry.right] += derivative * x_left * x_inner;
            grad_values[entry.inner] += derivative * x_left * x_right;
        }
    }
}

fn add_ell16_grad_to_x(grad_x: &mut [f64], x_base: usize, grad_values: &[f64; 16]) {
    let target = &mut grad_x[x_base..x_base + 16];
    add_ell16_grad_to_row(target, grad_values);
}

#[inline(always)]
fn add_ell16_grad_to_row(target: &mut [f64], grad_values: &[f64; 16]) {
    debug_assert_eq!(target.len(), 16);
    target[0] += grad_values[0];
    target[1] += grad_values[1];
    target[2] += grad_values[2];
    target[3] += grad_values[3];
    target[4] += grad_values[4];
    target[5] += grad_values[5];
    target[6] += grad_values[6];
    target[7] += grad_values[7];
    target[8] += grad_values[8];
    target[9] += grad_values[9];
    target[10] += grad_values[10];
    target[11] += grad_values[11];
    target[12] += grad_values[12];
    target[13] += grad_values[13];
    target[14] += grad_values[14];
    target[15] += grad_values[15];
}

#[inline(always)]
fn add_ell16_grad_to_row_f32(target: &mut [f32], grad_values: &[f32; 16]) {
    debug_assert_eq!(target.len(), 16);
    target[0] += grad_values[0];
    target[1] += grad_values[1];
    target[2] += grad_values[2];
    target[3] += grad_values[3];
    target[4] += grad_values[4];
    target[5] += grad_values[5];
    target[6] += grad_values[6];
    target[7] += grad_values[7];
    target[8] += grad_values[8];
    target[9] += grad_values[9];
    target[10] += grad_values[10];
    target[11] += grad_values[11];
    target[12] += grad_values[12];
    target[13] += grad_values[13];
    target[14] += grad_values[14];
    target[15] += grad_values[15];
}

fn scalar_monomial_first_derivative_global(
    coeff: f64,
    indices: &[usize],
    x_row: &[f64],
    x_base: usize,
    derivative_position: usize,
) -> f64 {
    let mut derivative = coeff;
    for (position, &index) in indices.iter().enumerate() {
        if position != derivative_position {
            derivative *= x_row[index - x_base];
        }
    }
    derivative
}

fn accumulate_scalar_local_jacobian_term(
    coeff: f64,
    indices: &[usize],
    x_row: &[f64],
    jacobian_x: &mut [f64],
) {
    if coeff == 0.0 {
        return;
    }
    for (position, &index) in indices.iter().enumerate() {
        jacobian_x[index] +=
            scalar_monomial_first_derivative_global(coeff, indices, x_row, 0, position);
    }
}

fn monomial_except_value_and_dot(
    indices: &[usize],
    excluded_position: usize,
    x: &[f64],
    x_dot: &[f64],
) -> (f64, f64) {
    let mut value = 1.0;
    for (position, &index) in indices.iter().enumerate() {
        if position != excluded_position {
            value *= x[index];
        }
    }

    let mut dot = 0.0;
    for tangent_position in 0..indices.len() {
        if tangent_position == excluded_position {
            continue;
        }
        let mut term = x_dot[indices[tangent_position]];
        for (position, &index) in indices.iter().enumerate() {
            if position != excluded_position && position != tangent_position {
                term *= x[index];
            }
        }
        dot += term;
    }
    (value, dot)
}

fn accumulate_monomial_vjp_linearize(
    indices: &[usize],
    coeff: f64,
    bar_output: f64,
    bar_output_dot: f64,
    x: &[f64],
    x_dot: &[f64],
    grad_x_dot: &mut [f64],
) {
    for position in 0..indices.len() {
        let (except_value, except_dot) = monomial_except_value_and_dot(indices, position, x, x_dot);
        grad_x_dot[indices[position]] +=
            coeff * (bar_output_dot * except_value + bar_output * except_dot);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn mace_f64_avx2_enabled_from_flags(
    disable_simd: bool,
    disable_avx2: bool,
    disable_avx: bool,
    has_avx2: bool,
    has_fma: bool,
) -> bool {
    !disable_simd && !disable_avx2 && !disable_avx && has_avx2 && has_fma
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn mace_f64_avx2_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        // The AVX2 MACE contraction path uses lane-wise FMA accumulation rather
        // than the scalar Kahan reduction. Keep it runtime-gated so strict
        // diagnostics can force the scalar path with LLMLIP_DISABLE_SIMD/AVX2.
        mace_f64_avx2_enabled_from_flags(
            std::env::var_os("LLMLIP_DISABLE_SIMD").is_some(),
            std::env::var_os("LLMLIP_DISABLE_AVX2").is_some(),
            std::env::var_os("LLMLIP_DISABLE_AVX").is_some(),
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("fma"),
        )
    })
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments, unsafe_code)]
fn mace_scalar_contraction_forward_node_avx2_f64_safe(
    contraction: &MaceScalarSymmetricContraction<'_>,
    x_node: &[f64],
    element: usize,
    out_row: &mut [f64],
    u1_entries: &[ScalarU1Entry],
    u2_entries: &[ScalarU2Entry],
    u3_entries: &[ScalarU3Entry],
) {
    debug_assert!(mace_f64_avx2_enabled());
    // SAFETY: runtime AVX2/FMA detection guards this call. Callers pass
    // validated `num_ell == 16` per-node rows and contiguous feature weights.
    unsafe {
        mace_scalar_contraction_forward_node_avx2_f64(
            contraction,
            x_node,
            element,
            out_row,
            u1_entries,
            u2_entries,
            u3_entries,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[allow(clippy::too_many_arguments, unsafe_code)]
unsafe fn mace_scalar_contraction_forward_node_avx2_f64(
    contraction: &MaceScalarSymmetricContraction<'_>,
    x_node: &[f64],
    element: usize,
    out_row: &mut [f64],
    u1_entries: &[ScalarU1Entry],
    u2_entries: &[ScalarU2Entry],
    u3_entries: &[ScalarU3Entry],
) {
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn load4_feature(ptr: *const f64) -> __m256d {
        _mm256_set_pd(*ptr.add(48), *ptr.add(32), *ptr.add(16), *ptr)
    }

    let num_features = contraction.num_features;
    let mut feature = 0;
    while feature + 4 <= num_features {
        let mut x_by_ell = [_mm256_setzero_pd(); 16];
        for (ell, value) in x_by_ell.iter_mut().enumerate() {
            *value = load4_feature(x_node.as_ptr().add(feature * 16 + ell));
        }
        let mut acc = _mm256_setzero_pd();
        for entry in u1_entries {
            let weight = _mm256_loadu_pd(contraction.weights_u1.as_ptr().add(
                contraction.weight_index(element, entry.param, feature, contraction.u1_params),
            ));
            acc = _mm256_fmadd_pd(
                _mm256_mul_pd(weight, _mm256_set1_pd(entry.value)),
                x_by_ell[entry.ell],
                acc,
            );
        }
        for entry in u2_entries {
            let weight = _mm256_loadu_pd(contraction.weights_u2.as_ptr().add(
                contraction.weight_index(element, entry.param, feature, contraction.u2_params),
            ));
            acc = _mm256_fmadd_pd(
                _mm256_mul_pd(weight, _mm256_set1_pd(entry.value)),
                _mm256_mul_pd(x_by_ell[entry.left], x_by_ell[entry.right]),
                acc,
            );
        }
        for entry in u3_entries {
            let weight = _mm256_loadu_pd(contraction.weights_u3.as_ptr().add(
                contraction.weight_index(element, entry.param, feature, contraction.u3_params),
            ));
            acc = _mm256_fmadd_pd(
                _mm256_mul_pd(weight, _mm256_set1_pd(entry.value)),
                _mm256_mul_pd(
                    _mm256_mul_pd(x_by_ell[entry.left], x_by_ell[entry.right]),
                    x_by_ell[entry.inner],
                ),
                acc,
            );
        }
        _mm256_storeu_pd(out_row.as_mut_ptr().add(feature), acc);
        feature += 4;
    }
    while feature < num_features {
        out_row[feature] = mace_scalar_contraction_forward_feature_scalar_f64(
            contraction,
            x_node,
            element,
            feature,
            u1_entries,
            u2_entries,
            u3_entries,
        );
        feature += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments, unsafe_code)]
fn mace_scalar_contraction_vjp_node_avx2_f64_safe(
    contraction: &MaceScalarSymmetricContraction<'_>,
    x_node: &[f64],
    grad_out_row: &[f64],
    element: usize,
    grad_x_node: &mut [f64],
    u1_entries: &[ScalarU1Entry],
    u2_entries: &[ScalarU2Entry],
    u3_entries: &[ScalarU3Entry],
) {
    debug_assert!(mace_f64_avx2_enabled());
    // SAFETY: runtime AVX2/FMA detection guards this call. Callers pass
    // validated `num_ell == 16` per-node rows and disjoint gradient rows.
    unsafe {
        mace_scalar_contraction_vjp_node_avx2_f64(
            contraction,
            x_node,
            grad_out_row,
            element,
            grad_x_node,
            u1_entries,
            u2_entries,
            u3_entries,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[allow(clippy::too_many_arguments, unsafe_code)]
unsafe fn mace_scalar_contraction_vjp_node_avx2_f64(
    contraction: &MaceScalarSymmetricContraction<'_>,
    x_node: &[f64],
    grad_out_row: &[f64],
    element: usize,
    grad_x_node: &mut [f64],
    u1_entries: &[ScalarU1Entry],
    u2_entries: &[ScalarU2Entry],
    u3_entries: &[ScalarU3Entry],
) {
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn load4_feature(ptr: *const f64) -> __m256d {
        _mm256_set_pd(*ptr.add(48), *ptr.add(32), *ptr.add(16), *ptr)
    }

    #[inline(always)]
    unsafe fn add_store4_feature(ptr: *mut f64, value: __m256d) {
        let mut lanes = [0.0_f64; 4];
        _mm256_storeu_pd(lanes.as_mut_ptr(), value);
        for (lane, value) in lanes.iter().enumerate() {
            *ptr.add(lane * 16) += *value;
        }
    }

    let num_features = contraction.num_features;
    let mut feature = 0;
    while feature + 4 <= num_features {
        let bar = _mm256_loadu_pd(grad_out_row.as_ptr().add(feature));
        let mut x_by_ell = [_mm256_setzero_pd(); 16];
        for (ell, value) in x_by_ell.iter_mut().enumerate() {
            *value = load4_feature(x_node.as_ptr().add(feature * 16 + ell));
        }
        let mut grad = [_mm256_setzero_pd(); 16];
        for entry in u1_entries {
            let weight = _mm256_loadu_pd(contraction.weights_u1.as_ptr().add(
                contraction.weight_index(element, entry.param, feature, contraction.u1_params),
            ));
            let derivative = _mm256_mul_pd(_mm256_mul_pd(bar, weight), _mm256_set1_pd(entry.value));
            grad[entry.ell] = _mm256_add_pd(grad[entry.ell], derivative);
        }
        for entry in u2_entries {
            let weight = _mm256_loadu_pd(contraction.weights_u2.as_ptr().add(
                contraction.weight_index(element, entry.param, feature, contraction.u2_params),
            ));
            let derivative = _mm256_mul_pd(_mm256_mul_pd(bar, weight), _mm256_set1_pd(entry.value));
            grad[entry.left] = _mm256_fmadd_pd(derivative, x_by_ell[entry.right], grad[entry.left]);
            grad[entry.right] =
                _mm256_fmadd_pd(derivative, x_by_ell[entry.left], grad[entry.right]);
        }
        for entry in u3_entries {
            let weight = _mm256_loadu_pd(contraction.weights_u3.as_ptr().add(
                contraction.weight_index(element, entry.param, feature, contraction.u3_params),
            ));
            let derivative = _mm256_mul_pd(_mm256_mul_pd(bar, weight), _mm256_set1_pd(entry.value));
            let left = x_by_ell[entry.left];
            let right = x_by_ell[entry.right];
            let inner = x_by_ell[entry.inner];
            if entry.left == entry.right {
                if entry.right == entry.inner {
                    grad[entry.left] = _mm256_fmadd_pd(
                        _mm256_mul_pd(_mm256_set1_pd(3.0), derivative),
                        _mm256_mul_pd(left, left),
                        grad[entry.left],
                    );
                } else {
                    grad[entry.left] = _mm256_fmadd_pd(
                        _mm256_mul_pd(_mm256_set1_pd(2.0), derivative),
                        _mm256_mul_pd(left, inner),
                        grad[entry.left],
                    );
                    grad[entry.inner] =
                        _mm256_fmadd_pd(derivative, _mm256_mul_pd(left, left), grad[entry.inner]);
                }
            } else if entry.right == entry.inner {
                grad[entry.left] =
                    _mm256_fmadd_pd(derivative, _mm256_mul_pd(right, right), grad[entry.left]);
                grad[entry.right] = _mm256_fmadd_pd(
                    _mm256_mul_pd(_mm256_set1_pd(2.0), derivative),
                    _mm256_mul_pd(left, right),
                    grad[entry.right],
                );
            } else {
                grad[entry.left] =
                    _mm256_fmadd_pd(derivative, _mm256_mul_pd(right, inner), grad[entry.left]);
                grad[entry.right] =
                    _mm256_fmadd_pd(derivative, _mm256_mul_pd(left, inner), grad[entry.right]);
                grad[entry.inner] =
                    _mm256_fmadd_pd(derivative, _mm256_mul_pd(left, right), grad[entry.inner]);
            }
        }
        for (ell, value) in grad.iter().enumerate() {
            add_store4_feature(grad_x_node.as_mut_ptr().add(feature * 16 + ell), *value);
        }
        feature += 4;
    }
    while feature < num_features {
        let bar_output = grad_out_row[feature];
        if bar_output != 0.0 {
            let x_row = &x_node[feature * 16..feature * 16 + 16];
            let grad_row = &mut grad_x_node[feature * 16..feature * 16 + 16];
            for entry in u1_entries {
                let weight = contraction.weights_u1[contraction.weight_index(
                    element,
                    entry.param,
                    feature,
                    contraction.u1_params,
                )];
                grad_row[entry.ell] += bar_output * entry.value * weight;
            }
            for entry in u2_entries {
                let derivative = bar_output
                    * entry.value
                    * contraction.weights_u2[contraction.weight_index(
                        element,
                        entry.param,
                        feature,
                        contraction.u2_params,
                    )];
                grad_row[entry.left] += derivative * x_row[entry.right];
                grad_row[entry.right] += derivative * x_row[entry.left];
            }
            for entry in u3_entries {
                let derivative = bar_output
                    * entry.value
                    * contraction.weights_u3[contraction.weight_index(
                        element,
                        entry.param,
                        feature,
                        contraction.u3_params,
                    )];
                let x_left = x_row[entry.left];
                let x_right = x_row[entry.right];
                let x_inner = x_row[entry.inner];
                if entry.left == entry.right {
                    if entry.right == entry.inner {
                        grad_row[entry.left] += 3.0 * derivative * x_left * x_left;
                    } else {
                        grad_row[entry.left] += 2.0 * derivative * x_left * x_inner;
                        grad_row[entry.inner] += derivative * x_left * x_left;
                    }
                } else if entry.right == entry.inner {
                    grad_row[entry.left] += derivative * x_right * x_right;
                    grad_row[entry.right] += 2.0 * derivative * x_left * x_right;
                } else {
                    grad_row[entry.left] += derivative * x_right * x_inner;
                    grad_row[entry.right] += derivative * x_left * x_inner;
                    grad_row[entry.inner] += derivative * x_left * x_right;
                }
            }
        }
        feature += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments, unsafe_code)]
fn mace_vector_contraction_forward_node_avx2_f64_safe(
    contraction: &MaceVectorSymmetricContraction<'_>,
    x_node: &[f64],
    element: usize,
    out_row: &mut [f64],
    u1_entries: &[Vec<VectorU1Entry>],
    u2_entries: &[Vec<VectorU2Entry>],
    u3_entries: &[Vec<VectorU3Entry>],
) {
    debug_assert!(mace_f64_avx2_enabled());
    // SAFETY: runtime AVX2/FMA detection guards this call. Callers pass
    // validated `num_ell == 16` per-node rows and output rows.
    unsafe {
        mace_vector_contraction_forward_node_avx2_f64(
            contraction,
            x_node,
            element,
            out_row,
            u1_entries,
            u2_entries,
            u3_entries,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[allow(clippy::too_many_arguments, unsafe_code)]
unsafe fn mace_vector_contraction_forward_node_avx2_f64(
    contraction: &MaceVectorSymmetricContraction<'_>,
    x_node: &[f64],
    element: usize,
    out_row: &mut [f64],
    u1_entries: &[Vec<VectorU1Entry>],
    u2_entries: &[Vec<VectorU2Entry>],
    u3_entries: &[Vec<VectorU3Entry>],
) {
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn load4_feature(ptr: *const f64) -> __m256d {
        _mm256_set_pd(*ptr.add(48), *ptr.add(32), *ptr.add(16), *ptr)
    }

    #[inline(always)]
    unsafe fn store4_strided(ptr: *mut f64, stride: usize, value: __m256d) {
        let mut lanes = [0.0_f64; 4];
        _mm256_storeu_pd(lanes.as_mut_ptr(), value);
        for (lane, value) in lanes.iter().enumerate() {
            *ptr.add(lane * stride) = *value;
        }
    }

    let num_features = contraction.num_features;
    let num_equiv = contraction.num_equivariance;
    let mut feature = 0;
    while feature + 4 <= num_features {
        let mut x_by_ell = [_mm256_setzero_pd(); 16];
        for (ell, value) in x_by_ell.iter_mut().enumerate() {
            *value = load4_feature(x_node.as_ptr().add(feature * 16 + ell));
        }
        for equiv in 0..num_equiv {
            let mut acc = _mm256_setzero_pd();
            for entry in &u1_entries[equiv] {
                let weight = _mm256_loadu_pd(contraction.weights_u1.as_ptr().add(
                    contraction.weight_index(element, entry.param, feature, contraction.u1_params),
                ));
                acc = _mm256_fmadd_pd(
                    _mm256_mul_pd(weight, _mm256_set1_pd(entry.value)),
                    x_by_ell[entry.ell],
                    acc,
                );
            }
            for entry in &u2_entries[equiv] {
                let weight = _mm256_loadu_pd(contraction.weights_u2.as_ptr().add(
                    contraction.weight_index(element, entry.param, feature, contraction.u2_params),
                ));
                acc = _mm256_fmadd_pd(
                    _mm256_mul_pd(weight, _mm256_set1_pd(entry.value)),
                    _mm256_mul_pd(x_by_ell[entry.left], x_by_ell[entry.right]),
                    acc,
                );
            }
            for entry in &u3_entries[equiv] {
                let weight = _mm256_loadu_pd(contraction.weights_u3.as_ptr().add(
                    contraction.weight_index(element, entry.param, feature, contraction.u3_params),
                ));
                acc = _mm256_fmadd_pd(
                    _mm256_mul_pd(weight, _mm256_set1_pd(entry.value)),
                    _mm256_mul_pd(
                        _mm256_mul_pd(x_by_ell[entry.left], x_by_ell[entry.right]),
                        x_by_ell[entry.inner],
                    ),
                    acc,
                );
            }
            store4_strided(
                out_row.as_mut_ptr().add(feature * num_equiv + equiv),
                num_equiv,
                acc,
            );
        }
        feature += 4;
    }
    while feature < num_features {
        for equiv in 0..num_equiv {
            out_row[feature * num_equiv + equiv] =
                mace_vector_contraction_forward_feature_scalar_f64(
                    contraction,
                    x_node,
                    element,
                    feature,
                    equiv,
                    u1_entries,
                    u2_entries,
                    u3_entries,
                );
        }
        feature += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments, unsafe_code)]
fn mace_vector_contraction_vjp_node_avx2_f64_safe(
    contraction: &MaceVectorSymmetricContraction<'_>,
    x_node: &[f64],
    grad_out_row: &[f64],
    element: usize,
    grad_x_node: &mut [f64],
    u1_entries: &[Vec<VectorU1Entry>],
    u2_entries: &[Vec<VectorU2Entry>],
    u3_entries: &[Vec<VectorU3Entry>],
) {
    debug_assert!(mace_f64_avx2_enabled());
    // SAFETY: runtime AVX2/FMA detection guards this call. Callers pass
    // validated `num_ell == 16` per-node rows and disjoint gradient rows.
    unsafe {
        mace_vector_contraction_vjp_node_avx2_f64(
            contraction,
            x_node,
            grad_out_row,
            element,
            grad_x_node,
            u1_entries,
            u2_entries,
            u3_entries,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[allow(clippy::too_many_arguments, unsafe_code)]
unsafe fn mace_vector_contraction_vjp_node_avx2_f64(
    contraction: &MaceVectorSymmetricContraction<'_>,
    x_node: &[f64],
    grad_out_row: &[f64],
    element: usize,
    grad_x_node: &mut [f64],
    u1_entries: &[Vec<VectorU1Entry>],
    u2_entries: &[Vec<VectorU2Entry>],
    u3_entries: &[Vec<VectorU3Entry>],
) {
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn load4_feature(ptr: *const f64) -> __m256d {
        _mm256_set_pd(*ptr.add(48), *ptr.add(32), *ptr.add(16), *ptr)
    }

    #[inline(always)]
    unsafe fn load4_strided(ptr: *const f64, stride: usize) -> __m256d {
        _mm256_set_pd(
            *ptr.add(3 * stride),
            *ptr.add(2 * stride),
            *ptr.add(stride),
            *ptr,
        )
    }

    #[inline(always)]
    unsafe fn add_store4_feature(ptr: *mut f64, value: __m256d) {
        let mut lanes = [0.0_f64; 4];
        _mm256_storeu_pd(lanes.as_mut_ptr(), value);
        for (lane, value) in lanes.iter().enumerate() {
            *ptr.add(lane * 16) += *value;
        }
    }

    let num_features = contraction.num_features;
    let num_equiv = contraction.num_equivariance;
    let mut feature = 0;
    while feature + 4 <= num_features {
        let mut x_by_ell = [_mm256_setzero_pd(); 16];
        for (ell, value) in x_by_ell.iter_mut().enumerate() {
            *value = load4_feature(x_node.as_ptr().add(feature * 16 + ell));
        }
        let mut grad = [_mm256_setzero_pd(); 16];
        for equiv in 0..num_equiv {
            let bar = load4_strided(
                grad_out_row.as_ptr().add(feature * num_equiv + equiv),
                num_equiv,
            );
            for entry in &u1_entries[equiv] {
                let weight = _mm256_loadu_pd(contraction.weights_u1.as_ptr().add(
                    contraction.weight_index(element, entry.param, feature, contraction.u1_params),
                ));
                let derivative =
                    _mm256_mul_pd(_mm256_mul_pd(bar, weight), _mm256_set1_pd(entry.value));
                grad[entry.ell] = _mm256_add_pd(grad[entry.ell], derivative);
            }
            for entry in &u2_entries[equiv] {
                let weight = _mm256_loadu_pd(contraction.weights_u2.as_ptr().add(
                    contraction.weight_index(element, entry.param, feature, contraction.u2_params),
                ));
                let derivative =
                    _mm256_mul_pd(_mm256_mul_pd(bar, weight), _mm256_set1_pd(entry.value));
                grad[entry.left] =
                    _mm256_fmadd_pd(derivative, x_by_ell[entry.right], grad[entry.left]);
                grad[entry.right] =
                    _mm256_fmadd_pd(derivative, x_by_ell[entry.left], grad[entry.right]);
            }
            for entry in &u3_entries[equiv] {
                let weight = _mm256_loadu_pd(contraction.weights_u3.as_ptr().add(
                    contraction.weight_index(element, entry.param, feature, contraction.u3_params),
                ));
                let derivative =
                    _mm256_mul_pd(_mm256_mul_pd(bar, weight), _mm256_set1_pd(entry.value));
                let left = x_by_ell[entry.left];
                let right = x_by_ell[entry.right];
                let inner = x_by_ell[entry.inner];
                if entry.left == entry.right {
                    if entry.right == entry.inner {
                        grad[entry.left] = _mm256_fmadd_pd(
                            _mm256_mul_pd(_mm256_set1_pd(3.0), derivative),
                            _mm256_mul_pd(left, left),
                            grad[entry.left],
                        );
                    } else {
                        grad[entry.left] = _mm256_fmadd_pd(
                            _mm256_mul_pd(_mm256_set1_pd(2.0), derivative),
                            _mm256_mul_pd(left, inner),
                            grad[entry.left],
                        );
                        grad[entry.inner] = _mm256_fmadd_pd(
                            derivative,
                            _mm256_mul_pd(left, left),
                            grad[entry.inner],
                        );
                    }
                } else if entry.right == entry.inner {
                    grad[entry.left] =
                        _mm256_fmadd_pd(derivative, _mm256_mul_pd(right, right), grad[entry.left]);
                    grad[entry.right] = _mm256_fmadd_pd(
                        _mm256_mul_pd(_mm256_set1_pd(2.0), derivative),
                        _mm256_mul_pd(left, right),
                        grad[entry.right],
                    );
                } else {
                    grad[entry.left] =
                        _mm256_fmadd_pd(derivative, _mm256_mul_pd(right, inner), grad[entry.left]);
                    grad[entry.right] =
                        _mm256_fmadd_pd(derivative, _mm256_mul_pd(left, inner), grad[entry.right]);
                    grad[entry.inner] =
                        _mm256_fmadd_pd(derivative, _mm256_mul_pd(left, right), grad[entry.inner]);
                }
            }
        }
        for (ell, value) in grad.iter().enumerate() {
            add_store4_feature(grad_x_node.as_mut_ptr().add(feature * 16 + ell), *value);
        }
        feature += 4;
    }
    while feature < num_features {
        mace_vector_contraction_vjp_feature_scalar_f64(
            contraction,
            x_node,
            grad_out_row,
            element,
            feature,
            grad_x_node,
            u1_entries,
            u2_entries,
            u3_entries,
        );
        feature += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn mace_scalar_contraction_forward_feature_scalar_f64(
    contraction: &MaceScalarSymmetricContraction<'_>,
    x_node: &[f64],
    element: usize,
    feature: usize,
    u1_entries: &[ScalarU1Entry],
    u2_entries: &[ScalarU2Entry],
    u3_entries: &[ScalarU3Entry],
) -> f64 {
    let x_row = &x_node[feature * 16..feature * 16 + 16];
    let mut output = 0.0_f64;
    for entry in u1_entries {
        let weight = contraction.weights_u1
            [contraction.weight_index(element, entry.param, feature, contraction.u1_params)];
        output += entry.value * weight * x_row[entry.ell];
    }
    for entry in u2_entries {
        let weight = contraction.weights_u2
            [contraction.weight_index(element, entry.param, feature, contraction.u2_params)];
        output += entry.value * weight * x_row[entry.left] * x_row[entry.right];
    }
    for entry in u3_entries {
        let weight = contraction.weights_u3
            [contraction.weight_index(element, entry.param, feature, contraction.u3_params)];
        output +=
            entry.value * weight * x_row[entry.left] * x_row[entry.right] * x_row[entry.inner];
    }
    output
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn mace_vector_contraction_forward_feature_scalar_f64(
    contraction: &MaceVectorSymmetricContraction<'_>,
    x_node: &[f64],
    element: usize,
    feature: usize,
    equiv: usize,
    u1_entries: &[Vec<VectorU1Entry>],
    u2_entries: &[Vec<VectorU2Entry>],
    u3_entries: &[Vec<VectorU3Entry>],
) -> f64 {
    let x_row = &x_node[feature * 16..feature * 16 + 16];
    let mut output = 0.0_f64;
    for entry in &u1_entries[equiv] {
        let weight = contraction.weights_u1
            [contraction.weight_index(element, entry.param, feature, contraction.u1_params)];
        output += entry.value * weight * x_row[entry.ell];
    }
    for entry in &u2_entries[equiv] {
        let weight = contraction.weights_u2
            [contraction.weight_index(element, entry.param, feature, contraction.u2_params)];
        output += entry.value * weight * x_row[entry.left] * x_row[entry.right];
    }
    for entry in &u3_entries[equiv] {
        let weight = contraction.weights_u3
            [contraction.weight_index(element, entry.param, feature, contraction.u3_params)];
        output +=
            entry.value * weight * x_row[entry.left] * x_row[entry.right] * x_row[entry.inner];
    }
    output
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn mace_vector_contraction_vjp_feature_scalar_f64(
    contraction: &MaceVectorSymmetricContraction<'_>,
    x_node: &[f64],
    grad_out_row: &[f64],
    element: usize,
    feature: usize,
    grad_x_node: &mut [f64],
    u1_entries: &[Vec<VectorU1Entry>],
    u2_entries: &[Vec<VectorU2Entry>],
    u3_entries: &[Vec<VectorU3Entry>],
) {
    let x_row = &x_node[feature * 16..feature * 16 + 16];
    let grad_row = &mut grad_x_node[feature * 16..feature * 16 + 16];
    for equiv in 0..contraction.num_equivariance {
        let bar_output = grad_out_row[feature * contraction.num_equivariance + equiv];
        if bar_output == 0.0 {
            continue;
        }
        for entry in &u1_entries[equiv] {
            let weight = contraction.weights_u1
                [contraction.weight_index(element, entry.param, feature, contraction.u1_params)];
            grad_row[entry.ell] += bar_output * entry.value * weight;
        }
        for entry in &u2_entries[equiv] {
            let derivative = bar_output
                * entry.value
                * contraction.weights_u2[contraction.weight_index(
                    element,
                    entry.param,
                    feature,
                    contraction.u2_params,
                )];
            grad_row[entry.left] += derivative * x_row[entry.right];
            grad_row[entry.right] += derivative * x_row[entry.left];
        }
        for entry in &u3_entries[equiv] {
            let derivative = bar_output
                * entry.value
                * contraction.weights_u3[contraction.weight_index(
                    element,
                    entry.param,
                    feature,
                    contraction.u3_params,
                )];
            let x_left = x_row[entry.left];
            let x_right = x_row[entry.right];
            let x_inner = x_row[entry.inner];
            if entry.left == entry.right {
                if entry.right == entry.inner {
                    grad_row[entry.left] += 3.0 * derivative * x_left * x_left;
                } else {
                    grad_row[entry.left] += 2.0 * derivative * x_left * x_inner;
                    grad_row[entry.inner] += derivative * x_left * x_left;
                }
            } else if entry.right == entry.inner {
                grad_row[entry.left] += derivative * x_right * x_right;
                grad_row[entry.right] += 2.0 * derivative * x_left * x_right;
            } else {
                grad_row[entry.left] += derivative * x_right * x_inner;
                grad_row[entry.right] += derivative * x_left * x_inner;
                grad_row[entry.inner] += derivative * x_left * x_right;
            }
        }
    }
}

#[inline(always)]
fn kahan_add(sum: &mut f64, compensation: &mut f64, term: f64) {
    let y = term - *compensation;
    let t = *sum + y;
    *compensation = (t - *sum) - y;
    *sum = t;
}

fn flatten_weighted_entry_groups<T: Clone>(groups: &[Vec<T>]) -> (Vec<T>, Vec<(usize, usize)>) {
    let total_len = groups.iter().map(Vec::len).sum();
    let mut flat = Vec::with_capacity(total_len);
    let mut ranges = Vec::with_capacity(groups.len());
    for group in groups {
        let start = flat.len();
        flat.extend_from_slice(group);
        ranges.push((start, flat.len()));
    }
    (flat, ranges)
}

fn validate_nonzero(op: &'static str, value: usize) -> Result<()> {
    if value == 0 {
        Err(CoreError::InvalidTensorProduct(format!(
            "{op} must be non-zero"
        )))
    } else {
        Ok(())
    }
}
