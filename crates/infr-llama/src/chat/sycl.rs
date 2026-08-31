//! [`SyclDenseChat`]: the Intel SYCL/oneAPI [`ChatModel`] for dense/MoE (`--dev sycl` /
//! `INFR_DEV=sycl`). Only compiled with the `sycl` feature — see `crates/infr-sycl`'s doc.

use super::ChatModel;
use crate::{GenStats, SeamModel};
use anyhow::Result;

/// Intel SYCL/oneAPI backend for dense/MoE: the agnostic compute-graph forward, driven through
/// [`infr_sycl::SyclBackend`] (which forwards every kernel to the CPU reference interpreter — see
/// that crate's doc — while still initializing a real SYCL device). Stateless full-prefill each
/// turn, like [`super::CpuDenseChat`]; the shared `Chat` feeds the full rendered history every
/// turn, so multi-turn context still works.
pub struct SyclDenseChat {
    model: SeamModel,
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl SyclDenseChat {
    pub fn new(model: SeamModel) -> Self {
        Self { model }
    }
}

#[cfg_attr(infr_profile, infr_prof::instrument)]
impl ChatModel for SyclDenseChat {
    fn render_model(&self) -> &SeamModel {
        &self.model
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_new: usize,
        req: Option<&crate::sampling::RequestCtx>,
        on_piece: &mut dyn FnMut(&str),
    ) -> Result<GenStats> {
        self.model
            .generate_sycl(prompt, max_new, req, |p| on_piece(p))
    }
}
