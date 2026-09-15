// SPDX-License-Identifier: MIT OR Apache-2.0

//! Example per-step observer that writes JSON Lines (JSONL) to a writer.
//!
//! Serialization lives in application/example code, not in `plasticity-lab`.
//! The core crate only delivers a borrowed [`TrainingStepEvent`].

use std::io::{self, Write};

use neuromod::SpikingNetwork;
use plasticity_lab::{
    PlasticityTrainer, TrainingConfig, TrainingExample, TrainingObserver, TrainingStepEvent,
};
use serde_json::json;

/// Writes one JSON object per successful training step.
struct JsonlObserver<W: Write> {
    writer: W,
}

impl<W: Write> TrainingObserver for JsonlObserver<W> {
    type Error = io::Error;

    fn on_step(&mut self, event: TrainingStepEvent<'_>) -> Result<(), Self::Error> {
        let record = json!({
            "step_index": event.step_index,
            "reward": event.reward,
            "spike_indices": event.spike_indices,
            "steps_processed": event.steps_processed,
            "total_spikes": event.total_spikes,
            "modulators": {
                "dopamine": event.modulators.dopamine,
                "serotonin": event.modulators.serotonin,
                "acetylcholine": event.modulators.acetylcholine,
                "norepinephrine": event.modulators.norepinephrine,
            },
        });
        writeln!(self.writer, "{record}")
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut trainer = PlasticityTrainer::new(TrainingConfig::default());
    let mut network = SpikingNetwork::with_dimensions(4, 2, 8);
    let batch = vec![
        TrainingExample {
            stimuli: vec![0.25; 8],
            reward: 0.2,
        },
        TrainingExample {
            stimuli: vec![0.4; 8],
            reward: -0.1,
        },
    ];

    let mut observer = JsonlObserver {
        writer: io::stdout(),
    };
    let summary = trainer.run_session_with_observer(&mut network, &batch, &mut observer)?;
    eprintln!(
        "processed={}, avg_reward={}, total_spikes={}",
        summary.steps_processed, summary.avg_reward, summary.total_spikes
    );
    Ok(())
}
