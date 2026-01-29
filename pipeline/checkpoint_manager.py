"""
Checkpoint manager for tracking stage outputs and checkpoint provenance.

Handles:
- Tracking outputs from each pipeline stage
- Skip-if-epochs-zero logic for checkpoints
- Recording whether checkpoints were provided externally or trained
"""

import os
import json
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class StageResult:
    """Result of a pipeline stage."""
    stage_name: str
    status: str  # "completed", "skipped", "failed"
    output_dir: Optional[str] = None
    checkpoint_path: Optional[str] = None
    checkpoint_source: str = "trained"  # "trained", "provided", "skipped"
    duration_seconds: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass 
class PipelineState:
    """Complete pipeline state tracking."""
    run_id: str
    config_path: Optional[str] = None
    stages: Dict[str, StageResult] = field(default_factory=dict)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    status: str = "running"  # "running", "completed", "failed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            'run_id': self.run_id,
            'config_path': self.config_path,
            'stages': {k: asdict(v) for k, v in self.stages.items()},
            'start_time': self.start_time,
            'end_time': self.end_time,
            'status': self.status,
        }


class CheckpointManager:
    """
    Manages checkpoints and stage outputs for the pipeline.
    
    Responsibilities:
    - Track outputs from each stage
    - Handle checkpoint injection (use provided vs train new)
    - Implement skip-if-epochs-zero logic
    - Record checkpoint provenance for reporting
    """
    
    STAGE_ORDER = [
        "data_prep",
        "scattering", 
        "splitting",
        "train_grassy",
        "train_dit",
        "evaluate",
        "sample_unconstrained",
        "sample_property_opt",
    ]
    
    def __init__(self, output_dir: str, run_id: Optional[str] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Base output directory for all pipeline artifacts
            run_id: Unique run identifier (auto-generated if None)
        """
        self.output_dir = Path(output_dir)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state = PipelineState(run_id=self.run_id)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_stage_dir(self, stage_name: str) -> Path:
        """Get output directory for a stage."""
        stage_idx = self.STAGE_ORDER.index(stage_name) + 1
        stage_dir = self.output_dir / f"stage_{stage_idx}_{stage_name}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir
    
    def should_skip_training(self, stage_name: str, checkpoint: Optional[str], epochs: int) -> bool:
        """
        Determine if training should be skipped for a stage.
        
        Args:
            stage_name: Name of the training stage
            checkpoint: Path to provided checkpoint (or None)
            epochs: Number of epochs to train
        
        Returns:
            True if training should be skipped (checkpoint provided + epochs=0)
        
        Raises:
            ValueError: If epochs=0 but no checkpoint provided
        """
        if epochs == 0:
            if checkpoint is None:
                raise ValueError(
                    f"Cannot skip {stage_name} training (epochs=0) without providing a checkpoint"
                )
            return True
        return False
    
    def register_provided_checkpoint(
        self, 
        stage_name: str, 
        checkpoint_path: str,
        copy_to_output: bool = True
    ) -> str:
        """
        Register an externally provided checkpoint.
        
        Args:
            stage_name: Name of the stage
            checkpoint_path: Path to the provided checkpoint
            copy_to_output: Whether to copy checkpoint to output directory
        
        Returns:
            Path to the checkpoint (copied path if copy_to_output=True)
        """
        stage_dir = self.get_stage_dir(stage_name)
        
        if copy_to_output:
            # Copy checkpoint to stage directory
            src = Path(checkpoint_path)
            dst = stage_dir / f"provided_{src.name}"
            shutil.copy2(src, dst)
            final_path = str(dst)
        else:
            final_path = checkpoint_path
        
        # Record in state
        self.state.stages[stage_name] = StageResult(
            stage_name=stage_name,
            status="skipped",
            output_dir=str(stage_dir),
            checkpoint_path=final_path,
            checkpoint_source="provided",
        )
        
        return final_path
    
    def register_stage_start(self, stage_name: str) -> Path:
        """
        Register that a stage is starting.
        
        Returns:
            Output directory for the stage
        """
        stage_dir = self.get_stage_dir(stage_name)
        
        self.state.stages[stage_name] = StageResult(
            stage_name=stage_name,
            status="running",
            output_dir=str(stage_dir),
            checkpoint_source="trained",
        )
        
        return stage_dir
    
    def register_stage_complete(
        self,
        stage_name: str,
        checkpoint_path: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        duration_seconds: float = 0.0,
    ) -> None:
        """Register successful completion of a stage."""
        if stage_name in self.state.stages:
            result = self.state.stages[stage_name]
            result.status = "completed"
            result.checkpoint_path = checkpoint_path
            result.duration_seconds = duration_seconds
            result.metrics = metrics or {}
        else:
            stage_dir = self.get_stage_dir(stage_name)
            self.state.stages[stage_name] = StageResult(
                stage_name=stage_name,
                status="completed",
                output_dir=str(stage_dir),
                checkpoint_path=checkpoint_path,
                checkpoint_source="trained",
                duration_seconds=duration_seconds,
                metrics=metrics or {},
            )
    
    def register_stage_failed(self, stage_name: str, error: str) -> None:
        """Register stage failure."""
        if stage_name in self.state.stages:
            result = self.state.stages[stage_name]
            result.status = "failed"
            result.error = error
        else:
            stage_dir = self.get_stage_dir(stage_name)
            self.state.stages[stage_name] = StageResult(
                stage_name=stage_name,
                status="failed",
                output_dir=str(stage_dir),
                error=error,
            )
        self.state.status = "failed"
    
    def get_stage_output(self, stage_name: str) -> Optional[StageResult]:
        """Get result from a completed stage."""
        return self.state.stages.get(stage_name)
    
    def get_checkpoint(self, stage_name: str) -> Optional[str]:
        """Get checkpoint path for a stage."""
        result = self.state.stages.get(stage_name)
        return result.checkpoint_path if result else None
    
    def finalize(self, status: str = "completed") -> None:
        """Finalize the pipeline run."""
        self.state.end_time = datetime.now().isoformat()
        self.state.status = status
        self.save_state()
    
    def save_state(self) -> None:
        """Save pipeline state to JSON file."""
        state_path = self.output_dir / "pipeline_state.json"
        with open(state_path, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
    
    def load_state(self, state_path: str) -> None:
        """Load pipeline state from JSON file."""
        with open(state_path, 'r') as f:
            data = json.load(f)
        
        self.state = PipelineState(
            run_id=data['run_id'],
            config_path=data.get('config_path'),
            start_time=data['start_time'],
            end_time=data.get('end_time'),
            status=data['status'],
        )
        
        for name, stage_data in data.get('stages', {}).items():
            self.state.stages[name] = StageResult(**stage_data)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline state for reporting."""
        return {
            'run_id': self.run_id,
            'output_dir': str(self.output_dir),
            'status': self.state.status,
            'stages': {
                name: {
                    'status': result.status,
                    'checkpoint_source': result.checkpoint_source,
                    'duration': f"{result.duration_seconds:.1f}s",
                }
                for name, result in self.state.stages.items()
            },
            'total_duration': sum(
                r.duration_seconds for r in self.state.stages.values()
            ),
        }
