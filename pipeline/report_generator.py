"""
Report generator for pipeline results.

Generates comprehensive reports in Markdown and JSON formats,
including checkpoint provenance and metrics from all stages.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .checkpoint_manager import CheckpointManager
from .pipeline_config import PipelineConfig


class ReportGenerator:
    """Generates comprehensive pipeline reports."""
    
    def __init__(self, output_dir: str):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        ckpt_manager: CheckpointManager,
        config: PipelineConfig,
        results: Dict[str, Any],
    ) -> str:
        """
        Generate the final pipeline report.
        
        Args:
            ckpt_manager: Checkpoint manager with stage results
            config: Pipeline configuration
            results: Results dictionary from all stages
        
        Returns:
            Path to the generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate JSON report
        json_report = self._generate_json_report(ckpt_manager, config, results)
        json_path = self.output_dir / f"pipeline_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        # Generate Markdown report
        md_report = self._generate_markdown_report(ckpt_manager, config, results)
        md_path = self.output_dir / f"pipeline_report_{timestamp}.md"
        with open(md_path, 'w') as f:
            f.write(md_report)
        
        print(f"\nReports generated:")
        print(f"  - {json_path}")
        print(f"  - {md_path}")
        
        return str(md_path)
    
    def _generate_json_report(
        self,
        ckpt_manager: CheckpointManager,
        config: PipelineConfig,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate JSON report."""
        report = {
            'metadata': {
                'run_id': ckpt_manager.run_id,
                'timestamp': datetime.now().isoformat(),
                'output_dir': str(self.output_dir),
            },
            'config': config.to_dict(),
            'stages': {},
            'summary': {},
        }
        
        # Stage results
        for stage_name, stage_result in ckpt_manager.state.stages.items():
            report['stages'][stage_name] = {
                'status': stage_result.status,
                'checkpoint_source': stage_result.checkpoint_source,
                'checkpoint_path': stage_result.checkpoint_path,
                'duration_seconds': stage_result.duration_seconds,
                'metrics': stage_result.metrics,
            }
        
        # Summary
        total_duration = sum(
            r.duration_seconds for r in ckpt_manager.state.stages.values()
        )
        report['summary'] = {
            'total_duration_seconds': total_duration,
            'total_duration_formatted': self._format_duration(total_duration),
            'stages_completed': sum(
                1 for r in ckpt_manager.state.stages.values() if r.status == 'completed'
            ),
            'stages_skipped': sum(
                1 for r in ckpt_manager.state.stages.values() if r.status == 'skipped'
            ),
            'stages_failed': sum(
                1 for r in ckpt_manager.state.stages.values() if r.status == 'failed'
            ),
        }
        
        # Evaluation metrics (if available)
        if 'evaluate' in results and 'metrics' in results['evaluate']:
            report['evaluation_metrics'] = results['evaluate']['metrics']
        
        return report
    
    def _generate_markdown_report(
        self,
        ckpt_manager: CheckpointManager,
        config: PipelineConfig,
        results: Dict[str, Any],
    ) -> str:
        """Generate Markdown report."""
        lines = []
        
        # Header
        lines.append("# GRASSY-Net Pipeline Report")
        lines.append("")
        lines.append(f"**Run ID:** {ckpt_manager.run_id}")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Output Directory:** `{self.output_dir}`")
        lines.append("")
        
        # Configuration Summary
        lines.append("## Configuration")
        lines.append("")
        lines.append("### Dataset")
        lines.append(f"- **Input:** `{config.dataset.input_file}`")
        lines.append("")
        
        lines.append("### Splitting")
        lines.append(f"- **Train/Val/Test:** {config.splitting.train_ratio}/{config.splitting.val_ratio}/{config.splitting.test_ratio}")
        lines.append(f"- **Seed:** {config.splitting.seed}")
        lines.append("")
        
        lines.append("### GRASSY")
        lines.append(f"- **Epochs:** {config.grassy.epochs}")
        lines.append(f"- **Checkpoint:** `{config.grassy.checkpoint or 'None (trained from scratch)'}`")
        lines.append(f"- **Bottleneck dim:** {config.grassy.bottle_dim}")
        lines.append("")
        
        lines.append("### DiT")
        lines.append(f"- **Epochs:** {config.dit.epochs}")
        lines.append(f"- **Checkpoint:** `{config.dit.checkpoint or 'None (trained from scratch)'}`")
        lines.append(f"- **Architecture:** {config.dit.num_layer} layers × {config.dit.hidden_size} hidden × {config.dit.num_head} heads")
        lines.append(f"- **Noise:** prob={config.dit.noise_prob}, std={config.dit.noise_std}")
        lines.append("")
        
        # Stage Results
        lines.append("## Pipeline Stages")
        lines.append("")
        lines.append("| Stage | Status | Source | Duration |")
        lines.append("|-------|--------|--------|----------|")
        
        for stage_name in CheckpointManager.STAGE_ORDER:
            if stage_name in ckpt_manager.state.stages:
                result = ckpt_manager.state.stages[stage_name]
                status_icon = {
                    'completed': '✅',
                    'skipped': '⏭️',
                    'failed': '❌',
                    'running': '🔄',
                }.get(result.status, '❓')
                
                source = result.checkpoint_source if result.checkpoint_source else '-'
                duration = self._format_duration(result.duration_seconds)
                
                lines.append(f"| {stage_name} | {status_icon} {result.status} | {source} | {duration} |")
        
        lines.append("")
        
        # Checkpoint Provenance
        lines.append("## Checkpoint Provenance")
        lines.append("")
        lines.append("Tracks whether checkpoints were trained in this run or provided externally.")
        lines.append("")
        
        grassy_result = ckpt_manager.state.stages.get('train_grassy')
        if grassy_result:
            if grassy_result.checkpoint_source == 'provided':
                lines.append(f"- **GRASSY:** Provided externally (`{grassy_result.checkpoint_path}`)")
            else:
                lines.append(f"- **GRASSY:** Trained in this run (`{grassy_result.checkpoint_path}`)")
        
        dit_result = ckpt_manager.state.stages.get('train_dit')
        if dit_result:
            if dit_result.checkpoint_source == 'provided':
                lines.append(f"- **DiT:** Provided externally (`{dit_result.checkpoint_path}`)")
            else:
                lines.append(f"- **DiT:** Trained in this run (`{dit_result.checkpoint_path}`)")
        
        lines.append("")
        
        # Evaluation Results
        if 'evaluate' in results and 'metrics' in results['evaluate']:
            metrics = results['evaluate']['metrics']
            lines.append("## Evaluation Results")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            
            metric_display = [
                ('Validity', 'validity'),
                ('Validity (Lipinski)', 'validity_filtered'),
                ('Uniqueness', 'uniqueness'),
                ('Novelty', 'novelty'),
                ('Diversity', 'diversity'),
                ('FCD', 'fcd'),
            ]
            
            for display_name, key in metric_display:
                value = metrics.get(key, 'N/A')
                if isinstance(value, float):
                    if key == 'fcd':
                        lines.append(f"| {display_name} | {value:.4f} |")
                    else:
                        lines.append(f"| {display_name} | {value:.2%} |")
                else:
                    lines.append(f"| {display_name} | {value} |")
            
            lines.append("")
        
        # Summary
        total_duration = sum(
            r.duration_seconds for r in ckpt_manager.state.stages.values()
        )
        lines.append("## Summary")
        lines.append("")
        lines.append(f"**Total Duration:** {self._format_duration(total_duration)}")
        lines.append(f"**Status:** {ckpt_manager.state.status}")
        lines.append("")
        
        # Files generated
        lines.append("## Output Files")
        lines.append("")
        lines.append("```")
        lines.append(str(self.output_dir) + "/")
        
        for stage_name in CheckpointManager.STAGE_ORDER:
            stage_dir = self.output_dir / f"stage_{CheckpointManager.STAGE_ORDER.index(stage_name) + 1}_{stage_name}"
            if stage_dir.exists():
                lines.append(f"├── {stage_dir.name}/")
                for item in sorted(stage_dir.iterdir())[:5]:
                    lines.append(f"│   ├── {item.name}")
                if len(list(stage_dir.iterdir())) > 5:
                    lines.append(f"│   └── ...")
        
        lines.append("├── pipeline_config.yaml")
        lines.append("├── pipeline_state.json")
        lines.append("└── pipeline_report_*.md")
        lines.append("```")
        lines.append("")
        
        return "\n".join(lines)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
