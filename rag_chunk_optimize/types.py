from dataclasses import dataclass, field

@dataclass
class ChunkConfig:
    strategy: str = "sentence"; size: int = 500; overlap: int = 0

@dataclass
class TrialResult:
    config: ChunkConfig; chunk_count: int = 0; avg_hits: float = 0.0; coverage: float = 0.0; score: float = 0.0

@dataclass
class OptimizeReport:
    trials: list[TrialResult] = field(default_factory=list); best: TrialResult | None = None; total_tested: int = 0
    def summary(self) -> str:
        if not self.best: return "No trials"
        return f"Tested {self.total_tested} configs\nBest: {self.best.config.strategy} size={self.best.config.size} overlap={self.best.config.overlap}\n  Score: {self.best.score:.4f} | Coverage: {self.best.coverage:.1%}"
