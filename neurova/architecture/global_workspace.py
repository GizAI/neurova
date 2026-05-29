from dataclasses import dataclass
from typing import Any, Dict, List, Callable
import time
import uuid
import heapq

@dataclass
class WorkspaceIdea:
    idea_id: str
    source_module: str
    content: Any
    energy: float
    timestamp: float
    metadata: Dict[str, Any]

    def __lt__(self, other):
        # Higher energy comes first in max-heap (implemented as min-heap with negative energy)
        return self.energy > other.energy

class GlobalWorkspace:
    """The central nervous system of the Active Inference Engine.
    Modules subscribe and publish to this workspace. It runs an event loop,
    allowing ideas to compete based on energy (activation).
    """
    def __init__(self, clock_rate_hz: float = 10.0):
        self.clock_rate_hz = clock_rate_hz
        self.buffer: List[WorkspaceIdea] = []
        self.subscribers: List[Callable[[WorkspaceIdea], None]] = []
        self.history: List[WorkspaceIdea] = []
        self.running = False
        
    def publish(self, source_module: str, content: Any, energy: float, metadata: Dict[str, Any] = None):
        idea = WorkspaceIdea(
            idea_id=f"idea_{uuid.uuid4().hex[:8]}",
            source_module=source_module,
            content=content,
            energy=energy,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        heapq.heappush(self.buffer, idea)
        
    def subscribe(self, callback: Callable[[WorkspaceIdea], None]):
        self.subscribers.append(callback)
        
    def tick(self) -> WorkspaceIdea:
        """Process one clock cycle of the global workspace."""
        if not self.buffer:
            return None
            
        # The idea with the highest energy wins the workspace this tick
        winning_idea = heapq.heappop(self.buffer)
        self.history.append(winning_idea)
        
        # Decay remaining ideas
        new_buffer = []
        for idea in self.buffer:
            idea.energy *= 0.9  # Decay factor
            if idea.energy > 0.05:  # Threshold
                new_buffer.append(idea)
        self.buffer = new_buffer
        heapq.heapify(self.buffer)
        
        # Broadcast the winning idea to all modules
        print(f"[WORKSPACE DEBUG] Broadcasting idea from {winning_idea.source_module} to {len(self.subscribers)} subscribers")
        for sub in self.subscribers:
            print(f"[WORKSPACE DEBUG] Calling {sub.__name__}")

        for sub in self.subscribers:
            try:
                sub(winning_idea)
            except Exception as e:
                import traceback
                print(f"Error in subscriber {sub}: {e}")
                traceback.print_exc()

                
        return winning_idea
