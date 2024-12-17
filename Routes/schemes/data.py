from pydantic import BaseModel
from typing import Optional

class ProcessRequest(BaseModel):
    
    chunk_size: Optional[int] = 200
    overlap_size: Optional[int] = 40
    do_reset : Optional[int] = 0